#!/usr/bin/env python3
"""
jaccl_tps_bench.py  --  Production-grade distributed inference benchmark
for MLX + JACCL (RDMA over Thunderbolt) clusters.

Reports:
  1. Cluster topology -- nodes, chips, RAM, RDMA devices
  2. Parallelism strategy -- how the model is sharded
  3. Model loading time + per-node memory
  4. RDMA probe -- sync latency + bandwidth
  5. Prefill tok/s  (prompt processing)
  6. Decode tok/s   (token generation)
  7. TTFT           (time to first token)
  8. Multi-run stats (mean +/- std)
  9. Per-node memory per run via all_gather

Architecture:
  MLX (Apple ML framework)
    -> mlx.distributed (all_sum, all_gather)
        -> JACCL backend (RDMA over Thunderbolt, bypasses TCP/IP)
            -> Tensor Parallelism: model.shard() splits every weight
               matrix column-wise across nodes. Each layer does local
               matmul then all_reduce via RDMA.

Usage:
    .venv/bin/mlx.launch --backend jaccl \
      --hostfile hostfiles/hosts-2node.json \
      --env MLX_METAL_FAST_SYNCH=1 -- \
      scripts/jaccl_tps_bench.py --model ~/models_mlx/Qwen3-8B-4bit

Env vars:
    BENCH_RUNS     Number of timed runs   (default 3)
    BENCH_WARMUP   Number of warmup runs  (default 1)
    BENCH_VERBOSE  Show generated text     (default 0)
"""

import argparse, gc, json, math, os, socket, subprocess, sys, time
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_model

try:
    from mlx_lm import stream_generate
except ImportError:
    from mlx_lm.generate import stream_generate

try:
    from mlx_lm.utils import generate
except ImportError:
    from mlx_lm.generate import generate

# -- Config --
NUM_RUNS   = int(os.environ.get("BENCH_RUNS",    "3"))
NUM_WARMUP = int(os.environ.get("BENCH_WARMUP",  "1"))
VERBOSE    = os.environ.get("BENCH_VERBOSE", "0") == "1"

# -- ANSI --
_TTY = sys.stdout.isatty()
def _c(code, txt): return f"\033[{code}m{txt}\033[0m" if _TTY else str(txt)
BOLD=lambda t:_c("1",t); DIM=lambda t:_c("2",t)
GREEN=lambda t:_c("32",t); YELLOW=lambda t:_c("33",t)
CYAN=lambda t:_c("36",t); RED=lambda t:_c("31",t); MAGENTA=lambda t:_c("35",t)

# ============================================================================
#  Helpers
# ============================================================================
def _gb(b): return b/(1024**3)
def _mb(b): return b/(1024**2)

def _get_mem():
    return (_gb(mx.get_active_memory()), _gb(mx.get_peak_memory()), _gb(mx.get_cache_memory()))

def _reset_peak(): mx.reset_peak_memory()

def _gather_mem(world):
    a, p, c = _get_mem()
    local = mx.array([a, p, c], dtype=mx.float32)
    g = mx.distributed.all_gather(local); mx.eval(g)
    ws = world.size()
    out = []
    for i in range(ws):
        out.append({"active_gb": float(g[i*3]), "peak_gb": float(g[i*3+1]), "cache_gb": float(g[i*3+2])})
    return out

def _sysctl(key):
    try: return subprocess.check_output(["sysctl","-n",key], stderr=subprocess.DEVNULL).decode().strip()
    except: return "unknown"

def _mean(v): return sum(v)/len(v) if v else 0.0
def _std(v):
    if len(v)<2: return 0.0
    m=_mean(v); return math.sqrt(sum((x-m)**2 for x in v)/(len(v)-1))
def _fmt(val, unit="", d=1):
    if abs(val)>=10000: return f"{val:,.{d}f} {unit}".strip()
    return f"{val:.{d}f} {unit}".strip()

# ============================================================================
#  Gather node info from all ranks
# ============================================================================
def _local_info():
    ram = int(_sysctl("hw.memsize") or 0)
    return {
        "hostname": socket.gethostname(),
        "chip": _sysctl("machdep.cpu.brand_string"),
        "total_ram_gb": round(ram/(1024**3), 1),
        "model_id": _sysctl("hw.model"),
        "mlx_version": mx.__version__,
        "mlx_device": str(mx.default_device()),
    }

def _gather_nodes(world):
    info = _local_info()
    def _s2t(s, n=64):
        b=s.encode("utf-8")[:n]; b=b+b"\x00"*(n-len(b))
        return mx.array([int(c) for c in b], dtype=mx.int32)

    h_t = _s2t(info["hostname"]); c_t = _s2t(info["chip"])
    n_t = mx.array([info["total_ram_gb"]], dtype=mx.float32)

    ah = mx.distributed.all_gather(h_t)
    ac = mx.distributed.all_gather(c_t)
    an = mx.distributed.all_gather(n_t)
    mx.eval(ah, ac, an)

    ws = world.size(); nodes = []
    for i in range(ws):
        hb = bytes(int(x) for x in ah[i*64:(i+1)*64].tolist() if int(x)!=0)
        cb = bytes(int(x) for x in ac[i*64:(i+1)*64].tolist() if int(x)!=0)
        nodes.append({
            "rank": i,
            "hostname": hb.decode("utf-8", errors="replace"),
            "chip": cb.decode("utf-8", errors="replace"),
            "total_ram_gb": float(an[i]),
        })
    return nodes, info

# ============================================================================
#  Model info from config.json
# ============================================================================
def _model_info(path):
    info = {"name": Path(path).name}
    cfg_path = Path(path)/"config.json"
    if cfg_path.exists():
        cfg = json.load(open(cfg_path))
        info["arch"]        = cfg.get("architectures",["unknown"])[0]
        info["hidden"]      = cfg.get("hidden_size", 0)
        info["layers"]      = cfg.get("num_hidden_layers", 0)
        info["heads"]       = cfg.get("num_attention_heads", 0)
        info["kv_heads"]    = cfg.get("num_key_value_heads", 0)
        info["ffn"]         = cfg.get("intermediate_size", 0)
        info["vocab"]       = cfg.get("vocab_size", 0)
        info["max_seq"]     = cfg.get("max_position_embeddings", 0)
        q = cfg.get("quantization", cfg.get("quantization_config", {}))
        info["qbits"]       = q.get("bits")
        info["qgroup"]      = q.get("group_size")
    else:
        info["arch"] = "unknown"
    sfs = list(Path(path).glob("*.safetensors"))
    info["disk_gb"] = round(sum(f.stat().st_size for f in sfs)/(1024**3), 2)
    return info

# ============================================================================
#  Hostfile
# ============================================================================
def _load_hostfile():
    for p in [os.environ.get("HOSTFILE",""), "hostfiles/hosts-2node.json", "hosts-2node.json"]:
        if p and os.path.isfile(p):
            return json.load(open(p))
    return None

# ============================================================================
#  Tokenizer wrapper
# ============================================================================
class TokenizerWrapper:
    def __init__(self, t): self._tok = t
    def __getattr__(self, n): return getattr(self._tok, n)
    def encode(self, text, **kw): return self._tok.encode(text)
    def decode(self, tokens, **kw): return self._tok.decode(tokens)

def _load_custom_tok(path):
    p = Path(path); sys.path.insert(0, str(p))
    for tf in p.glob("tokenization_*.py"):
        mod = __import__(tf.stem)
        for attr in dir(mod):
            cls = getattr(mod, attr)
            if isinstance(cls, type) and hasattr(cls, "from_pretrained"):
                try: return TokenizerWrapper(cls.from_pretrained(p))
                except: continue
    raise RuntimeError(f"Could not load tokenizer from {path}")

# ============================================================================
#  Sharded model loading
# ============================================================================
def sharded_load_with_fallback(repo):
    try:
        from mlx_lm.utils import sharded_load
        model, tok = sharded_load(repo)
        has_tp = hasattr(model, "shard")
        has_pp = hasattr(getattr(model, "model", None), "pipeline")
        strategy = "Tensor Parallelism" if has_tp else ("Pipeline Parallelism" if has_pp else "Auto")
        return model, tok, strategy
    except Exception as e:
        if "tokenizer" not in str(e).lower() and "NoneType" not in str(e): raise

    tok = _load_custom_tok(repo)
    model, _ = load_model(Path(repo), lazy=True, strict=False)
    tg = mx.distributed.init()
    strategy = "Tensor Parallelism (fallback)"
    if hasattr(model, "shard"): model.shard(tg)
    elif hasattr(model.model, "pipeline"): model.model.pipeline(tg); strategy = "Pipeline (fallback)"
    mx.eval(model.parameters())
    mx.eval(mx.distributed.all_sum(mx.array(1.0), stream=mx.cpu))
    return model, tok, strategy

# ============================================================================
#  RDMA probes
# ============================================================================
def rdma_latency_probe(world, rounds=20):
    x = mx.ones((1,), dtype=mx.float32); mx.eval(x)
    for _ in range(5): mx.eval(mx.distributed.all_sum(x))  # warmup
    times = []
    for _ in range(rounds):
        t0=time.perf_counter(); mx.eval(mx.distributed.all_sum(x)); t1=time.perf_counter()
        times.append((t1-t0)*1e6)
    return {"mean_us":_mean(times), "min_us":min(times), "max_us":max(times), "std_us":_std(times)}

def rdma_bw_probe(world, num_elems=16_777_216, rounds=5):
    t = mx.random.normal((num_elems,), dtype=mx.float32); mx.eval(t)
    sz = num_elems * 4
    for _ in range(2): mx.eval(mx.distributed.all_sum(t))
    times = []
    for _ in range(rounds):
        t0=time.perf_counter(); r=mx.distributed.all_sum(t); mx.eval(r); t1=time.perf_counter()
        times.append(t1-t0); del r
    del t; gc.collect(); mx.clear_cache()
    avg = _mean(times)
    bw = (sz/(1024**3))/avg if avg>0 else 0
    return {"size_mb": round(sz/(1024**2)), "avg_sec": avg, "bw_gbs": round(bw, 2)}

# ============================================================================
#  Benchmark run (stream_generate)
# ============================================================================
def bench_run(model, tok, prompt, max_tokens, world):
    _reset_peak(); gc.collect(); mx.clear_cache()
    ttft = None; final = None; text_parts = []
    t0 = time.perf_counter()
    for resp in stream_generate(model, tok, prompt, max_tokens=max_tokens):
        if ttft is None and resp.token is not None:
            ttft = (time.perf_counter()-t0)*1000
        if resp.text: text_parts.append(resp.text)
        final = resp
    t1 = time.perf_counter()
    mem = _gather_mem(world)
    return {
        "prompt_tokens": final.prompt_tokens if final else 0,
        "prompt_tps":    final.prompt_tps    if final else 0.0,
        "gen_tokens":    final.generation_tokens if final else 0,
        "gen_tps":       final.generation_tps if final else 0.0,
        "peak_mem_gb":   final.peak_memory   if final else 0.0,
        "ttft_ms":       ttft or 0.0,
        "total_sec":     t1-t0,
        "text":          "".join(text_parts),
        "finish":        final.finish_reason if final else "?",
        "node_mem":      mem,
    }

# ============================================================================
#  Printers (rank 0 only)
# ============================================================================
def pr_banner():
    s = "=" * 66
    print(f"\n  {BOLD(s)}")
    print(f"  {BOLD('  MLX-JACCL Distributed Inference Benchmark')}")
    print(f"  {BOLD(s)}\n")

def pr_topology(nodes, hostfile):
    print(f"  {BOLD('-- Cluster Topology')} {'_'*46}\n")
    rdma_devs = {}
    if hostfile:
        for i, e in enumerate(hostfile):
            devs = [d for d in e.get("rdma",[]) if d]; rdma_devs[i] = devs[0] if devs else "-"
    for n in nodes:
        r = n["rank"]; dev = rdma_devs.get(r, "-")
        marker = CYAN("*") if r==0 else GREEN("*")
        role = "coordinator" if r==0 else "worker"
        print(f"  {marker} rank {r}  {BOLD(n['hostname'][:20]):<28s}"
              f"{n['chip']:<18s}{n['total_ram_gb']:.0f} GB   {DIM(dev)}")
        print(f"          {DIM(role)}")
    if len(nodes)==2:
        a, b = nodes[0]["hostname"][:12], nodes[1]["hostname"][:12]
        print(f"\n    {CYAN(a)}  <{'='*4} RDMA (Thunderbolt) {'='*4}>  {GREEN(b)}")
    print()

def pr_stack(strategy, info, mi, ws):
    print(f"  {BOLD('-- Stack')} {'_'*56}\n")
    print(f"  {DIM('Framework')}       MLX {info['mlx_version']}")
    print(f"  {DIM('Device')}          {info['mlx_device']}")
    print(f"  {DIM('Backend')}         JACCL (RDMA over Thunderbolt)")
    print(f"  {DIM('Parallelism')}     {GREEN(strategy)}")
    print()
    print(f"  {DIM('How Tensor Parallelism works in this cluster:')}")
    print(f"  {DIM('  * model.shard() splits every weight matrix W across')} {ws} {DIM('nodes')}")
    print(f"  {DIM('  * Each node holds columns W[:, start:end] (column-wise split)')}")
    print(f"  {DIM('  * Forward: local matmul -> all_reduce(sum) via RDMA each layer')}")
    print(f"  {DIM('  * This is NOT pipeline parallelism (sequential layer split)')}")
    nl = mi.get("layers","?")
    print(f"  {DIM(f'  * {nl} layers x all_reduce = high RDMA traffic per token')}")
    print()

def pr_model(mi, ws):
    print(f"  {BOLD('-- Model')} {'_'*56}\n")
    print(f"  {DIM('Name')}            {CYAN(mi['name'])}")
    print(f"  {DIM('Architecture')}    {mi.get('arch','?')}")
    h=mi.get("hidden",0); nl=mi.get("layers",0); nh=mi.get("heads",0)
    nkv=mi.get("kv_heads",0); ffn=mi.get("ffn",0); v=mi.get("vocab",0)
    if h:  print(f"  {DIM('Hidden size')}     {h}   {DIM(f'({nh} attn heads, {nkv} KV heads)')}")
    if nl: print(f"  {DIM('Layers')}          {nl}   {DIM(f'(all layers on all nodes with TP)')}")
    if ffn: print(f"  {DIM('FFN size')}        {ffn}")
    if v:  print(f"  {DIM('Vocab')}           {v:,}")
    qb=mi.get("qbits"); qg=mi.get("qgroup")
    if qb: print(f"  {DIM('Quantization')}   {qb}-bit  (group size {qg})")
    d=mi.get("disk_gb",0)
    if d:  print(f"  {DIM('Size on disk')}    {d:.2f} GB total  ->  ~{d/ws:.2f} GB weights/node (sharded)")
    print()

def pr_load(t, ws):
    print(f"  {BOLD('-- Model Load')} {'_'*51}\n")
    print(f"  {DIM('Load time')}       {GREEN(_fmt(t,'s'))}  (sharded across {ws} nodes)")
    print()

def pr_mem_table(label, nodes, mem):
    print(f"  {BOLD(f'-- Memory ({label})')} {'_'*(50-len(label))}\n")
    for i, n in enumerate(nodes):
        m = mem[i]; ram = n["total_ram_gb"]
        pct = (m["active_gb"]/ram*100) if ram>0 else 0
        bar_len = int(pct*30/100)
        bar = GREEN("=" * bar_len) + DIM("-" * (30-bar_len))
        print(f"  rank {i}  {n['hostname'][:16]:<18s}"
              f"{_fmt(m['active_gb'],'GB')} active   "
              f"{_fmt(m['peak_gb'],'GB')} peak   "
              f"{DIM(f'/ {ram:.0f} GB')}")
        print(f"        {'':18s}[{bar}] {pct:.1f}%")
    print()

def pr_rdma(lat, bw):
    print(f"  {BOLD('-- RDMA Probe')} {'_'*51}\n")
    print(f"  {DIM('Sync latency')}    {GREEN(_fmt(lat['mean_us'],'us'))}"
          f"  (min {_fmt(lat['min_us'],'us')}, max {_fmt(lat['max_us'],'us')},"
          f" std {_fmt(lat['std_us'],'us')})")
    print(f"  {DIM('Bandwidth')}       {GREEN(_fmt(bw['bw_gbs'],'GB/s'))}"
          f"  ({bw['size_mb']} MB tensor)")
    print()

def pr_run(num, total, r, nodes, warmup=False):
    label = f"Warmup {num}" if warmup else f"Run {num}/{total}"
    print(f"  {BOLD(f'-- {label}')} {'_'*(56-len(label))}")
    print(f"  {DIM('Prompt')}     {r['prompt_tokens']:<8} tok    "
          f"{DIM('Prefill')}   {CYAN(_fmt(r['prompt_tps'],'tok/s'))}")
    print(f"  {DIM('Generated')}  {r['gen_tokens']:<8} tok    "
          f"{DIM('Decode')}    {CYAN(_fmt(r['gen_tps'],'tok/s'))}")
    print(f"  {DIM('TTFT')}       {_fmt(r['ttft_ms'],'ms'):<16}"
          f"{DIM('Total')}     {_fmt(r['total_sec'],'s')}")
    mem = r.get("node_mem",[])
    if mem:
        parts = []
        for i, m in enumerate(mem):
            nm = nodes[i]["hostname"][:10] if i<len(nodes) else f"rank{i}"
            parts.append(f"{nm}: {_fmt(m['peak_gb'],'GB')}")
        print(f"  {DIM('Peak mem')}   {DIM(' | ').join(parts)}")
    if VERBOSE and r.get("text"):
        preview = r["text"][:300].replace("\n"," ")
        if len(r["text"])>300: preview += "..."
        print(f"  {DIM('Output:')}    {preview}")
    print()

def pr_summary(results, nodes, ws):
    print(f"  {BOLD('-- Summary')} {'_'*54}\n")
    dv=[r["gen_tps"]    for r in results]
    pv=[r["prompt_tps"] for r in results]
    tv=[r["ttft_ms"]    for r in results]
    sv=[r["total_sec"]  for r in results]
    max_peak = {}
    for r in results:
        for i, m in enumerate(r.get("node_mem",[])):
            max_peak[i] = max(max_peak.get(i,0), m["peak_gb"])
    print(f"  {BOLD('Decode')}          {GREEN(_fmt(_mean(dv),'tok/s'))}  +/- {_fmt(_std(dv),'tok/s')}")
    print(f"  {BOLD('Prefill')}         {GREEN(_fmt(_mean(pv),'tok/s'))}  +/- {_fmt(_std(pv),'tok/s')}")
    print(f"  {BOLD('TTFT')}            {GREEN(_fmt(_mean(tv),'ms'))}  +/- {_fmt(_std(tv),'ms')}")
    print(f"  {BOLD('Total')}           {_fmt(_mean(sv),'s')}  +/- {_fmt(_std(sv),'s')}")
    print()
    for i in sorted(max_peak):
        nm = nodes[i]["hostname"][:16] if i<len(nodes) else f"rank {i}"
        ram = nodes[i]["total_ram_gb"] if i<len(nodes) else 0
        pk = max_peak[i]; pct = (pk/ram*100) if ram>0 else 0
        print(f"  {DIM(f'Peak mem rank {i}')}  {nm}: {GREEN(_fmt(pk,'GB'))}  / {ram:.0f} GB  ({pct:.1f}%)")
    print()
    sj = {
        "decode_tps_mean": round(_mean(dv),2), "decode_tps_std": round(_std(dv),2),
        "prefill_tps_mean": round(_mean(pv),2), "prefill_tps_std": round(_std(pv),2),
        "ttft_ms_mean": round(_mean(tv),2), "ttft_ms_std": round(_std(tv),2),
        "total_sec_mean": round(_mean(sv),3), "world_size": ws, "runs": len(results),
        "peak_mem_per_node_gb": {str(i):round(v,3) for i,v in max_peak.items()},
    }
    print(f"  {DIM('JSON:')} {json.dumps(sj)}")
    print(f"\n  {BOLD('='*66)}\n")

# ============================================================================
#  Main
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description="MLX-JACCL Distributed Inference Benchmark")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="Explain how RDMA over Thunderbolt enables distributed ML inference on Apple Silicon.")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--runs",   type=int, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    args = ap.parse_args()
    num_runs   = args.runs   or NUM_RUNS
    num_warmup = args.warmup or NUM_WARMUP

    # -- Distributed init --
    world = mx.distributed.init()
    rank  = world.rank()

    # -- Gather node info from all ranks --
    nodes, local = _gather_nodes(world)
    mi       = _model_info(args.model)
    hostfile = _load_hostfile()

    # -- Load model (timed, all ranks) --
    _reset_peak()
    t0 = time.perf_counter()
    model, tok, strategy = sharded_load_with_fallback(args.model)
    load_time = time.perf_counter() - t0

    # -- Post-load memory (all ranks -> rank 0) --
    mem_load = _gather_mem(world)

    # -- RDMA probes (all ranks participate) --
    lat = rdma_latency_probe(world, rounds=20)
    bw  = rdma_bw_probe(world, num_elems=16_777_216, rounds=5)

    # -- Print everything (rank 0) --
    if rank == 0:
        pr_banner()
        pr_topology(nodes, hostfile)
        pr_stack(strategy, local, mi, world.size())
        pr_model(mi, world.size())
        pr_load(load_time, world.size())
        pr_mem_table("after model load", nodes, mem_load)
        pr_rdma(lat, bw)

    # -- Warmup --
    for w in range(num_warmup):
        r = bench_run(model, tok, args.prompt, args.max_tokens, world)
        if rank == 0: pr_run(w+1, num_warmup, r, nodes, warmup=True)
        gc.collect(); mx.clear_cache()

    # -- Timed runs --
    results = []
    for i in range(num_runs):
        r = bench_run(model, tok, args.prompt, args.max_tokens, world)
        results.append(r)
        if rank == 0: pr_run(i+1, num_runs, r, nodes)
        gc.collect(); mx.clear_cache()

    # -- Summary --
    if rank == 0:
        pr_summary(results, nodes, world.size())

if __name__ == "__main__":
    main()
