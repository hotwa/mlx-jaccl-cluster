import argparse, time, sys
from pathlib import Path
import mlx.core as mx
from mlx_lm.utils import load_model

# generate() import differs across mlx-lm branches
try:
    from mlx_lm.utils import generate
except Exception:
    from mlx_lm.generate import generate


class TokenizerWrapper:
    """Wrapper to handle encode kwargs that some custom tokenizers don't support."""
    def __init__(self, tokenizer):
        self._tok = tokenizer

    def __getattr__(self, name):
        return getattr(self._tok, name)

    def encode(self, text, **kwargs):
        return self._tok.encode(text)

    def decode(self, tokens, **kwargs):
        return self._tok.decode(tokens)


def load_custom_tokenizer(model_path):
    """Load custom tokenizer directly when AutoTokenizer fails."""
    model_path = Path(model_path)
    sys.path.insert(0, str(model_path))

    # Try common custom tokenizer patterns
    for tok_file in model_path.glob("tokenization_*.py"):
        module_name = tok_file.stem
        mod = __import__(module_name)
        for attr in dir(mod):
            cls = getattr(mod, attr)
            if isinstance(cls, type) and hasattr(cls, 'from_pretrained'):
                try:
                    tok = cls.from_pretrained(model_path)
                    return TokenizerWrapper(tok)
                except:
                    continue
    raise RuntimeError(f"Could not load custom tokenizer from {model_path}")


def sharded_load_with_fallback(repo):
    """Load model with fallback for custom tokenizers."""
    model_path = Path(repo)

    # Try standard sharded_load first
    try:
        from mlx_lm.utils import sharded_load
        return sharded_load(repo)
    except Exception as e:
        if "tokenizer" not in str(e).lower() and "NoneType" not in str(e):
            raise

    # Fallback: load model and tokenizer separately
    tok = load_custom_tokenizer(model_path)

    model, config = load_model(model_path, lazy=True, strict=False)

    tensor_group = mx.distributed.init()
    if hasattr(model, "shard"):
        model.shard(tensor_group)

    mx.eval(model.parameters())

    # Sync
    x = mx.zeros((1,))
    mx.eval(mx.distributed.all_sum(x))

    return model, tok


def _token_count(tok, text: str) -> int:
    return len(tok.encode(text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    world = mx.distributed.init()
    rank = world.rank()

    model, tok = sharded_load_with_fallback(args.model)

    # warmup
    _ = generate(model, tok, "hi", max_tokens=8)
    mx.eval()

    t0 = time.time()
    out = generate(model, tok, args.prompt, max_tokens=args.max_tokens)
    mx.eval()
    t1 = time.time()

    prompt_tokens = _token_count(tok, args.prompt)
    out_tokens = _token_count(tok, out)
    gen_tokens = max(out_tokens - prompt_tokens, 1) if out.startswith(args.prompt) else max(out_tokens, 1)

    secs = max(t1 - t0, 1e-9)
    if rank == 0:
        print("==========")
        print(f"model={args.model}")
        print(f"world_size={world.size()}")
        print(f"prompt_tokens={prompt_tokens}")
        print(f"gen_tokens={gen_tokens}")
        print(f"seconds={secs:.3f}")
        print(f"tokens_per_sec={gen_tokens/secs:.3f}")

if __name__ == "__main__":
    main()
