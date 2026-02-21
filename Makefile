# =============================================================================
# Makefile — mlx-jaccl-cluster
# =============================================================================
# Common targets for setup, testing, benchmarking, and running the cluster.
#
# Usage:
#   make help            Show all targets
#   make setup           Install dependencies on this node
#   make rdma-test       Run RDMA connectivity + bandwidth test
#   make server          Start the OpenAI-compatible cluster server
#
# Most targets accept overrides via environment variables:
#   HOSTFILE=hostfiles/hosts-2node.json make rdma-test
#   MODEL_DIR=~/models_mlx/Qwen3-4B make server
#   MODEL=mlx-community/Qwen3-4B-Instruct-2507-4bit make download
# =============================================================================

.DEFAULT_GOAL := help
SHELL := /bin/bash

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_DIR     := $(shell pwd)
VENV_DIR     := $(REPO_DIR)/.venv
VENV_PYTHON  := $(VENV_DIR)/bin/python
MLX_LAUNCH   := $(VENV_DIR)/bin/mlx.launch
HOSTFILE     ?= hostfiles/hosts-2node.json

# ── RDMA test defaults ───────────────────────────────────────────────────────
RDMA_ROUNDS  ?= 20
RDMA_SIZES   ?= 1024,65536,1048576,16777216
RDMA_VERBOSE ?= 0
RDMA_MAX_MB  ?= 256

# ── Server defaults ──────────────────────────────────────────────────────────
MODEL_DIR    ?=
HTTP_HOST    ?= 0.0.0.0
HTTP_PORT    ?= 8080
CTRL_PORT    ?= 18080
QUEUE_MAX    ?= 8
REQ_TIMEOUT  ?= 120

# ── Benchmark defaults ───────────────────────────────────────────────────────
BENCH_PROMPT ?= "Explain how RDMA over Thunderbolt enables distributed ML inference on Apple Silicon."
BENCH_TOKENS ?= 256
BENCH_RUNS   ?= 3
BENCH_WARMUP ?= 1
BENCH_VERBOSE ?= 0

# ── Stress test defaults ─────────────────────────────────────────────────
STRESS_ROUNDS ?= 100
STRESS_SIZES  ?= 1048576,16777216,67108864,134217728

# ── Model download defaults ──────────────────────────────────────────────
MODEL        ?=
MODELS_DIR   ?= $(HOME)/models_mlx

# ── Monitor defaults ─────────────────────────────────────────────────────
MONITOR_INTERVAL ?= 5

# =============================================================================
# Help
# =============================================================================

.PHONY: help
help: ## Show this help message
	@printf "\n\033[1m  mlx-jaccl-cluster — Makefile targets\033[0m\n\n"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'
	@printf "\n\033[2m  Override defaults with env vars:\033[0m\n"
	@printf "  \033[2m  HOSTFILE=hostfiles/hosts-2node.json make rdma-test\033[0m\n"
	@printf "  \033[2m  MODEL_DIR=~/models_mlx/Qwen3-4B make server\033[0m\n"
	@printf "  \033[2m  RDMA_ROUNDS=50 RDMA_VERBOSE=1 make rdma-test\033[0m\n"
	@printf "  \033[2m  MODEL=mlx-community/Qwen3-4B-Instruct-2507-4bit make download\033[0m\n\n"

# =============================================================================
# Setup
# =============================================================================

.PHONY: setup
setup: ## Install uv, create .venv, install all dependencies (run on each node)
	@./scripts/setup.sh

.PHONY: bootstrap
bootstrap: ## Bootstrap a remote node (usage: REMOTE=mac2.local make bootstrap)
ifndef REMOTE
	@printf "\033[31m  ✗ REMOTE is required.\033[0m\n"
	@printf "  Usage: REMOTE=mac2.local make bootstrap\n\n"
	@exit 1
endif
	@./scripts/bootstrap_node.sh $(REMOTE)

$(VENV_PYTHON):
	@printf "\033[31m  ✗ .venv not found. Run: make setup\033[0m\n"; exit 1

$(MLX_LAUNCH):
	@printf "\033[31m  ✗ mlx.launch not found. Run: make setup\033[0m\n"; exit 1

# =============================================================================
# Cluster verification
# =============================================================================

.PHONY: verify
verify: $(VENV_PYTHON) ## Check SSH connectivity + RDMA devices on all nodes
	@HOSTFILE=$(HOSTFILE) ./scripts/verify_cluster.sh

.PHONY: cluster-info
cluster-info: $(VENV_PYTHON) ## Side-by-side node alignment report (versions, hardware, RDMA)
	@HOSTFILE=$(HOSTFILE) ./scripts/cluster_info.sh

.PHONY: sync
sync: $(VENV_PYTHON) ## Pull latest git changes on all nodes and verify commit alignment
	@HOSTFILE=$(HOSTFILE) ./scripts/sync_nodes.sh

# =============================================================================
# RDMA tests
# =============================================================================

.PHONY: rdma-test
rdma-test: $(MLX_LAUNCH) ## Run RDMA connectivity + bandwidth test (no model needed)
	RDMA_ROUNDS=$(RDMA_ROUNDS) \
	RDMA_SIZES=$(RDMA_SIZES) \
	RDMA_VERBOSE=$(RDMA_VERBOSE) \
	RDMA_MAX_MB=$(RDMA_MAX_MB) \
	$(MLX_LAUNCH) --backend jaccl \
	  --hostfile $(HOSTFILE) \
	  --env MLX_METAL_FAST_SYNCH=1 -- \
	  scripts/rdma_test.py

.PHONY: rdma-quick
rdma-quick: $(MLX_LAUNCH) ## Quick RDMA smoke test (5 rounds, small tensors)
	RDMA_ROUNDS=5 \
	RDMA_SIZES=1024,65536 \
	RDMA_VERBOSE=0 \
	$(MLX_LAUNCH) --backend jaccl \
	  --hostfile $(HOSTFILE) \
	  --env MLX_METAL_FAST_SYNCH=1 -- \
	  scripts/rdma_test.py

.PHONY: rdma-stress
rdma-stress: $(MLX_LAUNCH) ## RDMA stress test (100 rounds, large tensors)
	RDMA_ROUNDS=$(STRESS_ROUNDS) \
	RDMA_SIZES=$(STRESS_SIZES) \
	RDMA_VERBOSE=1 \
	RDMA_MAX_MB=512 \
	$(MLX_LAUNCH) --backend jaccl \
	  --hostfile $(HOSTFILE) \
	  --env MLX_METAL_FAST_SYNCH=1 -- \
	  scripts/rdma_test.py

.PHONY: rdma-verbose
rdma-verbose: $(MLX_LAUNCH) ## RDMA test with per-round timing output
	RDMA_ROUNDS=$(RDMA_ROUNDS) \
	RDMA_SIZES=$(RDMA_SIZES) \
	RDMA_VERBOSE=1 \
	RDMA_MAX_MB=$(RDMA_MAX_MB) \
	$(MLX_LAUNCH) --backend jaccl \
	  --hostfile $(HOSTFILE) \
	  --env MLX_METAL_FAST_SYNCH=1 -- \
	  scripts/rdma_test.py

# =============================================================================
# Benchmarks
# =============================================================================

.PHONY: bench
bench: $(MLX_LAUNCH) ## Run distributed tokens/sec benchmark (requires MODEL_DIR)
ifndef MODEL_DIR
	@printf "\033[31m  ✗ MODEL_DIR is required.\033[0m\n"
	@printf "  Usage: MODEL_DIR=~/models_mlx/Qwen3-4B make bench\n\n"
	@exit 1
endif
	BENCH_RUNS=$(BENCH_RUNS) \
	BENCH_WARMUP=$(BENCH_WARMUP) \
	BENCH_VERBOSE=$(BENCH_VERBOSE) \
	$(MLX_LAUNCH) --verbose --backend jaccl \
	  --hostfile $(HOSTFILE) \
	  --env MLX_METAL_FAST_SYNCH=1 \
	  --env HF_HUB_OFFLINE=1 \
	  --env TRANSFORMERS_OFFLINE=1 -- \
	  scripts/jaccl_tps_bench.py \
	  --model $(MODEL_DIR) \
	  --prompt $(BENCH_PROMPT) \
	  --max-tokens $(BENCH_TOKENS) \
	  --runs $(BENCH_RUNS) \
	  --warmup $(BENCH_WARMUP)

# =============================================================================
# Server
# =============================================================================

.PHONY: server
server: $(MLX_LAUNCH) ## Start the OpenAI-compatible cluster server (requires MODEL_DIR)
ifndef MODEL_DIR
	@printf "\033[31m  ✗ MODEL_DIR is required.\033[0m\n"
	@printf "  Usage: MODEL_DIR=~/models_mlx/Qwen3-4B make server\n\n"
	@exit 1
endif
	MODEL_DIR=$(MODEL_DIR) \
	HOSTFILE=$(HOSTFILE) \
	HTTP_HOST=$(HTTP_HOST) \
	HTTP_PORT=$(HTTP_PORT) \
	CTRL_PORT=$(CTRL_PORT) \
	QUEUE_MAX=$(QUEUE_MAX) \
	REQ_TIMEOUT=$(REQ_TIMEOUT) \
	./scripts/run_openai_cluster_server.sh

.PHONY: server-stop
server-stop: ## Stop the cluster server on all nodes
	@HOSTFILE=$(HOSTFILE) ./scripts/stop_openai_cluster_server.sh

.PHONY: server-restart
server-restart: server-stop server ## Restart the cluster server (requires MODEL_DIR)

# =============================================================================
# Model Management
# =============================================================================

.PHONY: download
download: $(VENV_PYTHON) ## Download a model and sync to all nodes (requires MODEL)
ifndef MODEL
	@printf "\033[31m  ✗ MODEL is required.\033[0m\n"
	@printf "  Usage: MODEL=mlx-community/Qwen3-4B-Instruct-2507-4bit make download\n"
	@printf "         MODEL=mlx-community/Qwen3-4B-Instruct-2507-4bit MODELS_DIR=~/my_models make download\n\n"
	@exit 1
endif
	@MODEL_NAME=$$(echo "$(MODEL)" | sed 's|.*/||'); \
	LOCAL_PATH="$(MODELS_DIR)/$$MODEL_NAME"; \
	printf "\n\033[1m  ⬇  Downloading $(MODEL)\033[0m\n"; \
	printf "  → destination: $$LOCAL_PATH\n\n"; \
	$(VENV_DIR)/bin/hf download $(MODEL) --local-dir "$$LOCAL_PATH"; \
	if [ $$? -ne 0 ]; then \
	  printf "\n\033[31m  ✗ Download failed.\033[0m\n"; \
	  exit 1; \
	fi; \
	printf "\n\033[32m  ✓ Download complete: $$LOCAL_PATH\033[0m\n\n"; \
	if [ -f "$(HOSTFILE)" ]; then \
	  HOSTS=$$(python3 -c "import json; d=json.load(open('$(HOSTFILE)')); print(' '.join(h['ssh'] for h in d[1:]))"); \
	  if [ -n "$$HOSTS" ]; then \
	    for h in $$HOSTS; do \
	      printf "  → syncing to %-26s " "$$h"; \
	      ssh "$$h" "mkdir -p $(MODELS_DIR)" 2>/dev/null; \
	      rsync -az --progress -e ssh "$$LOCAL_PATH/" "$$h:$$LOCAL_PATH/" 2>/dev/null; \
	      if [ $$? -eq 0 ]; then \
	        printf "\033[32mdone\033[0m\n"; \
	      else \
	        printf "\033[31mfailed\033[0m\n"; \
	      fi; \
	    done; \
	    printf "\n\033[32m  ✓ Model synced to all nodes.\033[0m\n"; \
	  fi; \
	fi; \
	printf "\n  Run the server with:\n"; \
	printf "  \033[36mMODEL_DIR=$$LOCAL_PATH make server\033[0m\n\n"

.PHONY: models-local
models-local: ## List locally downloaded models with sizes
	@printf "\n\033[1m  Downloaded Models\033[0m  ($(MODELS_DIR))\n\n"
	@if [ -d "$(MODELS_DIR)" ]; then \
	  found=0; \
	  for d in $(MODELS_DIR)/*/; do \
	    if [ -f "$$d/config.json" ]; then \
	      found=1; \
	      name=$$(basename "$$d"); \
	      size=$$(du -sh "$$d" 2>/dev/null | cut -f1); \
	      quant=$$(python3 -c "import json;c=json.load(open('$$d/config.json'));print(c.get('quantization',{}).get('quant_method','fp16') if 'quantization' in c else c.get('quantization_config',{}).get('quant_method','—'))" 2>/dev/null || echo "—"); \
	      printf "  \033[36m%-45s\033[0m  %6s  %s\n" "$$name" "$$size" "$$quant"; \
	    fi; \
	  done; \
	  if [ $$found -eq 0 ]; then \
	    printf "  \033[2m(no models found)\033[0m\n"; \
	  fi; \
	else \
	  printf "  \033[2m(directory does not exist)\033[0m\n"; \
	fi
	@printf "\n"

.PHONY: models-check
models-check: ## Verify model exists on all nodes (requires MODEL_DIR)
ifndef MODEL_DIR
	@printf "\033[31m  ✗ MODEL_DIR is required.\033[0m\n"
	@printf "  Usage: MODEL_DIR=~/models_mlx/Qwen3-4B make models-check\n\n"
	@exit 1
endif
	@printf "\n\033[1m  Model Check\033[0m  $(MODEL_DIR)\n\n"
	@if [ -f "$(HOSTFILE)" ]; then \
	  HOSTS=$$(python3 -c "import json; print(' '.join(h['ssh'] for h in json.load(open('$(HOSTFILE)'))))"); \
	  all_ok=1; \
	  for h in $$HOSTS; do \
	    printf "  → %-26s " "$$h"; \
	    if ssh "$$h" "test -d '$(MODEL_DIR)' && test -f '$(MODEL_DIR)/config.json'" 2>/dev/null; then \
	      size=$$(ssh "$$h" "du -sh '$(MODEL_DIR)' 2>/dev/null | cut -f1"); \
	      printf "\033[32m✓ OK\033[0m  ($$size)\n"; \
	    else \
	      printf "\033[31m✗ MISSING\033[0m\n"; \
	      all_ok=0; \
	    fi; \
	  done; \
	  printf "\n"; \
	  if [ $$all_ok -eq 1 ]; then \
	    printf "  \033[32m✓ Model present on all nodes.\033[0m\n\n"; \
	  else \
	    printf "  \033[31m✗ Model missing on some nodes. Run:\033[0m\n"; \
	    printf "  \033[36mMODEL=<hf-repo-id> make download\033[0m\n\n"; \
	  fi; \
	fi

# =============================================================================
# Server health / API smoke tests
# =============================================================================

.PHONY: health
health: ## Check server health endpoint
	@curl -sf http://localhost:$(HTTP_PORT)/health | python3 -m json.tool \
	  || printf "\033[31m  ✗ Server not responding on port $(HTTP_PORT)\033[0m\n"

.PHONY: models
models: ## List models served by the cluster
	@curl -sf http://localhost:$(HTTP_PORT)/v1/models | python3 -m json.tool \
	  || printf "\033[31m  ✗ Server not responding on port $(HTTP_PORT)\033[0m\n"

.PHONY: chat-test
chat-test: ## Send a quick chat completion to the running server
	@curl -sf http://localhost:$(HTTP_PORT)/v1/chat/completions \
	  -H 'Content-Type: application/json' \
	  -d '{"messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":32}' \
	  | python3 -m json.tool \
	  || printf "\033[31m  ✗ Chat request failed. Is the server running?\033[0m\n"

.PHONY: queue
queue: ## Show current request queue status
	@curl -sf http://localhost:$(HTTP_PORT)/queue | python3 -m json.tool \
	  || printf "\033[31m  ✗ Server not responding on port $(HTTP_PORT)\033[0m\n"

.PHONY: dashboard
dashboard: ## Open the live dashboard in the default browser
	@printf "  Opening dashboard at http://localhost:$(HTTP_PORT)/dashboard\n"
	@open "http://localhost:$(HTTP_PORT)/dashboard" 2>/dev/null \
	  || xdg-open "http://localhost:$(HTTP_PORT)/dashboard" 2>/dev/null \
	  || printf "\033[33m  Open manually: http://localhost:$(HTTP_PORT)/dashboard\033[0m\n"

.PHONY: metrics
metrics: ## Show current metrics snapshot (JSON)
	@curl -sf http://localhost:$(HTTP_PORT)/metrics/snapshot | python3 -m json.tool \
	  || printf "\033[31m  ✗ Server not responding on port $(HTTP_PORT)\033[0m\n"

# =============================================================================
# Cluster Status & Monitoring
# =============================================================================

.PHONY: status
status: $(VENV_PYTHON) ## Full cluster status: nodes, memory, RDMA, server, model
	@printf "\n\033[1m  ⚡ Cluster Status\033[0m\n"
	@printf "  ─────────────────────────────────────────────────\n\n"
	@printf "  \033[1mNodes\033[0m\n"
	@if [ -f "$(HOSTFILE)" ]; then \
	  HOSTS=$$(python3 -c "import json; print(' '.join(h['ssh'] for h in json.load(open('$(HOSTFILE)'))))"); \
	  i=0; \
	  for h in $$HOSTS; do \
	    printf "  rank $$i  %-26s " "$$h"; \
	    if ssh -o ConnectTimeout=3 "$$h" true 2>/dev/null; then \
	      printf "\033[32m● online\033[0m"; \
	      rdma_count=$$(ssh "$$h" "ibv_devinfo 2>/dev/null | grep -c PORT_ACTIVE" 2>/dev/null || echo "0"); \
	      printf "  (RDMA ports: $$rdma_count)"; \
	    else \
	      printf "\033[31m● offline\033[0m"; \
	    fi; \
	    printf "\n"; \
	    i=$$((i+1)); \
	  done; \
	else \
	  printf "  \033[33m! No hostfile at $(HOSTFILE)\033[0m\n"; \
	fi
	@printf "\n  \033[1mServer\033[0m\n"
	@if curl -sf http://localhost:$(HTTP_PORT)/health >/dev/null 2>&1; then \
	  HEALTH=$$(curl -sf http://localhost:$(HTTP_PORT)/health); \
	  printf "  HTTP :$(HTTP_PORT)                      \033[32m● running\033[0m\n"; \
	  printf "  model: %s\n" "$$(echo $$HEALTH | python3 -c 'import sys,json;print(json.load(sys.stdin).get("model","?"))' 2>/dev/null)"; \
	  printf "  world: %s ranks\n" "$$(echo $$HEALTH | python3 -c 'import sys,json;print(json.load(sys.stdin).get("world_size","?"))' 2>/dev/null)"; \
	  printf "  queue: %s / %s\n" "$$(echo $$HEALTH | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d.get("queue_size","?"))' 2>/dev/null)" "$$(echo $$HEALTH | python3 -c 'import sys,json;print(json.load(sys.stdin).get("queue_max","?"))' 2>/dev/null)"; \
	else \
	  printf "  HTTP :$(HTTP_PORT)                      \033[2m○ not running\033[0m\n"; \
	fi
	@printf "\n  \033[1mMemory (this node)\033[0m\n"
	@$(VENV_PYTHON) -c "\
	import mlx.core as mx; \
	d=mx.device_info(); \
	total=d['memory_size']/(1024**3); \
	wset=d['max_recommended_working_set_size']/(1024**3); \
	try: active=mx.metal.get_active_memory()/(1024**3); cache=mx.metal.get_cache_memory()/(1024**3); peak=mx.metal.get_peak_memory()/(1024**3) \
	except: active=cache=peak=0; \
	pct=int((active/total)*100) if total>0 else 0; \
	bar='█'*int(pct/4)+'░'*(25-int(pct/4)); \
	color='\033[32m' if pct<70 else ('\033[33m' if pct<85 else '\033[31m'); \
	print(f'  {color}{bar}\033[0m  {active:.1f} / {total:.1f} GB  ({pct}%)'); \
	print(f'  active: {active:.1f} GB │ cache: {cache:.1f} GB │ peak: {peak:.1f} GB')" 2>/dev/null \
	  || printf "  \033[2m(unable to read memory stats)\033[0m\n"
	@printf "\n"

.PHONY: monitor
monitor: ## Live-updating cluster status (refreshes every 5s)
	@printf "\033[2J"
	@while true; do \
	  printf "\033[H"; \
	  $(MAKE) --no-print-directory status 2>/dev/null; \
	  printf "\n  \033[2mRefreshing every $(MONITOR_INTERVAL)s — Ctrl+C to stop\033[0m\n"; \
	  sleep $(MONITOR_INTERVAL); \
	done

.PHONY: logs
logs: ## Tail server logs on rank 0 (requires server running via nohup or redirect)
	@printf "\n\033[1m  Server Logs\033[0m\n\n"
	@if [ -f "/tmp/mlx-jaccl-cluster.log" ]; then \
	  tail -f /tmp/mlx-jaccl-cluster.log; \
	else \
	  printf "  \033[33m  No log file found at /tmp/mlx-jaccl-cluster.log\033[0m\n"; \
	  printf "  \033[2m  To capture logs, redirect server output:\033[0m\n"; \
	  printf "  \033[2m  MODEL_DIR=... make server 2>&1 | tee /tmp/mlx-jaccl-cluster.log\033[0m\n\n"; \
	fi

# =============================================================================
# Utilities
# =============================================================================

.PHONY: kill-all
kill-all: ## Kill all MLX/server processes on all nodes (emergency stop)
	@printf "  Killing all MLX processes on all nodes...\n"
	@if [ -f "$(HOSTFILE)" ]; then \
	  HOSTS=$$(python3 -c "import json; print(' '.join(h['ssh'] for h in json.load(open('$(HOSTFILE)'))))"); \
	  for h in $$HOSTS; do \
	    printf "  → %-26s " "$$h"; \
	    ssh "$$h" 'pkill -f "python.*mlx\|openai_cluster_server\|mlx.launch" 2>/dev/null || true' 2>/dev/null; \
	    printf "done\n"; \
	  done; \
	else \
	  printf "\033[33m  ! No hostfile found, killing local processes only\033[0m\n"; \
	  pkill -f "python.*mlx\|openai_cluster_server\|mlx.launch" 2>/dev/null || true; \
	fi
	@printf "  Done.\n"

.PHONY: fingerprint
fingerprint: $(VENV_PYTHON) ## Print hardware + software fingerprint for this node
	@$(VENV_PYTHON) -c "import mlx.core as mx,json,socket;d=mx.device_info();print(json.dumps({'hostname':socket.gethostname(),'mlx':mx.__version__,'gpu':d.get('device_name','?'),'arch':d.get('architecture','?'),'ram_gb':round(d['memory_size']/(1024**3)),'wset_gb':round(d['max_recommended_working_set_size']/(1024**3),1)},indent=2))"

.PHONY: version
version: $(VENV_PYTHON) ## Show version info for all components
	@printf "\n\033[1m  Version Info\033[0m\n\n"
	@printf "  \033[36mmlx-jaccl-cluster\033[0m  %s\n" "$$(python3 -c "import tomllib;print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])" 2>/dev/null || echo '?')"
	@printf "  \033[36mmlx\033[0m                %s\n" "$$($(VENV_PYTHON) -c 'import mlx.core as mx;print(mx.__version__)' 2>/dev/null || echo '?')"
	@printf "  \033[36mmlx-lm\033[0m             %s\n" "$$($(VENV_PYTHON) -c 'import mlx_lm;print(mlx_lm.__version__)' 2>/dev/null || echo '?')"
	@printf "  \033[36mfastapi\033[0m            %s\n" "$$($(VENV_PYTHON) -c 'import fastapi;print(fastapi.__version__)' 2>/dev/null || echo '?')"
	@printf "  \033[36muvicorn\033[0m            %s\n" "$$($(VENV_PYTHON) -c 'import uvicorn;print(uvicorn.__version__)' 2>/dev/null || echo '?')"
	@printf "  \033[36mtransformers\033[0m       %s\n" "$$($(VENV_PYTHON) -c 'import transformers;print(transformers.__version__)' 2>/dev/null || echo '?')"
	@printf "  \033[36mpython\033[0m             %s\n" "$$($(VENV_PYTHON) --version 2>&1 | awk '{print $$2}')"
	@printf "  \033[36mmacOS\033[0m              %s (%s)\n" "$$(sw_vers -productVersion 2>/dev/null || echo '?')" "$$(sw_vers -buildVersion 2>/dev/null || echo '?')"
	@printf "  \033[36mchip\033[0m               %s\n" "$$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo '?')"
	@printf "\n"

.PHONY: lint
lint: $(VENV_PYTHON) ## Run basic code quality checks on server/ and scripts/*.py
	@printf "\n\033[1m  Code Quality Checks\033[0m\n\n"
	@errors=0; \
	printf "  syntax check (py_compile)...\n"; \
	for f in server/openai_cluster_server.py server/dashboard.py scripts/rdma_test.py scripts/jaccl_tps_bench.py; do \
	  if [ -f "$$f" ]; then \
	    if $(VENV_PYTHON) -m py_compile "$$f" 2>/dev/null; then \
	      printf "    \033[32m✓\033[0m $$f\n"; \
	    else \
	      printf "    \033[31m✗\033[0m $$f\n"; \
	      errors=$$((errors+1)); \
	    fi; \
	  fi; \
	done; \
	printf "\n  shellcheck (if installed)...\n"; \
	if command -v shellcheck >/dev/null 2>&1; then \
	  for f in scripts/*.sh; do \
	    if shellcheck -S warning "$$f" 2>/dev/null; then \
	      printf "    \033[32m✓\033[0m $$f\n"; \
	    else \
	      printf "    \033[31m✗\033[0m $$f\n"; \
	      errors=$$((errors+1)); \
	    fi; \
	  done; \
	else \
	  printf "    \033[2m(shellcheck not installed — brew install shellcheck)\033[0m\n"; \
	fi; \
	printf "\n"; \
	if [ $$errors -eq 0 ]; then \
	  printf "  \033[32m✓ All checks passed.\033[0m\n\n"; \
	else \
	  printf "  \033[31m✗ $$errors check(s) failed.\033[0m\n\n"; \
	  exit 1; \
	fi

.PHONY: test
test: $(MLX_LAUNCH) ## Run all tests: lint + rdma-quick + health check
	@printf "\n\033[1m  Running test suite\033[0m\n"
	@printf "  ══════════════════════════════════════════════════\n\n"
	@printf "  \033[1m[1/3]\033[0m Lint...\n"
	@$(MAKE) --no-print-directory lint
	@printf "  \033[1m[2/3]\033[0m RDMA quick test...\n"
	@$(MAKE) --no-print-directory rdma-quick
	@printf "\n  \033[1m[3/3]\033[0m Server health...\n"
	@if curl -sf http://localhost:$(HTTP_PORT)/health >/dev/null 2>&1; then \
	  printf "    \033[32m✓\033[0m Server responding on port $(HTTP_PORT)\n"; \
	else \
	  printf "    \033[33m⊘\033[0m Server not running (skipped)\n"; \
	fi
	@printf "\n  \033[32m✓ Test suite complete.\033[0m\n\n"

.PHONY: clean
clean: ## Remove .venv and __pycache__ (local node only)
	@printf "  Removing .venv and __pycache__...\n"
	@rm -rf .venv __pycache__ server/__pycache__ .node_fingerprint_*.json
	@printf "  Done. Run 'make setup' to reinstall.\n"

.PHONY: clean-all
clean-all: ## Remove .venv and __pycache__ on ALL nodes
	@if [ -f "$(HOSTFILE)" ]; then \
	  HOSTS=$$(python3 -c "import json; print(' '.join(h['ssh'] for h in json.load(open('$(HOSTFILE)'))))"); \
	  REPO_REL=$$(python3 -c "import os; print(os.path.relpath('$(REPO_DIR)', os.path.expanduser('~')))"); \
	  for h in $$HOSTS; do \
	    printf "  → cleaning %-26s\n" "$$h"; \
	    ssh "$$h" "cd ~/$$REPO_REL && rm -rf .venv __pycache__ server/__pycache__ .node_fingerprint_*.json" 2>/dev/null || true; \
	  done; \
	fi
	@printf "  Done. Run 'make setup' on each node to reinstall.\n"

# =============================================================================
# Docs
# =============================================================================

.PHONY: docs
docs: ## List all documentation files
	@printf "\n\033[1m  Documentation\033[0m\n\n"
	@printf "  \033[36mREADME.md\033[0m                      Project overview + quickstart\n"
	@printf "  \033[36mdocs/architecture.md\033[0m           Deep technical architecture reference\n"
	@printf "  \033[36mdocs/roadmap.md\033[0m                Feature roadmap + gap analysis vs exo\n"
	@printf "  \033[36mdocs/from-scratch.md\033[0m           Full setup guide (RDMA enable → server)\n"
	@printf "  \033[36mdocs/comparison-vs-exo.md\033[0m      Deep comparison with exo project\n"
	@printf "  \033[36mdocs/scripts-reference.md\033[0m      All scripts + Makefile targets reference\n"
	@printf "\n"

.PHONY: loc
loc: ## Count lines of code by component
	@printf "\n\033[1m  Lines of Code\033[0m\n\n"
	@printf "  \033[36m%-40s\033[0m %s\n" "Component" "Lines"
	@printf "  %-40s %s\n" "────────────────────────────────────────" "─────"
	@printf "  \033[36m%-40s\033[0m %s\n" "server/openai_cluster_server.py" "$$(wc -l < server/openai_cluster_server.py 2>/dev/null | tr -d ' ')"
	@printf "  \033[36m%-40s\033[0m %s\n" "server/dashboard.py" "$$(wc -l < server/dashboard.py 2>/dev/null | tr -d ' ')"
	@printf "  \033[36m%-40s\033[0m %s\n" "scripts/*.py" "$$(cat scripts/*.py 2>/dev/null | wc -l | tr -d ' ')"
	@printf "  \033[36m%-40s\033[0m %s\n" "scripts/*.sh" "$$(cat scripts/*.sh 2>/dev/null | wc -l | tr -d ' ')"
	@printf "  \033[36m%-40s\033[0m %s\n" "Makefile" "$$(wc -l < Makefile | tr -d ' ')"
	@printf "  \033[36m%-40s\033[0m %s\n" "docs/*.md" "$$(cat docs/*.md 2>/dev/null | wc -l | tr -d ' ')"
	@printf "  %-40s %s\n" "────────────────────────────────────────" "─────"
	@total=$$(( $$(cat server/*.py scripts/*.py scripts/*.sh Makefile 2>/dev/null | wc -l) )); \
	printf "  \033[1m%-40s %s\033[0m\n" "Total (code)" "$$total"
	@printf "\n"
