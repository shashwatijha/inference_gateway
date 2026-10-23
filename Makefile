.PHONY: check-python install install-experiments test up up-fake up-native down logs bench sim model-native kind-up images k8s-deploy keda k8s-forward k8s-down

CLUSTER := inference
NS := inference

PYTHON ?= python

check-python:     ## Supported: 3.10 to 3.14 (3.12 recommended, same as Docker and CI)
	@$(PYTHON) -c 'import sys; v = sys.version_info[:2]; ok = (3, 10) <= v <= (3, 14); \
	  print(f"Python {v[0]}.{v[1]}: " + ("ok" if ok else "unsupported. Recreate the venv with: python3.12 -m venv .venv")); \
	  sys.exit(not ok)'

install: check-python  ## Python deps for tests and tools (use a venv)
	$(PYTHON) -m pip install -r requirements-dev.txt

install-experiments: check-python  ## PyTorch + transformers for experiments/ (about 1 GB)
	$(PYTHON) -m pip install -r experiments/requirements.txt

test:             ## Unit + end-to-end tests (needs redis-server: brew install redis)
	python -m pytest -q

up:               ## Full stack with llama.cpp in Docker (CPU)
	docker compose --profile model up --build -d

up-fake:          ## Full stack with the fake model server (no download, instant)
	MODEL_URL=http://fake-model:8080 docker compose --profile fake up --build -d

model-native:     ## llama.cpp natively on the Mac GPU (Metal). Run in its own terminal.
	llama-server -hf Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M --port 8080 --parallel 4 --ctx-size 8192 --metrics

up-native:        ## Stack that talks to `make model-native` on the host
	MODEL_URL=http://host.docker.internal:8080 docker compose up --build -d

down:
	docker compose --profile model --profile fake down

logs:
	docker compose logs -f gateway worker

bench:            ## Load test; override e.g. make bench ARGS="--rate 4 --requests 200"
	python loadtest/bench.py $(ARGS)

sim:              ## Batching simulator (pure Python)
	python experiments/04_batching_sim.py $(ARGS)

kind-up:          ## Local 3-node Kubernetes cluster
	kind create cluster --name $(CLUSTER) --config deploy/k8s/kind-config.yaml

images:           ## Build images and load them into kind (no registry needed)
	docker build -t inference-gateway/gateway:dev -f gateway/Dockerfile .
	docker build -t inference-gateway/worker:dev -f worker/Dockerfile .
	kind load docker-image inference-gateway/gateway:dev inference-gateway/worker:dev --name $(CLUSTER)

k8s-deploy: images
	kubectl apply -k deploy/k8s
	kubectl -n $(NS) rollout status deploy/gateway

keda:             ## Install KEDA and the queue-based worker autoscaler
	helm repo add kedacore https://kedacore.github.io/charts && helm repo update
	helm upgrade --install keda kedacore/keda -n keda --create-namespace --wait
	kubectl apply -n $(NS) -f deploy/k8s/keda-scaledobject.yaml

k8s-forward:      ## Gateway on localhost:8000
	kubectl -n $(NS) port-forward svc/gateway 8000:8000

k8s-down:
	kind delete cluster --name $(CLUSTER)
