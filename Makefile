.PHONY: up down train batch fault-inject recover reset-agent live-train stats logs \
        test-env test-inventory test-agent test-erp test-all

PYTEST := $(shell command -v .venv/bin/pytest 2>/dev/null || echo python3 -m pytest)

# ---------------------------------------------------------------------------
# Stack lifecycle
# ---------------------------------------------------------------------------

up: ## Start all services (docker compose up --build -d)
	docker compose up --build -d

down: ## Stop all services
	docker compose down

# ---------------------------------------------------------------------------
# Demo targets (Section 7.2 of Technical Spec)
# ---------------------------------------------------------------------------

train: ## Train agent for 500 episodes
	curl -s -X POST http://localhost:8001/train \
	  -H 'Content-Type: application/json' \
	  -d '{"episodes": 500}' | python3 -m json.tool

live-train: ## Train 200 episodes (live demo — watch reward curve on dashboard)
	curl -s -X POST http://localhost:8001/train \
	  -H 'Content-Type: application/json' \
	  -d '{"episodes": 200}' | python3 -m json.tool

batch: ## Run A/B comparison (20 orders each policy)
	curl -s -X POST http://localhost:8003/orders/batch \
	  -H 'Content-Type: application/json' \
	  -d '{"count": 20}' | python3 -m json.tool

fault-inject: ## Stop Inventory Service to trigger stale-cache degraded mode
	docker compose stop inventory
	@echo ""
	@echo "Inventory Service stopped. Dashboard will show degraded state."
	@echo "Run 'make recover' to restart it."

recover: ## Restart Inventory Service after fault injection
	docker compose start inventory

reset-agent: ## Zero the Q-table and reset epsilon (use before live-train demo)
	curl -s -X POST http://localhost:8001/reset | python3 -m json.tool

stats: ## Print current agent stats
	curl -s http://localhost:8001/stats | python3 -m json.tool

logs: ## Tail logs from all services
	docker compose logs -f

# ---------------------------------------------------------------------------
# Test targets
# ---------------------------------------------------------------------------

test-env: ## Run grid environment acceptance tests (5 tests)
	$(PYTEST) environment/tests/ -v

test-inventory: ## Run inventory service acceptance tests (7 tests)
	cd inventory-service && if [ -x ../.venv/bin/pytest ]; then ../.venv/bin/pytest tests/ -v; else python3 -m pytest tests/ -v; fi

test-agent: ## Run agent service unit tests (6 tests, excludes integration)
	cd agent-service && if [ -x ../.venv/bin/pytest ]; then ../.venv/bin/pytest tests/ -v -m "not integration"; else python3 -m pytest tests/ -v -m "not integration"; fi

test-erp: ## Run ERP stub acceptance tests (4 tests)
	cd erp-stub && if [ -x ../.venv/bin/pytest ]; then ../.venv/bin/pytest tests/ -v; else python3 -m pytest tests/ -v; fi

test-all: ## Run all unit tests across all components
	$(MAKE) test-env
	$(MAKE) test-inventory
	$(MAKE) test-agent
	$(MAKE) test-erp
