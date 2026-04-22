.PHONY: up down test-inventory

up: ## Start inventory service (Docker)
	docker compose up --build -d

down: ## Stop inventory service
	docker compose down

test-inventory: ## Run inventory service pytest suite (uses .venv if present)
	cd inventory-service && if [ -x .venv/bin/pytest ]; then .venv/bin/pytest tests/ -v; else python3 -m pytest tests/ -v; fi
