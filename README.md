# ALRO — Autonomous Logistics & Routing Optimizer

A hybrid AI logistics system combining Q-learning route optimization with a Claude AI reasoning layer, deployed as microservices. Built as a distributed systems course project (CMPE 273).

## What it does

ALRO proves that a **learned policy beats a static rule** in a logistics environment with real-world constraints — route congestion, inventory shortfalls, and warehouse proximity. When a service fails, the system degrades gracefully rather than crashing. Claude AI explains every non-trivial decision in plain English.

```
ERP Stub ──POST /route──► Agent Service ──GET /inventory──► Inventory Service
                               │ Q-table decision
                               │ Claude reasoning (uncertain or high-priority orders)
                               └──► Route plan + explanation
                                         │
                                    Dashboard (polls /stats every 2s)
```

## Architecture

| Service | Port | Responsibility | Status |
|---|---|---|---|
| Agent Service | 8001 | Q-learning decisions + Claude AI reasoning | ✅ Built |
| Inventory Service | 8002 | Warehouse stock levels + congestion weights | ✅ Built |
| ERP Stub | 8003 | Synthetic order generation + A/B comparison | ✅ Built |
| Dashboard | 8501 | Live metrics, reward curve, service health | ✅ Built |

All services communicate over HTTP REST. No message broker required.

## Repository layout

```
alro/
├── environment/            # Shared grid library (no HTTP — pure Python)
│   ├── grid.py             # LogisticsGrid: N×N adjacency graph, step, reset
│   ├── rewards.py          # Reward constants + calculate_reward()
│   ├── baseline.py         # Greedy baseline policy (A/B comparison)
│   └── tests/
│       └── test_grid.py    # 5 acceptance tests
│
├── agent-service/          # Component B — port 8001
│   ├── main.py             # FastAPI: /route /train /stats /reset /health
│   ├── q_agent.py          # QAgent: Q-table, epsilon-greedy, Bellman update
│   ├── claude_advisor.py   # ClaudeAdvisor: structured prompt, fallback
│   ├── cache.py            # InventoryCache: TTL-based snapshot
│   ├── tests/
│   │   └── test_agent.py   # 6 unit tests + 1 integration test
│   ├── requirements.txt
│   └── Dockerfile
│
├── inventory-service/      # Component C — port 8002
│   ├── main.py             # FastAPI: /inventory /congestion /health
│   ├── state.py            # WarehouseState: in-memory stock + congestion
│   ├── tests/
│   │   └── test_inventory.py  # 7 acceptance tests
│   ├── requirements.txt
│   └── Dockerfile
│
├── erp-stub/               # Component D — port 8003
│   ├── main.py             # FastAPI: /orders/generate /orders/batch /orders/history
│   ├── order_gen.py        # Synthetic order generation
│   ├── tests/
│   │   └── test_erp_stub.py   # 4 acceptance tests
│   ├── requirements.txt
│   └── Dockerfile
│
├── dashboard/              # Component E — port 8501
│   ├── app.py              # Streamlit: reward curve, A/B chart, health, congestion map
│   ├── requirements.txt
│   └── Dockerfile
│
├── docker-compose.yml      # Full stack: all 4 services + named volume
├── Makefile                # Demo targets + test targets
└── .env.example            # All environment variables with defaults documented
```

## Quick start

```bash
# 1. Copy and fill in your API key
cp .env.example .env
# Edit .env: ANTHROPIC_API_KEY=sk-ant-...

# 2. Start all services
make up

# 3. Train the agent
make train

# 4. Run A/B comparison — verify RL beats greedy
make batch

# 5. Open the dashboard
open http://localhost:8501
```

## Demo sequence (15 minutes)

| Segment | Duration | Command | What it shows |
|---|---|---|---|
| 1 — Stack up | 3 min | `make up` | All 4 services green on dashboard |
| 2 — A/B proof | 5 min | `make train` then `make batch` | RL avg cost < greedy avg cost |
| 3 — Fault tolerance | 4 min | `make fault-inject` → `make recover` | Stale-cache degraded mode, auto-recovery |
| 4 — Live learning | 3 min | `make reset-agent` then `make live-train` | Reward curve rises on dashboard |

## Makefile targets

```bash
# Stack
make up              # docker compose up --build -d
make down            # docker compose down  (keeps Q-table volume)
make down -v         # also removes the Q-table volume

# Demo
make train           # train 500 episodes
make live-train      # train 200 episodes (for demo segment 4)
make batch           # A/B comparison — 20 orders each policy
make fault-inject    # stop Inventory Service (demo segment 3)
make recover         # restart Inventory Service
make reset-agent     # zero Q-table and reset epsilon
make stats           # print current agent stats
make logs            # tail all service logs

# Tests
make test-env        # environment library (5 tests)
make test-inventory  # inventory service (7 tests)
make test-agent      # agent service unit tests (6 tests)
make test-erp        # ERP stub (4 tests)
make test-all        # run all of the above
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(required)* | Anthropic API key — Agent Service only |
| `GRID_SIZE` | `5` | Grid dimension N — creates N×N node graph |
| `TRAINING_EPISODES` | `500` | Default episode count for `/train` |
| `EPSILON_START` | `1.0` | Initial exploration rate |
| `EPSILON_DECAY` | `0.995` | Per-episode decay multiplier |
| `EPSILON_MIN` | `0.05` | Exploration floor |
| `ALPHA` | `0.1` | Q-learning rate |
| `GAMMA` | `0.95` | Discount factor |
| `CACHE_TTL_SECONDS` | `60` | Stale inventory cache TTL |
| `CLAUDE_CONFIDENCE_THRESHOLD` | `0.10` | Q-value spread below this triggers Claude |
| `CLAUDE_TIMEOUT_SECONDS` | `3` | Anthropic API timeout |
| `QTABLE_PATH` | `/data/qtable.npy` | Q-table persistence path (Docker volume) |

## Component details

### Grid Environment Library (`environment/`)

Pure Python — no HTTP, no Docker. Imported by Agent Service and ERP Stub.

- **LogisticsGrid**: N×N adjacency graph. Warehouses on the left column, destinations on the right, transit hubs in the interior. Each episode randomises congestion weights (0–1 per edge) and stock levels (50–500 units per warehouse).
- **Reward function**: `+100` on-time delivery, `−1` per step, `−20` missed deadline, `−30` invalid warehouse (insufficient stock).
- **State encoding**: `current_node × n_nodes + destination_node` → flat integer for Q-table lookup.
- **Greedy baseline**: always picks the neighbour with the shortest Euclidean distance to the destination — ignores congestion and inventory. Used for A/B comparison.

### Agent Service (`agent-service/`) — port 8001

Core of the system. Owns the Q-table, runs training, integrates Claude.

| Endpoint | Method | Description |
|---|---|---|
| `/route` | POST | Accept order → Q-policy → optional Claude → route plan + explanation |
| `/train` | POST | Run N episodes against local grid, return reward curve |
| `/stats` | GET | Episode count, avg reward, Claude call count, recent decisions |
| `/reset` | POST | Zero Q-table and reset epsilon |
| `/health` | GET | Liveness check + Inventory Service reachability |

**Q-learning:** Q-table shape `(625, 25)` for 5×5 grid. Epsilon-greedy starts at 1.0, decays ×0.995/episode, floor 0.05. Inference uses `epsilon=0`.

**Claude trigger:** called when Q-value spread between top-2 actions < 10%, or order `priority=high`. Falls back silently on timeout — no order is blocked.

**Fault tolerance:** Inventory Service unreachable → falls back to TTL-60s cache with `stale_inventory: true`. Cache empty → HTTP 503.

### Inventory Service (`inventory-service/`) — port 8002

Thin stateless service — the fault-injection target in the demo.

| Endpoint | Method | Description |
|---|---|---|
| `/inventory` | GET | All warehouse stock levels |
| `/inventory/{node_id}` | GET | Single warehouse: stock + type |
| `/inventory/{node_id}` | PUT | Deduct quantity after delivery |
| `/congestion` | GET | Full edge congestion weight map |
| `/congestion/reset` | POST | Randomise all weights (new delivery window) |
| `/state/reset` | POST | Reset both stock and congestion |
| `/health` | GET | Liveness check |

### ERP Stub (`erp-stub/`) — port 8003

Simulates the enterprise system that originates orders. In a real deployment this service would be replaced by a connector to SAP/Oracle — the rest of the system is unchanged.

| Endpoint | Method | Description |
|---|---|---|
| `/orders/generate` | POST | Generate N orders, submit to Agent Service, return results |
| `/orders/batch` | POST | A/B comparison: same N orders via greedy then RL, return side-by-side metrics |
| `/orders/history` | GET | All submitted orders newest first (max 200) |
| `/health` | GET | Liveness check |

`/orders/batch` is the primary demo driver — it returns `improvement_cost_pct` and `improvement_ontime_pct` showing the RL advantage over greedy.

### Dashboard (`dashboard/`) — port 8501

Streamlit app. Polls Agent and Inventory services every 2 seconds. Open at `http://localhost:8501`.

**Panels:** reward curve, A/B comparison bar chart, recent decisions table (last 10), service health indicators, congestion heatmap.

**Sidebar actions:** Train 200 Episodes, Run A/B Batch, Reset Agent, Reset Congestion, Auto-refresh toggle.

## Real-world constraints modelled

| Constraint | How it works | Agent learning signal |
|---|---|---|
| Route congestion | Each edge gets a random weight [0–1] per episode; multiplies base travel cost | Avoid historically congested edges even when geometrically shorter |
| Inventory shortfall | Warehouse with stock < order quantity incurs −30 penalty | Check viability before committing to a source warehouse |
| Warehouse proximity | Proximity = 1/euclidean_distance; component of base cost | Proximity is one factor — a farther warehouse with low congestion sometimes wins |

## What Claude adds

The Q-table makes fast numeric decisions but cannot explain them. Claude bridges that gap:

- **Natural language explanation** included in every `/route` response
- **Cross-constraint reasoning** — evaluates congestion + inventory + deadline simultaneously
- **Accountability** — every high-priority order has a logged justification

Prompt structure (fixed template for comparability):
```
System:  You are a logistics routing advisor...
User:    ORDER: id, origin, destination, quantity, priority, deadline
         TOP ROUTE OPTIONS (Q-learning ranked): node paths + Q-values + est. cost
         CURRENT CONDITIONS: congestion weights, origin stock level
```

Response format: `DECISION: approve/reroute/escalate` + `REASON: [≤60 words]`

## Build sequence

| Step | Component | Gate |
|---|---|---|
| 1 | Grid Environment | `make test-env` → 5/5 ✅ |
| 2 | Inventory Service | `make test-inventory` → 7/7 ✅ |
| 3 | Agent Service | `make test-agent` → 6/6 unit ✅ |
| 4 | ERP Stub | `make test-erp` → 4/4 ✅ |
| 5 | Full stack | `make up && make train && make batch` → `improvement_cost_pct > 0` ✅ |

## Common issues

| Symptom | Fix |
|---|---|
| Agent returns 503 on `/route` | Inventory Service down + empty cache. Check `docker compose ps`. |
| Reward curve flat after training | Epsilon not decaying. Check `PENALTY_PER_STEP` is negative in env vars. |
| Claude always returns fallback | `ANTHROPIC_API_KEY` not set or timeout too low. Increase `CLAUDE_TIMEOUT_SECONDS=10` for testing. |
| Q-table not persisting across restarts | Volume not mounted. Check `qtable-data:/data` in docker-compose and `QTABLE_PATH=/data/qtable.npy`. |
| `batch improvement_cost_pct` is negative | Agent not trained before running batch. Run `make train` first. |
| Dashboard shows all services red | Services still starting. Wait for healthchecks to pass (`docker compose ps`). |
