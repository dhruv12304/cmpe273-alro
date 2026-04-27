# ALRO — Autonomous Logistics & Routing Optimizer

A hybrid AI logistics system combining Q-learning route optimization with a Claude AI reasoning layer, deployed as microservices. Built as a distributed systems course project (CMPE 273).

## What it does

ALRO proves that a **learned policy beats a static rule** in a logistics environment with real-world constraints — route congestion, inventory shortfalls, and warehouse proximity. When a service fails, the system degrades gracefully rather than crashing. Claude AI explains every non-trivial decision in plain English.

```
ERP Stub ──POST /route──► Agent Service ──GET /inventory──► Inventory Service
                               │ Q-table decision
                               │ Claude reasoning (uncertain or high-priority orders)
                               └──► Route plan + explanation
```

## Architecture

| Service | Port | Responsibility | Status |
|---|---|---|---|
| Agent Service | 8001 | Q-learning decisions + Claude AI reasoning | ✅ Built |
| Inventory Service | 8002 | Warehouse stock levels + congestion weights | ✅ Built |
| ERP Stub | 8003 | Synthetic order generation + A/B comparison | Pending |
| Dashboard | 8501 | Live metrics, reward curve, service health | Pending |

All services communicate over HTTP REST. No message broker required.

## Repository layout

```
alro/
├── environment/            # Shared grid library (no HTTP — pure Python)
│   ├── grid.py             # LogisticsGrid: N×N adjacency graph, step, reset
│   ├── rewards.py          # Reward constants + calculate_reward()
│   ├── baseline.py         # Greedy baseline policy (A/B comparison)
│   └── tests/
│       └── test_grid.py    # 5 acceptance tests (all passing)
│
├── agent-service/          # Component B — port 8001
│   ├── main.py             # FastAPI: /route /train /stats /reset /health
│   ├── q_agent.py          # QAgent: Q-table, epsilon-greedy, Bellman update
│   ├── claude_advisor.py   # ClaudeAdvisor: structured prompt, fallback
│   ├── cache.py            # InventoryCache: TTL-based snapshot
│   ├── requirements.txt
│   └── Dockerfile
│
├── inventory-service/      # Component C — port 8002
│   ├── main.py             # FastAPI: /inventory /congestion /health
│   ├── state.py            # WarehouseState: in-memory stock + congestion
│   ├── requirements.txt
│   └── Dockerfile
│
├── docker-compose.yml      # Brings up inventory + agent services
├── Makefile                # Named demo targets
└── docs/                   # Project concept + technical spec PDFs
```

## Quick start

```bash
# Copy and fill in your API key
cp .env.example .env
# ANTHROPIC_API_KEY=sk-ant-...

# Start all running services
docker compose up --build -d

# Train the agent
curl -s -X POST http://localhost:8001/train \
  -H 'Content-Type: application/json' \
  -d '{"episodes": 500}' | python3 -m json.tool

# Route a single order
curl -s -X POST http://localhost:8001/route \
  -H 'Content-Type: application/json' \
  -d '{"order_id":"test-1","origin":0,"destination":4,"quantity":100,"priority":"high","deadline":6}' \
  | python3 -m json.tool

# Check live stats
curl -s http://localhost:8001/stats | python3 -m json.tool
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

- **LogisticsGrid**: 5×5 (default) N×N adjacency graph. Warehouses on the left column, destinations on the right, transit hubs in the interior. Each episode randomizes congestion weights (0–1 per edge) and stock levels (50–500 units per warehouse).
- **Reward function**: `+100` on-time delivery, `−1` per step, `−20` missed deadline, `−30` invalid warehouse (insufficient stock).
- **State encoding**: `current_node × n_nodes + destination_node` → flat integer for Q-table lookup.
- **Greedy baseline**: always picks the neighbor with the shortest Euclidean distance to the destination — ignores congestion and inventory. Used for A/B comparison.

```bash
# Run environment acceptance tests
python -m pytest environment/tests/ -v
```

### Agent Service (`agent-service/`) — port 8001

Core of the system. Owns the Q-table, runs training, integrates Claude.

**API endpoints**

| Endpoint | Method | Description |
|---|---|---|
| `/route` | POST | Accept order → Q-policy → optional Claude → route plan + explanation |
| `/train` | POST | Run N episodes against local grid, return reward curve |
| `/stats` | GET | Episode count, avg reward, Claude call count, recent decisions |
| `/reset` | POST | Zero Q-table and reset epsilon (live demo: show learning from scratch) |
| `/health` | GET | Liveness check + Inventory Service reachability |

**Q-learning details**

- Q-table shape: `(n_states, n_actions)` = `(625, 25)` for default 5×5 grid
- Epsilon-greedy: starts at 1.0, decays ×0.995 per episode, floor 0.05
- Bellman update on every step; Q-table saved to disk after every update
- Inference uses `epsilon=0` (pure exploitation)

**Claude AI integration**

Claude is called when *either* condition is true:
1. The Q-value spread between the top-2 actions is < 10% (agent is uncertain)
2. The order has `priority: "high"`

On timeout or API error, the service falls back to the top Q-value action and sets `claude_invoked: false` — **no order is blocked**.

**Fault tolerance**

If the Inventory Service is unreachable, the Agent Service falls back to its TTL-60s cache and sets `stale_inventory: true` on the response. If the cache is also empty, it returns HTTP 503. No data loss occurs during temporary outages.

### Inventory Service (`inventory-service/`) — port 8002

Thin stateless service — the fault-injection target for the demo.

**API endpoints**

| Endpoint | Method | Description |
|---|---|---|
| `/inventory` | GET | All warehouse stock levels |
| `/inventory/{node_id}` | GET | Single warehouse: stock + type |
| `/inventory/{node_id}` | PUT | Deduct quantity after delivery |
| `/congestion` | GET | Full edge congestion weight map |
| `/congestion/reset` | POST | Randomize all weights (new delivery window) |
| `/state/reset` | POST | Reset both stock and congestion |
| `/health` | GET | Liveness check |

```bash
# Run inventory acceptance tests
python -m pytest inventory-service/tests/ -v
```

## Real-world constraints modelled

| Constraint | How it works | What the agent learns |
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

Build and verify in this order (each gate must pass before the next):

| Step | Component | Gate |
|---|---|---|
| 1 | Grid Environment | `pytest environment/tests/ -v` → 5/5 green ✅ |
| 2 | Inventory Service | `pytest inventory-service/tests/ -v` → 5/5 green ✅ |
| 3 | Agent Service | Integration tests (pending ERP Stub) |
| 4 | ERP Stub | 4 tests green |
| 5 | Dashboard | Loads + shows service health |
| 6 | Full stack | `docker compose up` → `make batch` → improvement_cost_pct > 0 |

## Makefile targets

```bash
make up            # Start all running services (docker compose up --build -d)
make down          # Stop all services
make test-inventory # Run inventory pytest suite
```

## Common issues

| Symptom | Fix |
|---|---|
| Agent returns 503 on `/route` | Inventory Service down + empty cache. Check `docker compose ps`. |
| Reward curve flat after training | Epsilon not decaying or reward constants wrong. Check `PENALTY_PER_STEP` is negative. |
| Claude always returns fallback | `ANTHROPIC_API_KEY` not set or timeout too low. Increase `CLAUDE_TIMEOUT_SECONDS=10` for testing. |
| Q-table not persisting across restarts | Volume not mounted. Check `qtable-data:/data` in docker-compose and `QTABLE_PATH=/data/qtable.npy`. |
