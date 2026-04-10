# AgentX

Arena-based tree-search ML agent for [MLE-Bench](https://github.com/openai/mle-bench) competitions. Built on the [A2A protocol](https://a2a-protocol.org/).

Instead of coordinating many narrow specialist agents, AgentX uses a single **AIDE-style tree search** engine that iteratively generates and improves complete Python solutions. A tournament **arena** runs multiple independent attempts with different strategy seeds and returns the best result.

## Architecture

3 services, inspired by the key findings from the [MLE-Bench paper](https://arxiv.org/abs/2410.07095) (ICLR 2025):

```
                         run_test.py
                              |
                              v
                   +---------------------+
                   |     Evaluator       |  port 9009
                   |  (MLE-Bench Green)  |
                   +---------------------+
                              |
                   A2A: tar.gz + instructions
                              |
                              v
                   +---------------------+
                   |       Arena         |  port 8000
                   |  (Tournament Host)  |
                   +---------------------+
                        |     |     |
                        v     v     v         ← parallel, independent
                   +-------------------------+
                   |     Solver (x3)         |  port 8001
                   |   Tree Search Engine    |
                   |                         |
                   |  quick_baseline seed    |
                   |  data_first seed        |
                   |  big_model seed         |
                   +-------------------------+
```

### Evaluator (port 9009)

MLE-Bench Green agent. Downloads competition data, sends it to the arena, and grades the returned submission against Kaggle leaderboards. Unchanged from the standard MLE-Bench setup.

### Arena (port 8000)

Tournament host implementing **structural pass@k**. Receives a competition bundle from the evaluator and fans out to the solver with 3 different strategy seeds in parallel. Collects all submissions and returns the one with the highest CV score.

No LLM in the arena — pure coordination logic.

### Solver (port 8001)

AIDE-style tree search engine. The core of AgentX.

Each solver attempt:
1. Reads the competition description and data files
2. Generates a **complete, self-contained Python script** (Node 0)
3. Executes it in an isolated subprocess
4. Parses the CV score from stdout
5. Selects the best-scoring node and asks the LLM to improve it
6. Repeats for N iterations (default: 12)
7. Returns submission.csv from the best node

```
Node 0: simplest model → CV=0.50
  ├── Node 1: fix preprocessing → CV=0.65
  │     ├── Node 3: add features → CV=0.78
  │     └── Node 4: try XGBoost → CV=0.80
  └── Node 2: different approach → CV=0.72
        └── Node 5: tune params → CV=0.83  ← best, returned
```

Key properties:
- **Every node is a complete runnable script** — any node can be the final submission
- **Always branches from the best node** — greedy selection drives improvement
- **Never stops early** — iterates until the budget is exhausted
- **Fresh subprocess per node** — no state leakage between iterations
- **Domain-agnostic** — the LLM writes arbitrary Python (sklearn, torch, transformers, etc.)

## Design Rationale

### Why Tree Search Over Multi-Agent Pipeline

The MLE-Bench paper (Chan et al., 2025) evaluated three scaffolds on 75 Kaggle competitions:

| Scaffold | Type | Medal Rate |
|----------|------|------------|
| **AIDE** | Tree search over solutions | **16.9%** |
| OpenHands | General-purpose agent + tools | 4.4% |
| MLAB | General-purpose agent + tools | 0.8% |

AIDE's dominance comes from two mechanisms:
1. **Iterative improvement** — keeps refining solutions until time runs out
2. **Branching** — explores multiple solution paths, not just one pipeline

Multi-agent pipelines (plan → code → review → repair) are fundamentally linear. If the plan is wrong, everything downstream fails. Tree search explores multiple directions and lets scores drive decisions.

### Why Structural pass@k

The paper's second finding: **independent attempts double medal rates**.

| Metric | pass@1 | pass@8 |
|--------|--------|--------|
| o1-preview (AIDE) | 16.9% | 34.1% |
| GPT-4o (AIDE) | 8.7% | 17.0% |

The arena implements this structurally — 3 independent solver runs with different strategy seeds. Their failures are uncorrelated, so the probability of at least one succeeding is much higher than any single attempt.

### Why No Domain-Specific Tools

MLE-Bench spans tabular, computer vision, NLP, signal processing, and multi-modal competitions. Fixed pipelines and structured tools only handle one domain. The LLM's pretraining data contains solutions across every ML domain — letting it write arbitrary code is the only approach that generalizes.

The solver has no `infer_tabular_task`, no `evaluate_candidates`, no domain routers. Just: generate code → run it → score it → improve it.

## Strategy Seeds

Each arena attempt uses a different strategy seed that biases the solver's initial approach:

| Seed | Approach | Best For |
|------|----------|----------|
| `quick_baseline` | Simplest possible model, minimal preprocessing | Getting a valid submission fast, simple tabular tasks |
| `data_first` | Deep EDA before modeling, data-driven feature engineering | Complex datasets, domain-specific problems |
| `big_model` | Neural networks if data suggests it, strong gradient boosting otherwise | CV, NLP, signal processing, large datasets |

The strategy seed only affects Node 0. Subsequent nodes are driven by the LLM improving the best-scoring solution, regardless of the initial strategy.

## Scores

Performance on the **spaceship-titanic** competition from MLE-Bench:

| Run | Score | Medal |
|-----|-------|-------|
| 1 | 0.83448 | Gold |
| 2 | 0.83218 | Gold |
| 3 | 0.82644 | Gold |
| 4 | 0.78736 | — |

Medal thresholds: Gold >= 0.82066, Silver >= 0.81388, Bronze >= 0.80967

Gold rate: 3/4 (75%). Mean: 0.8201. Best: 0.83448.

## Project Structure

```
AgentX/
├── src/
│   ├── evaluator/              # MLE-Bench evaluator (port 9009)
│   │   ├── server.py           # Uvicorn A2A server
│   │   ├── executor.py         # A2A request handler
│   │   ├── agent.py            # Downloads competition, grades submissions
│   │   ├── messenger.py        # A2A message helpers
│   │   └── instructions.txt    # Competition instructions template
│   ├── arena/                  # Tournament host (port 8000)
│   │   ├── server.py           # Uvicorn A2A server
│   │   ├── executor.py         # A2A request handler
│   │   └── agent.py            # Parallel dispatch, collect, pick best
│   └── solver/                 # Tree search engine (port 8001)
│       ├── server.py           # Uvicorn A2A server
│       ├── executor.py         # A2A request handler
│       ├── agent.py            # Receives tar.gz + strategy, runs tree search
│       ├── tree.py             # SolutionTree: select → expand → execute → score
│       ├── interpreter.py      # Subprocess-based Python execution (Windows-safe)
│       ├── llm.py              # OpenAI API client
│       └── strategies.py       # Strategy seed definitions
├── tests/
│   ├── conftest.py             # Pytest fixtures
│   └── test_agent.py           # A2A conformance tests
├── docs/
│   └── MLE-BENCH EVALUATING MACHINE LEARNING.pdf
├── run_test.py                 # End-to-end evaluation harness
├── start_all.sh                # Start all 3 services
├── stop_all.sh                 # Stop all services
└── pyproject.toml              # Dependencies
```

## Quick Start

### Prerequisites

- Python 3.11+ with conda
- OpenAI API key
- MLE-Bench data prepared

### Install

```bash
conda create -n test python=3.11
conda activate test
pip install -e ".[test]"
```

### Run

```bash
# Start all 3 services
bash start_all.sh test

# Run evaluation on spaceship-titanic
conda run -n test python run_test.py

# Stop all services
bash stop_all.sh
```

### Port Map

| Port | Service | Role |
|------|---------|------|
| 9009 | Evaluator | Grades submissions against Kaggle leaderboards |
| 8000 | Arena | Dispatches attempts, picks best result |
| 8001 | Solver | Tree search over complete Python solutions |

## Configuration

Environment variables for the solver:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_MODEL` | `o4-mini` | LLM model for code generation |
| `MAX_ITERATIONS` | `12` | Tree search nodes per attempt |
| `CODE_TIMEOUT` | `600` | Max seconds per script execution |
| `OPENAI_API_KEY` | — | API key (or reads from key file) |

Environment variables for the arena:

| Variable | Default | Description |
|----------|---------|-------------|
| `SOLVER_URL` | `http://127.0.0.1:8001/` | Solver endpoint |
| `STRATEGIES` | `quick_baseline,data_first,big_model` | Comma-separated strategy seeds |

## Agent Communication (A2A Protocol)

All agents communicate via the [A2A protocol](https://a2a-protocol.org/).

### Message Flow

```
Evaluator                    Arena                      Solver
    |                          |                          |
    |--- tar.gz + instructions -->                        |
    |                          |--- {strategy} + tar.gz -->|  (x3 parallel)
    |                          |                          |
    |                          |      tree search loop    |
    |                          |      (12 iterations)     |
    |                          |                          |
    |                          |<-- submission.csv --------|
    |                          |                          |
    |                          |  pick best CV score      |
    |                          |                          |
    |<-- submission.csv -------|                          |
    |                          |                          |
    |  grade against leaderboard                          |
    |  return score + medal                               |
```

### Agent Discovery

Each service publishes an `AgentCard` at `/.well-known/agent.json` describing its capabilities.

## Key Design Decisions

1. **Tree search over pipeline** — Iterative improvement with branching beats linear plan-code-review pipelines. The paper proves this with a 4x medal rate advantage.

2. **LLM writes complete scripts, not diffs** — Every tree node is independently runnable. No accumulated state, no context window overflow, no cascading failures.

3. **No domain-specific tools** — The LLM's pretraining knowledge covers all ML domains. Structured tools constrain generalizability to the domains they were built for.

4. **Structural pass@k via arena** — Independent attempts with uncorrelated failures. The probability of at least one good solution scales favorably with attempt count.

5. **Subprocess isolation** — Each script runs in a fresh Python process. No state leakage between tree nodes. Crash-safe on Windows.

## References

- Chan, J.S. et al. (2025). "MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering." ICLR 2025. [arXiv:2410.07095](https://arxiv.org/abs/2410.07095)
- Schmidhuber, F. et al. (2024). "AIDE: An Automatic Machine Learning Agent." [GitHub](https://github.com/WecoAI/aideml)
- [A2A Protocol](https://a2a-protocol.org/) — Open standard for agent-to-agent communication
