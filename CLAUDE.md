# CLAUDE.md — alertcal

Weekend project (3 days). Turns a bachelor's-thesis finding into a small, usable tool.
Read this fully before doing anything. Then read docs/SPEC.md and docs/HANDOFF.md.

## What this is
`alertcal` recalibrates anomaly-alert thresholds after deployment shift. A detector can rank
anomalies perfectly on a new server and still fire 4–11x the intended alerts, because the threshold
encodes the environment it was set in. Fix: re-estimate the (1 - alpha) quantile on a small benign
sample from the deployment environment. Model-agnostic: the tool consumes **scores**, never raw data.

Three faces of the same four functions: Python library, CLI (`alertcal`), FastAPI service in Docker
on Azure Container Apps.

## Conventions (do not change)
- Higher score = more anomalous. An event alerts when `score >= threshold`.
- `alpha` = target false-positive rate, strictly in (0, 1).
- Seeds are explicit everywhere (`np.random.default_rng(seed)`), never global state.
- Public data only: Server Machine Dataset (SMD). Nothing from the thesis datasets or from work.
- Package lives in `src/alertcal/`. Experiments in `experiments/`. No notebooks in the repo.

## How Tanay wants to work with you
- **core.py is his.** Coach, don't write. Explain the maths, point at the docstring and the test,
  review his implementation, name the bug and where it is. Only write the code if he says "just write it".
- Everything else (plumbing, Docker, Azure, README) you may write or fix directly, but explain each
  command in one line before running it. He is learning Docker and Azure by doing this.
- First principles before solutions, short answers, casual voice. No essays.
- If a test fails, show the failing assertion and ask what he thinks it means before explaining.
- Never touch the thesis folder. Read docs/HANDOFF.md instead; it was written from it.

## Commands
```
make install   # pip install -e ".[api,experiments,dev]"
make test      # pytest -q   (6 tests in tests/test_core.py; all fail until core.py is implemented)
make data      # scripts/get_smd.sh  (verify the GitHub paths first; see SPEC)
make demo      # experiments/host_shift.py -> prints 3 numbers, writes experiments/host_shift.png
make run       # uvicorn alertcal.api:app --reload
make docker    # build + run locally on :8000
azure/deploy.sh   # Day 2, needs `az login`
```

## 3-day plan
- Day 1: core.py passes all tests. Data downloaded. Demo prints the three numbers and the plot.
- Day 2: Docker works locally. Deployed on Azure. `/docs` reachable at a public URL.
- Day 3: README with plot + numbers. LinkedIn post. Resource group torn down or left up deliberately.
Scope is frozen. New ideas go to the Roadmap section of README.md, not into code.
