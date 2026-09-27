# Paste this into the NEW Claude Code chat, opened in the alertcal folder

Context: I'm Tanay, finishing a bachelor's thesis on normalizing flows for anomaly detection
(defense 12 Oct) and working as a Werkstudent in IT infrastructure architecture. This is a 3-day
weekend project. I'm learning Docker and Azure by doing it; the ML part I want to write myself with
you coaching.

Setup for this chat:
1. Read `CLAUDE.md` in full, then `docs/SPEC.md`, then `docs/HANDOFF.md`. HANDOFF.md was written by
   another Claude Code session from my thesis codebase; treat it as authoritative on conventions and
   pitfalls, and quote it back to me when it's relevant to what I'm doing.
   (If HANDOFF.md is missing, tell me and stop; it should be at
   `[ABSOLUTE PATH TO THESIS FOLDER]/handoff/alertcal_handoff.md`. Copy it into docs/.)
2. Read every file in the repo once. It was scaffolded elsewhere; check it against SPEC.md and list
   anything inconsistent before we start. Do not fix core.py.
3. Verify the SMD download paths in `scripts/get_smd.sh` against the actual GitHub repo layout, fix
   the script if the paths moved, and tell me what you changed.
4. Create a venv, run `make install`, run `make test`, and show me the 6 failing tests. Don't fix them.

Then Day 1 starts. Walk me through `core.py` one function at a time, in this order:
`realised_fpr` → `source_threshold` → `recalibrate` → `sample_size_curve`. For each: ask me what the
function should do in one sentence, correct me if needed, then let me write it. Review what I write,
run the relevant test, and if it fails point at the assertion and ask me what it means before you
explain. Keep each turn short.

After the tests pass: `make data`, `make demo`, and we look at the three numbers together. If the
fixed-threshold FPR isn't clearly above 0.05 on machine-1-2, help me pick a different deployment
machine and record why in the README.

Rules for the whole project:
- Public SMD data only. Never read or copy from the thesis folder.
- Scope is frozen to SPEC.md; new ideas go into the README roadmap.
- Before every shell command that changes state (docker, az, pip), one line on what it does.
- When I get something wrong, say so plainly.

Start with step 1.
