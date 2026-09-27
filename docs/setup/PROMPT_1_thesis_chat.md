# Paste this into the EXISTING thesis Claude Code chat

I'm starting a separate 3-day weekend project called `alertcal` in a new folder and a new Claude Code
chat. It packages one finding from my thesis as a small tool: a fixed anomaly threshold set in the
source environment does not transfer to the deployment environment even when ranking does, and
recalibrating the quantile on a small benign deployment slice repairs it. The tool exposes four
functions (`source_threshold`, `realised_fpr`, `recalibrate`, `sample_size_curve`), a CLI and a
FastAPI service, and runs a demo on the public Server Machine Dataset with an Isolation Forest.
It must NOT reuse thesis data or thesis-specific code, only the ideas and conventions.

Your job: analyse this thesis codebase and write a handoff document that points the new chat in the
right direction. Write it to `handoff/alertcal_handoff.md` in this folder. Be concrete: file paths,
function names, line references, actual code snippets where a convention is subtle. Cover:

1. **Score convention.** Where in the code scores are produced for each detector, and whether higher
   means more anomalous or the opposite (e.g. flow log-likelihoods are negated somewhere?). Quote it.
2. **Threshold sweep.** The exact function(s) that turn a validation percentile into a cutoff. Which
   quantile method (numpy default? something else?). Which data the percentile is taken on (fit split,
   validation split, both?). Quote the code.
3. **Realised-FPR / equal-realised-FPR protocol.** How the thesis compares methods at equal realised
   FPR by interpolating each seed's score distribution. Quote it. State whether alertcal should copy
   this or just report the realised rate.
4. **Recalibration protocol on BETH.** The 20 percent benign slice with a floor of 200, drawn per seed
   and removed from the evaluated set. Where it lives, how the slice is drawn, how removal is done.
5. **Seeds and RNG.** How seeds are handled (global numpy seed? default_rng? torch?). Recommend one
   pattern for alertcal.
6. **The three implementation errors found by adversarial review** (mentioned in the paper draft's
   limitations). What they were, and which of them could bite alertcal. This is the most valuable
   section; be specific.
7. **What NOT to copy.** Thesis-specific preprocessing, dataset loaders, window logic, anything tied to
   BETH/NSL-KDD/NAB. List it so the new chat does not drift into rebuilding the thesis.
8. **The NAB window harness** at `code/timeseries/analysis/a9_window_confusion.py` (or wherever it
   actually is): does anything in it generalise to a sample-size curve utility?
9. **Recommendations for alertcal's API** in 5–10 bullets: edge cases the thesis hit (ties in scores,
   tiny slices, alpha below 1/n, constant scores), and what the sample-size curve should report.
10. **One paragraph for the README "Why" section**, in my casual voice, linking the tool to the thesis
    and to the paper in preparation, with no confidential or unpublished numbers beyond what the
    paper draft already states.

Do not modify any thesis code. Read only. When done, print the path of the handoff file and a
10-line summary of the biggest pitfalls.
