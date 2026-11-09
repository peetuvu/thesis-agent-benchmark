# Running experiments

Finish the setup in the [README](../README.md) first, and run everything from the repository root with the virtual environment active. Items marked **[UNVERIFIED]** come from reading the code or from convention and weren't exercised.

## Batches

`scripts/run_batch.py` builds every combination of task, model, architecture and regime, repeats each `--reps` times, and calls `bench run` once per run, one at a time. The provider is worked out per model, so one batch can mix providers.

### Dry run first

Each main-experiment batch had this form, for one task at a time:

```bash
python scripts/run_batch.py \
  --tasks calculator \
  --models haiku-5.5 sonnet-5.5 opus-5.5 gpt-6-luna gpt-6.1-sol gpt-6-astra \
  --archs single_agent sequential \
  --regimes minimal production \
  --reps 5 \
  --reasoning-effort low \
  --dry-run
```

The dry run prints `Planning 120 runs across 24 combinations` and lists each run. Remove `--dry-run` to spend money.

A dry run only expands the combinations. It doesn't check that aliases exist, that keys are set, or that a model accepts the effort, so `bench models` and `python scripts/check_env.py` are still worth running first.

| Flag | Default | Note |
|---|---|---|
| `--archs` | `single_agent` | Pass both architectures explicitly |
| `--regimes` | `minimal production` | |
| `--reps` | 3 | The main experiment used 5 |
| `--reasoning-effort` | none | Required by the current models; the main experiment used `low`. Legacy models must run without it. |
| `--max-output-tokens` | `bench` default, 32000 | Per-call cap |
| `--max-wall-clock` | `bench` default, 1200 s | Per-run limit on the architecture only |
| `--provider` | from the registry | Forces one provider for every model |

For cost planning, the 480 main-experiment runs logged a total estimated cost of $41.63. Mean cost per run ranged from $0.001 (`gpt-6-luna`, single agent) to $0.34 (`gpt-6-astra`, sequential).

## The tag rule

**Batches run from a git tag in detached HEAD. Analysis is committed on `main`.** The point is that the code behind a set of results stays fixed and named while analysis scripts keep moving on `main`.

This is a project convention. Nothing in the code enforces it. Each log only records `git_commit`, `git_branch` (`HEAD` when detached) and `git_dirty`, so compliance has to be checked afterwards.

```bash
# On main, with everything needed for the run committed
git status --short                  # must print nothing
git tag experiment-vN
git checkout experiment-vN          # detached HEAD

python scripts/run_batch.py ... --dry-run
nohup python scripts/run_batch.py ... > ../batch_experiment-vN.log 2>&1 &

# When the batch has finished
git switch main
# run the analysis, then commit analysis/*.csv on main
```

Three details matter here:

- The project is installed in editable mode, so after `git checkout` the `bench` command runs the tag's code. In a test run from `experiment-v1.2`, the log recorded commit `dc1d2289`, branch `HEAD` and `git_dirty: false`.
- Keep the batch's output file **outside the repository**. `git_dirty` counts untracked files, and only `runs/*.json` and `runs/*.md` are ignored, so a log file in the repo marks every later run dirty.
- `runs/` survives the checkouts because its logs are gitignored, so results from the tag are there when you return to `main`.

### What the existing runs did

| Tag | Commit | Task | Runs | `git_branch` in logs | Batch summary |
|---|---|---|---|---|---|
| `experiment-v1` | `ee26343` | diffapply | 72 | **`main`** | `batch_26-11-05_135455.json` (3 reps) |
| `experiment-v1` | `ee26343` | diffapply | 48 | `HEAD` | `batch_26-11-05_145321.json` (2 reps) |
| `experiment-v1.1` | `ffcb7b9` | calculator | 120 | `HEAD` | `batch_26-11-05_173211.json` |
| `experiment-v1.1` | `ffcb7b9` | mdtable | 120 | `HEAD` | `batch_26-11-05_204943.json` |
| `experiment-v1.2` | `dc1d228` | urlshort | 120 | `HEAD` | `batch_26-11-05_223312.json` |

All 480 logs have `git_dirty: false` and a commit that matches a tag. The first 72 diffapply runs were at the tagged commit but on `main`, not in detached HEAD, so they're an exception to the letter of the rule. The code was the same.

## Analysis

All four scripts are offline: no API calls. Run them on `main`, in this order. Those that re-score execute the model-written code in temp directories, through the same hidden-test script as a run.

| Script | Question it answers | Scope |
|---|---|---|
| `rescore_runs.py` | Would any run's result change under the fixed code-block parser? | All tasks, or `--task T` |
| `aggregate_results.py` | For each configuration, what are pass@1, the mean hidden-test pass rate, tokens, latency and cost? | All tasks, or one |
| `analyze_stages.py` | In the sequential pipeline, did review and fix help or hurt compared with the implement step? How does the pipeline compare with single agent? | One task |
| `classify_failures.py` | Why did each mdtable run fail: a collection error, a reading of an ambiguous spec, or another bug? | mdtable only |

```bash
python scripts/rescore_runs.py
python scripts/aggregate_results.py --rescored
python scripts/analyze_stages.py mdtable --rescored --csv analysis/stage_analysis_mdtable.csv
python scripts/classify_failures.py --rescored --csv analysis/failure_classes_mdtable.csv
```

**`rescore_runs.py`** writes `analysis/rescore_<task>.csv`. The old parser ended a code block at the first ``` anywhere, which cut off code containing a literal ```. The fix is in `experiment-v1.2`, so only the urlshort runs used it live. It changed 26 of the 120 mdtable runs (2 went from failure to success, 24 stayed failures), and no diffapply, calculator or urlshort run. Re-running it for mdtable reproduced the committed CSV exactly.

**`aggregate_results.py`** prints a table per (task, provider, model, architecture, regime, effort). `pass@1` is the share of runs with status `success` that are also policy-compliant. It drops nothing: `error` and `timeout` runs count in the denominator. **Use `--rescored` for reported numbers**, or mdtable will be understated.

**`analyze_stages.py <task>`** re-scores each sequential run's implement-step code against the hidden tests and compares it with the final result, giving pass-to-pass, fail-to-pass, pass-to-fail and fail-to-fail counts. The `drift` column should be 0; anything else means today's hidden tests or task files score the run differently than the log recorded. Whether a review "reported issues" is a keyword heuristic, and the matched phrase is in the CSV so you can audit it. "Pass" here comes from test counts only and ignores policy status.

**`classify_failures.py`** runs six probes of `format_markdown`, one per README ambiguity it covers, and sorts each code version into `pass`, `collection_error`, `interpretation`, `other_failure`, `no_results` or `no_code`. `Probe false positives` should be 0. It reads ambiguity titles from the `## 4.` section of `docs/task_summary.md`, so keep that heading's number if you edit the document.

Without `--csv`, the two per-task scripts write into the current directory. `analyze_stages.py`, `classify_failures.py` and `rescore_runs.py` refuse to write inside `eval/` or `runs/`.

`scripts/inspect_runs.py` is a small helper for reading logs quickly: `python scripts/inspect_runs.py 'runs/26-11-05*.json'` prints one status line per run and one line per LLM call.

**[UNVERIFIED]** Whether each committed CSV in `analysis/` was generated with `--rescored` isn't recorded anywhere.

### Files in `runs/` that aren't the main experiment

The analysis scripts read everything in `runs/`, so `aggregate_results.py` reports 484 runs instead of 480. The four extra runs use legacy models with effort `-`, so they appear as separate rows and don't affect the main-experiment rows. There's no flag to exclude them; to analyse only the main experiment, copy its files to another directory and pass `--runs-dir`.

| File | What it is |
|---|---|
| `26-03-26_1557_anthropic_sonnet-4-6_single_agent_snake_minimal_bcb91671.{json,md}` | March pilot. Status `success` with hidden 0/0 and a policy violation, written before those cases were guarded against. |
| `26-03-27_0019_anthropic_sonnet-4-6_single_agent_calculator_minimal_46055b78.{json,md}` | March pilot |
| `26-03-27_0022_anthropic_haiku-4-5_single_agent_calculator_minimal_6c68429a.{json,md}` | March pilot |
| `26-09-15_1637_anthropic_haiku-4-5_single_agent_diffapply_minimal_7706c3ce.{json,md}` | September pilot, scored against the 15-test version of diffapply |
| `batch_26-07-01_200912.json` | July pilot batch summary. It's the only July file in `runs/`. |

The July batch's per-run logs are in the gitignored `runs_pilot/`: 17 of its 18 runs are there, and the 18th (`sonnet-4.6`, sequential, minimal) wasn't found. Other pilot and smoke logs are in `runs_pilot/` and `runs_smoke/`.

## If a batch stops partway

### What currently happens

The harness has no resume, no stop-on-error and no credit check. No run in the 480-run experiment ended in `error` or `timeout`, so the behaviour below comes from the code and from a simulation with a fake provider, not from a real credit outage.

| Event | Run log? | What the batch does |
|---|---|---|
| Provider error during a run (exhausted credits, rejected request, or a server error after SDK retries) | Yes, status `error`, reason in `stop_reason`. The hidden tests still run, against the untouched stub. | **Continues to the next run.** If credits run out, every remaining run for that provider fails quickly and writes an `error` log. |
| Wall-clock timeout | Yes, status `timeout` | Continues |
| Missing API key, or another crash before logging | No. The run prints a traceback and exits 1. | Continues; records status `unknown`, hidden `?/?`, and the last line of stderr |
| Ctrl-C **[UNVERIFIED]** | No log for the run in progress; completed runs keep theirs | Stops at once. **No batch summary is written**, because it's only written at the very end. |
| Killed, terminal closed without `nohup`, or machine shut down **[UNVERIFIED]** | As above. A hard kill can leave a `bench_*` directory in the system temp folder. | Stops; no summary |

Three more consequences:

- `run_batch.py` always exits 0 when it reaches the end, even if every run failed.
- For a provider error, the batch's `error` field and console warning show only the line of `=` signs from the end of `bench`'s output. The real message is in the run log's `stop_reason`.
- `error` logs stay in `runs/`, and `aggregate_results.py` counts them. Left in place, they lower pass@1 for their configurations.

Anthropic's and OpenAI's pinned SDKs both retry twice by default. **[UNVERIFIED]** Whether a credit-exhaustion response is retried depends on the status code the provider returns, which wasn't tested.

### What to do

This is the recommended procedure. Steps 3 and 5 are manual, and only the top-up in step 5 has been done on real runs.

1. **Stop the batch** if it's still running and burning through runs.
2. **Find the damage.** Stay on the same tag.
   ```bash
   python scripts/inspect_runs.py 'runs/YY-MM-DD*.json' | grep -E ' (error|timeout) '
   ```
   Open the `stop_reason` of a failed run for the actual cause.
3. **Set the failed runs aside.** Move all three files of each `error` run (`.json`, `.md`, `.artifacts.json`) to a directory outside the repository. Keep them rather than deleting them, since the thesis may need to report them.
4. **Fix the cause**, for example by adding credits. If the fix needs a code change, make a new tag and re-run that whole task from the new tag, so one table doesn't mix harness versions.
5. **Run only what's missing, from the same tag.** `run_batch.py` can't resume, so narrow it with `--tasks`, `--models`, `--archs`, `--regimes` and `--reps`, using several invocations if the gaps aren't rectangular. The diffapply results were built this way: 3 repetitions and then 2 more at the same commit. Repetition numbers restart, which is harmless because each run has its own id.
6. **Check the counts.** After returning to `main`, `aggregate_results.py <task>` should show `n` = 5 and `err` = 0 in every main-experiment row for that task.
7. **Write down what happened.** A partial or interrupted batch leaves no summary, so record it in the thesis notes.
