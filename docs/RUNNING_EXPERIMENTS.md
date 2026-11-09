# Running experiments

## Batch

```bash
MATRIX="--tasks calculator diffapply mdtable urlshort --models haiku-5.5 sonnet-5.5 opus-5.5 gpt-6-luna gpt-6.1-sol gpt-6-astra --archs single_agent sequential --regimes minimal production --reps 5 --reasoning-effort low"
python scripts/run_batch.py $MATRIX --dry-run
```

## Tag rule

Run every batch from a git tag in detached HEAD, and commit analysis on `main`.

```bash
MATRIX="--tasks calculator diffapply mdtable urlshort --models haiku-5.5 sonnet-5.5 opus-5.5 gpt-6-luna gpt-6.1-sol gpt-6-astra --archs single_agent sequential --regimes minimal production --reps 5 --reasoning-effort low"
test -z "$(git status --short)" && git tag experiment-v2 && git checkout experiment-v2 && nohup python scripts/run_batch.py $MATRIX > ../batch-experiment-v2.log 2>&1 &  # untested
```

Keep the log outside the repository, because any untracked file marks later runs as dirty.

### When the batch has finished

```bash
git switch main
```

## Analysis

```bash
python scripts/rescore_runs.py
python scripts/aggregate_results.py --rescored
for t in calculator diffapply mdtable urlshort; do python scripts/analyze_stages.py "$t" --rescored --csv "analysis/stage_analysis_$t.csv"; done
python scripts/classify_failures.py --rescored --csv analysis/failure_classes_mdtable.csv
git add analysis && git commit -m "Add experiment-v2 analysis"
```

## If a batch stops

1. Stop the batch.
   ```bash
   pkill -f run_batch.py  # untested
   ```
2. Print each failed run's cause and fix it.
   ```bash
   grep -lE '^  "status": "(error|timeout)"' runs/2*.json | xargs -r grep '^  "stop_reason"'
   ```
3. Move the failed runs out of `runs/`, because analysis counts them.
   ```bash
   mkdir -p ../runs_failed && grep -lE '^  "status": "(error|timeout)"' runs/2*.json | sed 's/\.json$//' | while read -r s; do mv "$s".* ../runs_failed/; done
   ```
4. Rerun only the missing combinations from the same tag, using the flags to narrow the batch.
   ```bash
   python scripts/run_batch.py --tasks calculator --models gpt-6-astra --archs sequential --regimes minimal --reps 2 --reasoning-effort low  # untested
   ```
5. Confirm every row shows `n` 5 and `err` 0.
   ```bash
   python scripts/aggregate_results.py calculator
   ```
