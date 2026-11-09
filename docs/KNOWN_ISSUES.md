# Known issues

## Using the harness

- Commands marked `# untested` were not run when these docs were written.
- Batch recovery has not been exercised on a real credit outage.
- Clones contain no run logs, because `runs/*.json` and `runs/*.md` are gitignored.
- Model-written code runs unsandboxed as your user during scoring, with access to the network, `eval/` and `runs/`.
- `run_batch.py` has no resume, no stop-on-error and no credit check, so exhausted credits turn every remaining run into an `error` log.
- `run_batch.py` writes its summary only at the end, so an interrupted batch has none.
- `run_batch.py` shows a line of `=` as the error for provider failures; read `stop_reason` in the run log instead.
- `run_batch.py` exits 0 even if every run failed, and its `errors` count includes ordinary test failures.
- `run_batch.py --dry-run` does not validate aliases, keys or efforts.
- `check_env.py` does not test whether a key works, and it prints the first 12 characters of each key.
- Nothing in the code enforces the tag rule.
- Hidden-test results count only `PASSED` and `FAILED` lines, so a fixture `ERROR` lowers the total silently.
- `classify_failures.py` covers mdtable only and needs the `## 4.` heading in `docs/task_summary.md`.
- `CLAUDE.md` and `STAGES.md` are out of date on project status.

## Reading the results

- Use `--rescored` for reported numbers: the old code-block parser changed 26 of 120 mdtable results, 2 of them from failure to success.
- Diffapply never specifies the trailing newline for an empty original, and that test alone failed 16 of its 120 runs (`gpt-6-luna` 10, `sonnet-5.5` 6).
- URL shortener hidden tests skip expired `resolve`, unknown codes, duplicate custom codes, rate limiting through `shorten`, and code exhaustion, so pass@1 overstates spec compliance.
- `max_tokens` and `max_tool_calls` are recorded but not enforced.
- The sequential pipeline gets the per-call output cap on each of its four calls, so its total budget is larger than single agent's.
- Runs are not deterministic: no seed is set and no temperature is sent.
- Each configuration has only five repetitions.
- Both architectures are single-turn with no tools, so the model never sees test output.
- SDK retries (2 per call by default) are not logged, so `api_calls` is a lower bound.
- Costs are estimates from list prices retrieved on 2026-11-04.
- The "review reported issues" column in `analyze_stages.py` is a keyword heuristic.
- Snake is a smoke test and is not part of the main experiment.
