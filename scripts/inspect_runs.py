import json, glob, sys
for p in sorted(glob.glob(sys.argv[1])):
    if p.endswith(".artifacts.json"):
        continue
    d = json.load(open(p))
    t, m = d["test_results"], d["actual_metrics"]
    print(f"{d['model']:<20} {d['architecture_name']:<13} {d['status']:<17}"
          f"hidden {t['hidden_passed']}/{t['hidden_total']}  "
          f"{m['wall_clock_seconds']:.0f}s  ${m.get('estimated_cost_usd') or 0:.3f}  "
          f"dirty={d.get('git_dirty')}")
    for c in d.get("llm_calls", []):
        rp = c.get("request_params", {})
        print(f"    stop={c.get('stop_reason')}  out={c.get('output_tokens')}  "
              f"effort={rp.get('reasoning_effort')}  temp={rp.get('temperature')}  "
              f"err={c.get('error')}")
