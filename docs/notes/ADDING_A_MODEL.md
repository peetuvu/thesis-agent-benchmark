# Adding a model

The worked example is `gpt-6.1-sol`, a current model. Follow the same steps for a new one, replacing the alias, API string and prices.

Adding a model on an existing provider (Anthropic or OpenAI) needs no code outside `providers/models.py`, `providers/pricing.py` and their tests. A new provider also needs its own file in `providers/` and an entry in `_PROVIDER_REGISTRY` in `providers/__init__.py`. No Google provider exists yet.

## 1. Add the registry entry

In `MODELS` in `providers/models.py`:

```python
"gpt-6.1-sol": {
    "api_string": "gpt-6.1-sol",
    "provider": "openai",
    "tier": "mid",
    "reasoning_efforts": ("low", "medium", "high", "xhigh", "max"),
    "requires_reasoning_effort": True,
    "description": "Near-Astra performance at lower cost; no 'none' effort ($2/$10 per MTok)",
},
```

The file itself keeps each entry on one line.

| Field | Rule |
|---|---|
| key | The alias used with `--model`. |
| `api_string` | The exact model id the provider expects. Must be lowercase (a test checks this). Prices are looked up by this string. |
| `provider` | A name in `_PROVIDER_REGISTRY`: `anthropic` or `openai`. |
| `tier` | `fast`, `mid` or `flagship`. |
| `reasoning_efforts` | See step 3. |
| `requires_reasoning_effort` | `True` for every current model (a test checks this). |

**Never edit or remove an existing legacy entry.** Run logs reference them, and `TestLegacyEntriesFrozen` in `tests/test_models.py` compares them field by field. Add new entries instead.

### The current lineup is one model per tier

`test_each_provider_has_one_model_per_tier` requires exactly one fast, one mid and one flagship model per provider among non-legacy aliases. So you can't simply add a fourth model. You have two options:

- **Replace a model in the lineup.** Add the old alias to `LEGACY_ALIASES`, add it to `FROZEN` in `TestLegacyEntriesFrozen`, and update the table in `TestCurrentLineup`. For an OpenAI model this needs test changes too. `test_legacy_openai_models_declare_min_reasoning_effort` requires `min_reasoning_effort` on legacy OpenAI entries, which the GPT-6 entries don't have, and the legacy entries are frozen so you can't add it later.
- **Extend the lineup.** Loosen that test on purpose.

Either way, changing these tests is a decision about the experiment design, so make it deliberately.

Also add the new alias to `TestCurrentLineup` and `TestCheckReasoningEffort` in `tests/test_models.py`.

## 2. Add the price, with its source

In `PRICES` in `providers/pricing.py`, keyed by **API string**:

```python
_OPENAI_SRC = "https://developers.openai.com/api/docs/models/{model} (retrieved 2026-11-04)"

"gpt-6.1-sol": Price(2.00, 10.00, _OPENAI_SRC.format(model="gpt-6.1-sol"), cached_input_per_mtok=0.10),
```

The numbers are USD per million tokens: input, output, then the source. Use standard first-party list prices (no batch, priority or regional pricing). The source must be the URL plus the date you retrieved it. Anthropic models share `_ANTHROPIC_SRC`; update the date when you re-check a price.

Optional fields:

| Field | Use | Example |
|---|---|---|
| `cached_input_per_mtok` | Rate for cached input tokens | The OpenAI entries |
| `long_prompt_threshold`, `long_input_per_mtok`, `long_output_per_mtok` | Higher rates above a prompt length | `claude-haiku-5-5` above 100,000 tokens |

`test_every_current_model_is_priced` fails if a current model has no price, and `test_every_price_cites_a_source` fails if a source is missing. Add a row to `test_list_prices` too.

A model with no entry still runs, but it logs `estimated_cost_usd: 0.0` and `price_source: null`, and its reports show it as unpriced. Costs are estimates from this table, not billing data.

## 3. Declare reasoning-effort support

`reasoning_efforts` lists the efforts this model accepts, taken from the provider's page for that model. It must include `low` and be a subset of `REASONING_EFFORTS`. Tests check both.

`gpt-6.1-sol` doesn't accept `none`, so it isn't in the tuple. `gpt-6-luna` does, so its tuple starts with `"none"`. The harness checks the value before any API call, in `RunConfig` and again in each provider.

How the effort reaches the API:

| Provider | Sent as | Also |
|---|---|---|
| Anthropic | `thinking={"type": "adaptive"}` and `output_config={"effort": ...}` | No temperature |
| OpenAI | `reasoning_effort=...` | Temperature only if the effort is `none` |

If a new model needs a different request shape, change that provider's file. Otherwise, no provider change is needed.

## 4. Check offline

These make no API calls:

```bash
pytest tests/test_models.py tests/test_pricing.py tests/test_sdk_compat.py
pytest
bench models | grep gpt-6.1-sol
bench run --task snake --arch single_agent --model gpt-6.1-sol --reasoning-effort none
```

The last command must exit 2 with `does not accept reasoning effort 'none'`, and it makes no call. For a new model, use an effort it doesn't list.

Check the alias with `bench models` carefully. A misspelled alias isn't rejected: it's passed through as a raw model string, the provider falls back to `anthropic`, and you only find out when the API call fails.

## 5. Smoke test

This makes one paid API call:

```bash
bench run --task snake --arch single_agent --model gpt-6.1-sol --reasoning-effort low
```

Expect `Status: success`, `Hidden: 8/8 passed`, `Public: 5/5 passed` and `Policy: compliant`. Earlier smoke runs of this model cost about $0.01. A test failure from the model doesn't by itself mean a bad registration; the checks below do.

Open the run's `.json` and confirm:

| Field | Expected for `gpt-6.1-sol` |
|---|---|
| `model` | `gpt-6.1-sol` |
| `requested_reasoning_effort` | `low` |
| `llm_calls[0].stop_reason` | `stop` (`end_turn` on Anthropic) |
| `llm_calls[0].request_params.sent_params` | `{"max_completion_tokens": 32000, "reasoning_effort": "low"}` |
| `llm_calls[0].request_params.temperature` | `null` |
| `llm_calls[0].input_tokens`, `output_tokens` | Both above 0 |
| `llm_calls[0].estimated_cost_usd` | Above 0 |
| `actual_metrics.price_source` | The source URL from step 2 |

The values above match a real `gpt-6.1-sol` log from the main experiment.

Then move the run's three files out of `runs/` so they don't enter the analysis. `runs_smoke/` is gitignored and exists for this:

```bash
mkdir -p runs_smoke
mv runs/*gpt-6.1-sol_single_agent_snake_minimal_* runs_smoke/
```

## 6. Using it in an experiment

Commit, tag, and run from the tag as described in [RUNNING_EXPERIMENTS.md](RUNNING_EXPERIMENTS.md). Results from a new model are only strictly comparable with earlier ones if the harness code, specs and hidden tests are the same, so check `git diff <old-tag> <new-tag> -- eval agent-bench architectures runner providers`.
