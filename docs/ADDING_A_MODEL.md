# Adding a model

## 1. Add the alias to `MODELS` in `providers/models.py`

```python
"gpt-6.1-sol": {"api_string": "gpt-6.1-sol", "provider": "openai", "tier": "mid", "reasoning_efforts": ("low", "medium", "high", "xhigh", "max"), "requires_reasoning_effort": True, "description": "Near-Astra performance at lower cost"},
```

List only the efforts the provider documents for the model, and include `low`.

Leave legacy entries unchanged, because run logs reference them.

Keep one current model per provider and tier, or update `test_each_provider_has_one_model_per_tier`.

## 2. Add the price to `PRICES` in `providers/pricing.py`

```python
"gpt-6.1-sol": Price(2.00, 10.00, _OPENAI_SRC.format(model="gpt-6.1-sol"), cached_input_per_mtok=0.10),
```

Enter list prices in USD per million tokens, keyed by API string.

Cite the source URL and retrieval date.

## 3. Run the tests

```bash
pytest tests/test_models.py tests/test_pricing.py tests/test_sdk_compat.py
bench models | grep gpt-6.1-sol
```

## 4. Run the smoke test

```bash
bench run --task snake --arch single_agent --model gpt-6.1-sol --reasoning-effort low  # untested
mkdir -p runs_smoke && mv runs/*gpt-6.1-sol_single_agent_snake_minimal_* runs_smoke/
```

Confirm `Status: success` and a nonzero `estimated_cost_usd` in the log before moving it out of the analysis.
