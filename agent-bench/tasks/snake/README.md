# Snake2 (logic only) — Benchmark Task

Implement the Snake2 game logic in:

- `agent-bench/snake/game.py`

You must implement **exactly** these functions:

- `init_state(...)`
- `step(state, action)`

Do not change the file structure. Do not add dependencies. Do not modify tests.
The evaluator will run both public sanity tests and separate hidden tests.

---

## Coordinate system and representation

- The grid is `width × height`
- Coordinates are `(x, y)`:
  - `(0, 0)` is top-left
  - `x` increases to the right
  - `y` increases downward

### Snake representation
- `snake` is a list of `(x, y)` tuples, **head first**
- Example: `[(2,2), (1,2), (0,2)]`

### Direction / actions
- Actions are strings: `"U"`, `"D"`, `"L"`, `"R"`
- The state includes a current `"direction"`
- **Instant reversal is forbidden**: if the action is opposite to current direction, ignore it and keep moving in the current direction.

Opposites: `U↔D`, `L↔R`.

---

## State dictionary

`init_state(...)` must return a **new dictionary** that contains at least these keys:

- `"width"`: int
- `"height"`: int
- `"snake"`: list of `(x,y)` head-first
- `"food"`: dict with **exact keys** `{"apple", "big"}`
- `"direction"`: `"U"|"D"|"L"|"R"`
- `"wrap"`: bool
- `"obstacles"`: set of `(x,y)` blocked cells
- `"alive"`: bool
- `"score"`: int
- `"steps"`: int
- `"pending_growth"`: int (internal counter you may use)

You may include additional keys if you want, but the above must exist.

---

## Food

Food is a dict:

```python
food = {"apple": (x,y) or None, "big": (x,y) or None}
```
