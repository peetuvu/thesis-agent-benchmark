# Snake (logic only) — Benchmark Task

Implement the Snake game logic in:

- `snake/game.py`

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
- **Instant reversal is forbidden**: if the action is opposite to the current direction, ignore it and keep moving in the current direction.

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
- `"alive"`: bool (initially `True`)
- `"score"`: int (initially `0`)
- `"steps"`: int (initially `0`)
- `"pending_growth"`: int (initially `0`, internal growth counter)

You may include additional keys if you want, but the above must exist.

---

## Food

Food is a dict with exactly two keys:

```python
food = {"apple": (x, y) or None, "big": (x, y) or None}
```

### Scoring and growth

| Food type | Score | Growth (segments) |
|-----------|-------|-------------------|
| `"apple"` | +1    | +1 (immediate)    |
| `"big"`   | +2    | +2 (over 2 steps) |

- When food is eaten, add the growth amount to `pending_growth`.
- Each step, if `pending_growth > 0`: keep the tail (the snake grows by one segment) and decrement `pending_growth` by 1.
- Otherwise: remove the tail (normal movement, snake length stays the same).

**Example — big food:**
1. Snake eats big food → `score += 2`, `pending_growth += 2`
2. This step: tail is kept (grow), `pending_growth` becomes 1 → snake length +1
3. Next step: tail is kept again (grow), `pending_growth` becomes 0 → snake length +1
4. Subsequent steps: tail is removed normally

### Deterministic food respawn

When a food item is eaten, it must respawn deterministically:

1. Scan all cells in **row-major order** — left to right, top to bottom:
   `(0,0), (1,0), (2,0), …, (w-1,0), (0,1), (1,1), …, (w-1,h-1)`
2. Skip any cell that is occupied by:
   - The snake body (after growth/tail handling for this step)
   - An obstacle
   - The other food item
3. Place the food at the **first free cell**.
4. If no free cell exists, set that food to `None`.

---

## Movement and collision — `step` logic

Each call to `step(state, action)` must execute in this order:

1. **Resolve direction**: if `action` is the opposite of the current `"direction"`, ignore it (keep moving in the current direction). Otherwise, update `"direction"` to `action`.

2. **Compute new head**: apply the direction delta to the current head position.
   - `"U"` → `(x, y-1)`
   - `"D"` → `(x, y+1)`
   - `"L"` → `(x-1, y)`
   - `"R"` → `(x+1, y)`

3. **Wall collision** (if `wrap` is `False`): if the new head is outside the grid bounds (`x < 0`, `x >= width`, `y < 0`, or `y >= height`), set `alive = False` and return the state.

4. **Wrap** (if `wrap` is `True`): wrap coordinates modulo grid size — `x % width`, `y % height`.

5. **Obstacle collision**: if the new head position is in `obstacles`, set `alive = False` and return.

6. **Self-collision**: if the new head position is in the **current** snake body (checked **before** any tail removal), set `alive = False` and return.

7. **Update snake**: prepend the new head to the snake list.

8. **Check food**:
   - If the new head is on `food["apple"]`: `score += 1`, `pending_growth += 1`, set `food["apple"] = None`, then respawn apple.
   - If the new head is on `food["big"]`: `score += 2`, `pending_growth += 2`, set `food["big"] = None`, then respawn big.

9. **Handle tail**:
   - If `pending_growth > 0`: decrement `pending_growth` (keep the tail → snake grows).
   - Else: remove the last element of `snake` (normal movement).

10. **Increment `steps`** by 1.

11. **Return** the updated state dictionary.

---

## Function signatures

```python
def init_state(
    width: int,
    height: int,
    snake: list[tuple[int, int]],
    food: dict[str, tuple[int, int] | None],
    direction: str,
    wrap: bool = False,
    obstacles: list[tuple[int, int]] | None = None,
) -> dict:
    """Create and return a new game state dictionary."""
    ...

def step(state: dict, action: str) -> dict:
    """Advance the game by one step. Returns the updated state."""
    ...
```

- `init_state` must return a new dict with all state keys listed above.
- `obstacles` defaults to an empty set when `None` is passed.
- `step` returns the state dict (it may mutate and return the same dict, or return a new one).
