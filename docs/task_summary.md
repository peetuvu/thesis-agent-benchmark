# Benchmark task summary

A factual description of the five benchmark tasks, written from the repository on 2026-11-05
(commit `69a792f`, plus the README fixes listed under "Changes" at the end). No code was changed and no API calls were made. Hidden tests are described
here, not quoted.

**Method.**
- Specs: `agent-bench/tasks/<task>/README.md`.
- Editable files: `runner/config.py:TaskPaths.allowed_edit_files`, which lists every `*.py` under
  `agent-bench/tasks/<task>/<task>/` except `__init__.py`.
- Test counts: `pytest --collect-only` on `agent-bench/tasks/<task>/tests/` (public) and on
  `eval/hidden_tests/<task>/` (hidden). For the hidden run, `BENCH_SOURCE_DIR` pointed at the
  unmodified stub. No test in any of the five tasks is parametrized, so the number of collected
  tests equals the number of test functions.
- Categories: hidden tests were grouped by reading each test body. The grouping is mine, not
  something the test files define.
- **[UNCERTAIN]** marks a statement that is an inference or that the repository does not settle.

## Overview

| Task | Directory | Editable files | Public tests | Hidden tests | Entry points imported by the tests |
|---|---|---|---|---|---|
| Snake (game logic) | `snake` | 1 | 5 | 8 | `snake.init_state`, `snake.step` |
| Calculator | `calculator` | 3 | 6 | 19 | `calculator.evaluate` |
| URL shortener | `urlshort` | 2 | 6 | 17 | `urlshort.URLShortener`, `urlshort.validators.is_valid_url`, `RateLimiter` |
| Markdown table formatter | `mdtable` | 1 | 5 | 13 | `mdtable.format_markdown` (hidden tests also import `parse_table` and `format_table` but never call them) |
| Unified diff applier | `diffapply` | 1 | 5 | 16 | `diffapply.apply_diff`, `parse_diff`, `diffapply.applier.Hunk` |
| **Total** | | **8** | **27** | **73** | |

Every task is a pure Python library: no I/O, no CLI and no network. The model returns full file
contents in labeled code blocks (the "Output format" section of each README).

---

## 1. Snake: `agent-bench/tasks/snake/`

**What the README asks for**
- Implement `init_state(width, height, snake, food, direction, wrap=False, obstacles=None)` and
  `step(state, action)` in `snake/game.py`. Coordinates are `(x, y)` with `(0,0)` at the top
  left, and the snake is a head-first list of tuples.
- The state dict has required keys `width`, `height`, `snake`, `food` (exactly `{"apple","big"}`),
  `direction`, `wrap`, `obstacles` (a set), `alive`, `score`, `steps` and `pending_growth`.
- Actions are `U`/`D`/`L`/`R`. An action that reverses the current direction is ignored.
- `step` runs a fixed 11-step order: resolve direction, compute the new head, wall collision (or
  wrap modulo the grid size), obstacle collision, self-collision against the **current** body
  **before** tail removal, prepend the head, eat food, handle the tail via `pending_growth`,
  increment `steps`.
- Food: an apple gives +1 score and +1 growth; big food gives +2 score and +2 growth, applied over
  two steps. An eaten item respawns deterministically at the first free cell in row-major order,
  skipping the snake, obstacles and the other food item. If no cell is free it becomes `None`.

**Editable files (1):** `snake/game.py`

**Tests:** 5 public (`agent-bench/tasks/snake/tests/test_public_sanity.py`), 8 hidden (`eval/hidden_tests/snake/test_hidden.py`)

| Category | Count | Hidden tests |
|---|---|---|
| Movement | 2 | `test_move_right_shifts_tail`, `test_wrap_keeps_alive_and_wraps_position` |
| Collisions (death) | 3 | `test_wall_collision_kills`, `test_obstacle_collision_kills`, `test_self_collision_kills_after_tail_handling` |
| Eating, score and growth | 2 | `test_eat_grows_and_scores`, `test_big_food_adds_two_growth_over_two_steps` |
| Deterministic respawn | 1 | `test_deterministic_respawn_respects_other_food_and_obstacles` |

**Note.** In `test_self_collision_kills_after_tail_handling`, the head moves onto the cell that
is currently the **tail**. The test expects death. That matches the README (self-collision is
checked before the tail is removed). The test's name and comment ("even after tail movement")
suggest the opposite, but the assertion follows the README.

---

## 2. Calculator: `agent-bench/tasks/calculator/`

**What the README asks for**
- A three-stage pipeline: `tokenize()` → `parse()` → `evaluate(expression, variables=None)`,
  using the given `TokenType`/`Token` classes and the AST classes `NumberNode`, `VariableNode`,
  `BinaryOpNode` and `UnaryOpNode`.
- Precedence from lowest to highest: `+ -` (left), `* / %` (left), unary `-`, `**` (right).
  So `-3 ** 2 == -9`. Unary minus is allowed as the right operand of `**`, so negative exponents
  work (`2 ** -3 == 0.125`).
- Integers and floats (including `.5`), case-sensitive alphabetic variables passed as a dict,
  and arbitrary whitespace. `**` is one `POWER` token.
- `/` is float division and results are floats.
- Errors are raised as `ValueError`. Division or modulo by zero must mention "division by zero",
  and an undefined variable must mention its name. Invalid characters and an empty expression
  also raise `ValueError`.

**Editable files (3):** `calculator/evaluator.py`, `calculator/parser.py`, `calculator/tokenizer.py`

**Tests:** 6 public, 19 hidden (`eval/hidden_tests/calculator/test_hidden.py`). All hidden tests call `evaluate()` only. The tokenizer and parser are tested only indirectly.

| Category | Count | Hidden tests |
|---|---|---|
| Precedence and associativity | 6 | `test_right_associative_exponent`, `test_unary_minus_vs_exponent_precedence`, `test_chained_subtraction_left_assoc`, `test_chained_division_left_assoc`, `test_unary_minus_with_addition`, `test_power_of_negative_exponent` |
| Parentheses | 3 | `test_parenthesized_negative_exponent`, `test_nested_parentheses`, `test_complex_nested_negation` |
| Operators and number formats | 2 | `test_modulo`, `test_float_numbers` |
| Variables | 1 | `test_variable_in_complex_expr` |
| Whitespace | 1 | `test_whitespace_variations` |
| Error handling | 6 | `test_modulo_by_zero`, `test_division_by_zero_in_subexpr`, `test_undefined_variable`, `test_empty_expression`, `test_whitespace_only`, `test_invalid_character` |

**Note.** Until 2026-11-05, the README's precedence table listed unary minus as "highest",
which contradicted its own examples, and it didn't say whether `2 ** -3` was allowed. The table
now puts unary minus below `**` and states that negative exponents must work. That matches
`test_unary_minus_vs_exponent_precedence`, `test_complex_nested_negation` and
`test_power_of_negative_exponent`.

---

## 3. URL shortener: `agent-bench/tasks/urlshort/`

**What the README asks for**
- `is_valid_url(url) -> bool`, implemented by hand without `urllib.parse`:
  - the scheme must be `http://` or `https://`
  - the domain must contain a dot, and must not start or end with `.` or `-`
  - each domain segment must be 1–63 characters, alphanumeric or hyphen
  - a `:port` is allowed; it is not part of the domain, and the dot rule still applies
  - whitespace or an empty string makes the URL invalid
- `RateLimiter(max_requests, window_seconds)`: a fixed-window limiter per key, with
  `is_allowed()` and `record()`.
- `URLShortener` with `shorten(url, client_ip, custom_code, ttl_seconds, current_time)`,
  `resolve(code, current_time)`, `get_stats(code)` and `_generate_code()`. Storage is in memory
  only.
- Codes are deterministic: a counter is converted to base-36 and zero-padded to `code_length`
  (the first code is `000000`), skipping codes that are already taken. Custom codes must be
  alphanumeric and are case-sensitive.
- Expiry happens when `current_time >= expires_at`. Expired links stay stored. `get_stats` still
  works on them and reports `is_expired`. `shorten` is rate-limited per `client_ip`, and the
  limit is checked **before** URL validation. `resolve` is not rate-limited.

**Editable files (2):** `urlshort/service.py`, `urlshort/validators.py`

**Tests:** 6 public, 17 hidden (`eval/hidden_tests/urlshort/test_hidden.py`)

| Category | Count | Hidden tests |
|---|---|---|
| URL validation | 7 | `test_url_missing_dot_in_domain`, `test_url_domain_starts_with_hyphen`, `test_url_domain_ends_with_hyphen`, `test_url_with_port`, `test_url_whitespace`, `test_url_scheme_only`, `test_url_long_domain_segment` |
| Rate limiter (tested directly, not through `shorten`) | 2 | `test_rate_limit_allows_up_to_max`, `test_rate_limit_window_reset` |
| Code generation and custom codes | 5 | `test_deterministic_code_generation`, `test_custom_code_collision`, `test_custom_code_case_sensitive`, `test_custom_code_non_alphanumeric_rejected`, `test_multiple_shortens_same_url` |
| Expiry, TTL and stats | 3 | `test_stats_after_expiration`, `test_shorten_with_ttl_override`, `test_resolve_increments_clicks` |

**Library, CLI or HTTP service?** A **library**: one in-memory Python class (`URLShortener`,
exported from `urlshort/__init__.py`), called directly. There is no CLI entry point, no HTTP
server and no persistence. `client_ip` is only a string key for the rate limiter.

**How errors are reported:** as Python exceptions, or as return values:

| Condition | Reported as (README) | Tested? |
|---|---|---|
| Expired link on `resolve` | `ValueError`, message must contain `"expired"` | Public only (`test_expired_link_raises`, which matches on `"expired"`). No hidden test resolves an expired link. |
| Expired link on `get_stats` | Not an error: returns `is_expired: True` | Hidden (`test_stats_after_expiration`) |
| Unknown code (`resolve`, `get_stats`) | `KeyError` | Not tested |
| Invalid URL | `ValueError` from `shorten`; `is_valid_url` returns `False` | Public (`ValueError`), hidden (`False` results) |
| Non-alphanumeric custom code | `ValueError` | Hidden |
| Duplicate custom code | `ValueError` | Not tested |
| Rate limit exceeded in `shorten` | `ValueError`, message containing `"rate limit"` | Not tested through `shorten`. Only `RateLimiter.is_allowed` returning `False` is tested. |
| Code space exhausted | `RuntimeError` from `_generate_code` | Not tested |

**Notes.**
- Until 2026-11-05, the README's port example was `http://localhost:8080/path`, which the dot
  rule forbids. The example is now `http://example.com:8080/path`. The README also says
  explicitly that the port is not part of the domain and that `localhost:8080` is invalid. That
  matches `test_url_with_port` and `test_url_missing_dot_in_domain`.
- `get_stats` takes no `current_time`, so `is_expired` uses `time.time()`.
  `test_stats_after_expiration` depends on the real clock being far past its fake timestamps,
  which is always true in practice.
- **[UNCERTAIN]** The `RateLimiter.is_allowed` docstring says both that denied calls don't
  count and that "only calls that return True count". Yet a separate `record()` exists. The
  hidden tests call `is_allowed` and then `record` for each allowed request, so they pass under
  the reading "only `record` counts" and also under "allowed `is_allowed` calls count" combined
  with a no-op `record`. Fixed and sliding windows also both pass the window-reset test.

---

## 4. Markdown table formatter: `agent-bench/tasks/mdtable/`

**What the README asks for**
- `parse_table(lines)` returns `{headers, alignments, rows}` or `None`. The second line must be a
  separator (`:?-+:?` per cell). Rows are padded or truncated to the header's column count.
- `format_table(table)` returns aligned lines: leading and trailing `|`, one space of padding,
  each column as wide as its widest cell, and alignment markers preserved.
- `format_markdown(text)` reformats every table in a document and leaves all other text
  untouched. Tables inside fenced code blocks (```` ``` ```` / `~~~`) or indented code blocks
  (4+ spaces or a tab) are left alone.
- Accepted input: standard pipe tables, tables without leading/trailing pipes, and misaligned
  tables. Edge cases named in the README: empty cells, escaped pipes `\|`, header-only tables,
  single-column tables, and multiple tables in one document.
- Cells are aligned by column alignment: right-aligned cells are padded on the left, and
  center-aligned cells are padded on both sides, with the extra space on the right when the
  padding is odd.

**Editable files (1):** `mdtable/formatter.py`

**Tests:** 5 public, 13 hidden (`eval/hidden_tests/mdtable/test_hidden.py`). Every hidden test
calls only `format_markdown` and compares the whole output string. No test input ends with a
newline.

| Category | Count | Hidden tests |
|---|---|---|
| Code blocks left untouched | 2 | `test_table_inside_fenced_code_block_not_reformatted`, `test_table_inside_indented_code_block_not_reformatted` |
| Cell content edge cases | 3 | `test_escaped_pipes_in_cells`, `test_empty_cells_preserved`, `test_rows_with_fewer_columns_padded` |
| Table shape | 2 | `test_single_column_table`, `test_table_with_no_data_rows` |
| Alignment | 2 | `test_center_aligned_padding_odd_width`, `test_mixed_alignment_table` |
| Document structure and blank lines | 4 | `test_multiple_tables_in_document`, `test_blank_lines_between_tables_preserved`, `test_table_at_start_of_document`, `test_table_at_end_of_document` |

**Ambiguities in the spec, and how the hidden tests resolve them**

| # | Ambiguity in the README | What the hidden tests assume |
|---|---|---|
| 1 | **Escaped pipes.** `\|` "is treated as a literal pipe character". This could mean unescape it in the output, or keep the escape. | The escape is kept verbatim in the output (`\|` stays `\|`), and the backslash counts toward the column width (`test_escaped_pipes_in_cells`). Writing a bare `|` would break the table and fail the test. What `parse_table` stores internally is not checked. |
| 2 | **Separator width.** "Dashes filling the full column width" combined with "one space of padding on each side" leaves open whether the separator spans *width* or *width + 2* characters, and whether it contains spaces. | Width + 2 characters, no spaces (`\|---\|` for a column of width 1). Alignment colons take the place of dashes inside that span: `:------:` for a centered width-6 column, `------:` for a right-aligned width-5 column (`test_center_aligned_padding_odd_width`, `test_mixed_alignment_table`, `test_single_column_table`). The README example is consistent with this. |
| 3 | **Minimum column width.** The README sets no minimum, while GFM-style formatters often enforce 3 dashes. | No minimum beyond the content: a 1-character column produces `\|---\|` (3 = 1 + 2) and data cells like `\| 1 \|`. **[UNCERTAIN]** A centered width-1 column (`:-:`) is never tested. |
| 4 | **Padding for `default` alignment.** The README describes padding for right and center alignment only. | `default` and `left` columns are left-justified, padded on the right (`test_mixed_alignment_table`, plus the empty cells in `test_empty_cells_preserved`, which become all spaces). |
| 5 | **Alignment of header cells.** Should headers follow the column alignment too? | Not determined. Every header in a non-default column is already the widest cell in its column, so no header needs padding. **[UNCERTAIN]** Either behaviour passes. |
| 6 | **Blank lines.** The README says to "ensure a single blank line before and after each table", but also that "consecutive blank lines … are preserved as-is". | Existing blank lines are **not** collapsed: two blank lines between two tables stay two (`test_blank_lines_between_tables_preserved`). No blank line is added at the start or end of the document (`test_table_at_start_of_document`, `test_table_at_end_of_document`). **[UNCERTAIN]** No hidden test has a table directly next to text without a blank line, so whether one gets *inserted* is never checked. |
| 7 | **Where a table ends.** The README doesn't define what ends a table block (a blank line, a line without a pipe, or a line with a different column count). | Not determined. Every table in the tests is followed by a blank line or the end of the document. |
| 8 | **Short and long rows.** Short rows are padded and long rows are truncated (both stated). | Padding is tested (`test_rows_with_fewer_columns_padded`, where a 1-cell row becomes 3 cells). Truncation of long rows is not tested. |
| 9 | **Indented code blocks.** The README says "4+ spaces or 1+ tab at line start". In CommonMark, an indented block also needs a preceding blank line and cannot interrupt a paragraph. | The only test has a blank line before the indented table and expects the document unchanged. Tab indentation and the blank-line question are not tested. |
| 10 | **Fences.** Both ```` ``` ```` and `~~~` are accepted, closing with a "matching fence". | Only ```` ``` ```` is tested. `~~~`, fences longer than 3 characters, and unclosed fences are not. |
| 11 | **Trailing newline.** Is a final newline preserved? | Not tested: every input and expected string has its trailing newline stripped. |

---

## 5. Unified diff applier: `agent-bench/tasks/diffapply/`

**What the README asks for**
- `parse_diff(diff_text) -> list[Hunk]` parses `@@ -s,c +s,c @@` headers, where a count may be
  omitted to mean 1 and text after the second `@@` is ignored. It skips `---`/`+++` file headers
  and raises `ValueError` for a malformed header, a bad line prefix, or a count mismatch.
- `apply_diff(original, diff_text, fuzz=0) -> str` applies hunks in order. It tracks a cumulative
  line offset (`mod_count - orig_count`) and raises `ValueError` on a context mismatch, an
  out-of-range hunk, or overlapping hunks.
- Fuzz: `fuzz=n` lets the first n and last n context lines of each hunk differ. The original
  file's line is then kept. Middle context lines must still match exactly.
- Edge cases: empty original, empty diff (returns the original), addition-only and
  deletion-only hunks, `\ No newline at end of file` (meaning the preceding line has no trailing
  newline), and CRLF input normalized to `\n` in the output.

**Editable files (1):** `diffapply/applier.py`

**Tests:** 5 public, 16 hidden (`eval/hidden_tests/diffapply/test_hidden.py`)

| Category | Count | Hidden tests |
|---|---|---|
| Multi-hunk offset accumulation | 2 | `test_multi_hunk_with_offset_adjustment`, `test_large_diff_three_hunks_with_offsets` |
| Context matching and fuzz | 3 | `test_fuzz_1_allows_first_last_context_mismatch`, `test_fuzz_0_strict_context_mismatch_fails`, `test_context_mismatch_in_middle_raises` |
| Parsing (headers and counts) | 3 | `test_file_header_lines_skipped`, `test_hunk_header_with_context_text`, `test_hunk_count_omitted_implies_one` (the only test that also inspects `parse_diff`'s `Hunk` fields) |
| Additions and deletions at the edges | 3 | `test_empty_original_with_additions`, `test_deletion_only_diff`, `test_diff_removes_all_lines_leaving_empty` |
| Line endings | 2 | `test_windows_line_endings_normalized`, `test_no_newline_at_end_of_file_marker` |
| Invalid application | 3 | `test_hunk_targets_beyond_file_raises`, `test_overlapping_hunks_raise`, `test_overlapping_hunks_with_consistent_content_raise` |

**Overlap test strengthened on 2026-11-05.** `test_overlapping_hunks_raise` alone did not test
overlap detection. A throwaway applier that applies hunks in order, checks context and removed
lines, and has **no** overlap check passed all 15 hidden tests that existed then, including this
one. Its `ValueError` comes from the removed-line check, because the second hunk removes a line
the first hunk already changed. `test_overlapping_hunks_with_consistent_content_raise` was
therefore added alongside it. In the new test, two hunks share an original context line and all
of their content still matches, so only an overlap check can raise. The naive applier fails it;
an applier with overlap detection passes all 16. The README now defines overlap explicitly: the
hunks' original line ranges share any line, context lines included. The throwaway appliers were
deleted. Runs scored before this change used 15 diffapply hidden tests, so they are not strictly
comparable with later runs.

---

## Changes (2026-11-05)

| File | Change |
|---|---|
| `agent-bench/tasks/calculator/README.md` | Precedence table now has unary `-` (3) below `**` (4). Added a paragraph stating `-3 ** 2 = -9`, that negative exponents must work (`2 ** -3 = 0.125`, `2 ** -1 ** 2 = 0.5`, as in Python), and a parser bullet for the negative exponent. |
| `agent-bench/tasks/urlshort/README.md` | Port example changed from `http://localhost:8080/path` to `http://example.com:8080/path`. The README now states that the port is not part of the domain and that `localhost:8080` is invalid. |
| `agent-bench/tasks/diffapply/README.md` | Added a definition of overlapping hunks: the original line ranges share any line, context lines included. Includes a header-only example (`-4,3` and `-6,2` overlap on line 6). |
| `eval/hidden_tests/diffapply/test_hidden.py` | Added `test_overlapping_hunks_with_consistent_content_raise`, next to the existing overlap test (15 → 16 hidden tests). |
| `agent-bench/tasks/urlshort/README.md` | Spec-audit clarifications: each domain segment must not start or end with `-` (`http://example-.com` is invalid); every `shorten()` call creates a new code, with no deduplication of repeated URLs; `RateLimiter.is_allowed()` only checks and never counts, and only `record()` counts (removed the sentence saying calls that return True count); `get_stats()` computes `is_expired` from the real clock (`time.time()`), since it takes no `current_time`. |
| `agent-bench/tasks/calculator/README.md` | Spec-audit clarification: an empty or whitespace-only expression (`""`, `"   "`) raises `ValueError`. |
| `agent-bench/tasks/mdtable/README.md` | Spec-audit clarification: "a single blank line before and after each table" replaced by "a blank line is ensured before and after each table (unless at document start/end); existing blank lines, including several in a row, are kept as they are". The `format_markdown` docstring says the same. Escaped-pipe handling (`\|`) deliberately left unclarified. |
| `eval/hidden_tests/urlshort/test_hidden.py` | Docstring of `test_url_domain_ends_with_hyphen` now says a domain *segment* ends with a hyphen. The "Now expired" comment in `test_stats_after_expiration` now explains that `is_expired` is true because `get_stats` uses the real clock. No assertions changed (still 17 hidden tests). Throwaway implementations written to the clarified READMEs pass all 19 calculator, 17 urlshort and 13 mdtable hidden tests; they were kept outside the repo. |
