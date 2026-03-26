# Markdown Table Formatter — Benchmark Task

Implement a markdown table parser and formatter in:

- `mdtable/formatter.py`

You must implement **exactly** the functions specified below. Do not change the file structure. Do not add dependencies. Do not modify tests.

The evaluator will run both public sanity tests and separate hidden tests.

---

## Overview

Build a tool that reads markdown text, finds tables in various formats, and rewrites them into a clean, consistently formatted pipe-table style. Non-table content must be preserved exactly as-is.

---

## Input formats to handle

### 1. Standard pipe tables

The most common markdown table format:

```
| Name  | Age | City     |
|-------|-----|----------|
| Alice | 30  | Helsinki |
| Bob   | 25  | Espoo    |
```

- Header row, separator row (dashes with optional colons for alignment), data rows
- Cells are separated by `|`
- Leading and trailing `|` may or may not be present

### 2. Simple pipe tables (no leading/trailing pipes)

```
Name  | Age | City
------|-----|--------
Alice | 30  | Helsinki
Bob   | 25  | Espoo
```

### 3. Misaligned tables

Tables where columns are not padded consistently:

```
|Name|Age|City|
|---|---|---|
|Alice|30|Helsinki|
|Bob|25|Espoo|
```

---

## Output format

All tables must be reformatted to this standard:

- Leading and trailing `|` on every row
- One space of padding on each side of every cell
- Column widths set to the widest cell in that column (including header)
- Separator row uses dashes filling the full column width
- Alignment markers preserved if present (`:---`, `:---:`, `---:`)
- A single blank line before and after each table (unless at document start/end)

Example output:

```
| Name  | Age | City     |
|-------|-----|----------|
| Alice | 30  | Helsinki |
| Bob   | 25  | Espoo    |
```

---

## Functions — `formatter.py`

```python
def parse_table(lines: list[str]) -> dict | None:
    """Attempt to parse a sequence of lines as a markdown table.

    Args:
        lines: Consecutive lines that might form a table.

    Returns:
        A dict with keys:
            "headers": list[str] — header cell contents, stripped of whitespace
            "alignments": list[str] — one of "left", "center", "right", "default" per column
            "rows": list[list[str]] — data rows, each a list of cell contents (stripped)
        Or None if the lines do not form a valid table.

    A valid table has:
        - At least 2 lines (header + separator)
        - The second line must be a separator row: cells contain only dashes,
          colons, and spaces (pattern: :?-+:?)
        - All rows must have the same number of columns (based on the header)
        - Rows with fewer columns are padded with empty strings
        - Rows with more columns than the header: extra columns are discarded
    """
    ...

def format_table(table: dict) -> list[str]:
    """Format a parsed table into clean, aligned markdown lines.

    Args:
        table: A dict as returned by parse_table().

    Returns:
        A list of formatted lines (without trailing newlines).
        Each line starts and ends with |, cells are padded with one space
        on each side, and columns are aligned to the widest cell.

    Alignment in separator row:
        "left"    → :--- + padding dashes
        "right"   → --- + padding dashes + :
        "center"  → :--- + padding dashes + :
        "default" → --- + padding dashes (no colons)
    """
    ...

def format_markdown(text: str) -> str:
    """Process a full markdown document: find tables, reformat them,
    preserve everything else unchanged.

    Args:
        text: The full markdown document as a single string.

    Returns:
        The document with all tables reformatted.
        Non-table content is preserved exactly (including blank lines,
        indentation, and inline formatting).

    Rules:
        - Tables inside fenced code blocks (``` or ~~~) must NOT be reformatted.
          A fenced code block starts with a line beginning with ``` or ~~~
          and ends with a matching fence.
        - Tables inside indented code blocks (4+ spaces or 1+ tab at line start)
          must NOT be reformatted.
        - Consecutive blank lines between non-table content are preserved as-is.
        - A single blank line is ensured before and after each reformatted table,
          unless the table is at the very start or end of the document.
    """
    ...
```

---

## Edge cases

- **Empty cells**: `| | data |` → the empty cell is preserved as an empty string
- **Escaped pipes**: `\|` within a cell is treated as a literal pipe character, not a column separator
- **Trailing whitespace**: cell contents are stripped, but lines outside tables are preserved exactly
- **No data rows**: a table with only a header and separator row is valid (0 data rows)
- **Single column**: `| Header |` with `|---|` is a valid single-column table
- **Mixed content**: a document can contain multiple tables separated by non-table text

---

## Examples

Input:
```
# My Document

Some introductory text.

|Name|Age|City|
|---|---|---|
|Alice|30|Helsinki|
|Bob|25|Espoo|

More text here.

| Score | Grade |
|------:|:-----:|
| 95    | A     |
| 82    | B     |
```

Output:
```
# My Document

Some introductory text.

| Name  | Age | City     |
|-------|-----|----------|
| Alice | 30  | Helsinki |
| Bob   | 25  | Espoo    |

More text here.

| Score | Grade |
|------:|:-----:|
|    95 |   A   |
|    82 |   B   |
```

Note: the second table has right-aligned "Score" and center-aligned "Grade" columns. Cell content is aligned accordingly: right-aligned cells are padded on the left, center-aligned cells are padded equally on both sides (extra space goes to the right if odd).

---

## Output format

Return complete file contents in a labeled code block:

```mdtable/formatter.py
# complete file contents here
```
