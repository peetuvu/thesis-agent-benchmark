# Unified Diff Applier — Benchmark Task

Implement a unified diff parser and applier in:

- `diffapply/applier.py`

You must implement **exactly** the functions specified below. Do not change the file structure. Do not add dependencies. Do not modify tests.

The evaluator will run both public sanity tests and separate hidden tests.

---

## Overview

Build a tool that takes the contents of a text file and a unified diff string, then applies the diff to produce the modified file. This is the same operation performed by `patch` on Unix systems, but implemented from scratch.

---

## Unified diff format

A unified diff consists of one or more **hunks**. Each hunk describes a contiguous change in the file.

### Hunk header

```
@@ -start,count +start,count @@
```

- `-start,count` refers to the original file: starting line number and number of lines shown from the original
- `+start,count` refers to the modified file: starting line number and number of lines in the result
- Line numbers are 1-based
- If `count` is 1, it may be omitted: `@@ -5 +5,2 @@` means `-5,1 +5,2`
- The `@@` may be followed by optional context text (e.g., function name) which should be ignored

### Hunk body lines

Each line in a hunk body starts with a single character prefix:

- ` ` (space) — context line: appears in both original and modified
- `-` — removed line: appears only in the original
- `+` — added line: appears only in the modified

### Example diff

```
@@ -1,4 +1,5 @@
 line one
-line two
+line 2
+line 2b
 line three
 line four
```

This means: in the original file starting at line 1, replace "line two" with "line 2" and "line 2b". Context lines ("line one", "line three", "line four") must match the original file.

---

## Functions — `applier.py`

```python
class Hunk:
    """Represents a single diff hunk."""

    def __init__(
        self,
        orig_start: int,
        orig_count: int,
        mod_start: int,
        mod_count: int,
        lines: list[tuple[str, str]],
    ):
        """
        Args:
            orig_start: 1-based start line in original file.
            orig_count: Number of lines from original in this hunk.
            mod_start: 1-based start line in modified file.
            mod_count: Number of lines in the modified result for this hunk.
            lines: List of (prefix, content) tuples where prefix is " ", "-", or "+".
        """
        self.orig_start = orig_start
        self.orig_count = orig_count
        self.mod_start = mod_start
        self.mod_count = mod_count
        self.lines = lines


def parse_diff(diff_text: str) -> list[Hunk]:
    """Parse a unified diff string into a list of Hunk objects.

    Args:
        diff_text: The diff as a string. May include file header lines
                   (--- a/file, +++ b/file) which should be skipped.
                   Only @@ hunk headers and their body lines are parsed.

    Returns:
        A list of Hunk objects in the order they appear.

    Raises:
        ValueError: If the diff is malformed:
            - Hunk header cannot be parsed
            - Line prefix is not " ", "-", or "+"
            - Hunk body line counts don't match the header
    """
    ...


def apply_diff(original: str, diff_text: str, fuzz: int = 0) -> str:
    """Apply a unified diff to the original file contents.

    Args:
        original: The complete original file as a single string.
        diff_text: The unified diff to apply.
        fuzz: Number of context lines that are allowed to not match.
              fuzz=0 means all context lines must match exactly.
              fuzz=1 means the first and last context line of each hunk
              may differ. fuzz=2 allows the first 2 and last 2 to differ.
              Non-matching context lines are taken from the original file
              (not from the diff).

    Returns:
        The modified file as a single string.

    Raises:
        ValueError: If the diff cannot be applied:
            - Context line mismatch (outside fuzz tolerance)
            - Hunk targets a line range outside the file
            - Hunks overlap after offset adjustment
    """
    ...
```

---

## Applying multi-hunk diffs

When a diff contains multiple hunks, earlier hunks may change the total line count of the file. Subsequent hunks must account for this **line offset**:

1. Process hunks in order (top to bottom).
2. After each hunk, calculate the offset: `offset += (mod_count - orig_count)` for that hunk.
3. For subsequent hunks, the actual position in the current file is `orig_start + accumulated_offset - 1` (converting to 0-based index).

Example: if hunk 1 adds 2 lines (orig_count=3, mod_count=5), subsequent hunks' positions shift by +2.

---

## Context line matching

Context lines (prefix ` `) must match the corresponding lines in the original file exactly (byte-for-byte comparison). If they don't match:

- With `fuzz=0`: raise ValueError immediately
- With `fuzz > 0`: the first `fuzz` and last `fuzz` context lines of each hunk are allowed to differ. If a context line is within the fuzz range, use the original file's line instead of the diff's context line. Context lines in the middle of the hunk (outside the fuzz zone) must still match exactly.

---

## Edge cases

- **Empty original file**: applying a diff that only adds lines should work
- **Empty diff**: return the original unchanged
- **Diff with only additions** (no `-` lines, no context): the `orig_count` is 0 and `orig_start` indicates the insertion point
- **Diff with only deletions** (no `+` lines): valid, removes lines
- **Trailing newline handling**: if the original ends with `\n`, the result should also end with `\n` (unless the diff removes the last line). If the diff contains `\ No newline at end of file`, that line is a comment and should be ignored (do not include it in the output), but it signals that the preceding line does NOT have a trailing newline.
- **File header lines**: `--- a/file.py` and `+++ b/file.py` may appear before the first hunk. Skip them.
- **Hunk header context**: `@@ -1,3 +1,4 @@ def my_function():` — the text after the second `@@` is ignored.
- **Windows line endings**: treat `\r\n` and `\n` equivalently. Output with `\n`.

---

## Examples

### Single hunk

Original:
```
apple
banana
cherry
date
```

Diff:
```
@@ -2,2 +2,3 @@
 banana
-cherry
+CHERRY
+cranberry
```

Result:
```
apple
banana
CHERRY
cranberry
date
```

### Multiple hunks with offset

Original (6 lines):
```
one
two
three
four
five
six
```

Diff:
```
@@ -1,3 +1,4 @@
 one
-two
+TWO
+TWO-B
 three
@@ -5,2 +6,2 @@
 five
-six
+SIX
```

Result:
```
one
TWO
TWO-B
three
four
five
SIX
```

Note: the second hunk header says `@@ -5,2 +6,2 @@`. The `-5` refers to line 5 in the original. After the first hunk added a line, the accumulated offset is +1, so line 5 of the original is now at position 6 in the working file. The applier must handle this.

---

## Output format

Return complete file contents in a labeled code block:

```diffapply/applier.py
# complete file contents here
```
