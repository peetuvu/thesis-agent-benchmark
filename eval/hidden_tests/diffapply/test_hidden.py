"""Hidden evaluation tests for the diffapply task.

These tests are NOT visible to the benchmarked agent.
They cover edge cases, error handling, fuzz tolerance, and
multi-hunk offset accumulation.
"""

import pytest

from diffapply import apply_diff, parse_diff
from diffapply.applier import Hunk


# ---------------------------------------------------------------------------
# 1. Multi-hunk with offset accumulation
# ---------------------------------------------------------------------------

def test_multi_hunk_with_offset_adjustment():
    """Three hunks where the first adds lines, shifting positions for later hunks."""
    original = (
        "alpha\n"
        "bravo\n"
        "charlie\n"
        "delta\n"
        "echo\n"
        "foxtrot\n"
        "golf\n"
        "hotel\n"
        "india\n"
        "juliet\n"
    )
    # Hunk 1: replace line 2 with 3 lines (+2 net)
    # Hunk 2: replace line 5 with 1 line (0 net), but offset is now +2
    # Hunk 3: replace line 9 with 2 lines (+1 net), offset is now +2
    diff = (
        "@@ -1,3 +1,5 @@\n"
        " alpha\n"
        "-bravo\n"
        "+BRAVO-1\n"
        "+BRAVO-2\n"
        "+BRAVO-3\n"
        " charlie\n"
        "@@ -5,1 +7,1 @@\n"
        "-echo\n"
        "+ECHO\n"
        "@@ -9,2 +11,3 @@\n"
        " india\n"
        "-juliet\n"
        "+JULIET-1\n"
        "+JULIET-2\n"
    )
    result = apply_diff(original, diff)
    expected = (
        "alpha\n"
        "BRAVO-1\n"
        "BRAVO-2\n"
        "BRAVO-3\n"
        "charlie\n"
        "delta\n"
        "ECHO\n"
        "foxtrot\n"
        "golf\n"
        "hotel\n"
        "india\n"
        "JULIET-1\n"
        "JULIET-2\n"
    )
    assert result == expected


# ---------------------------------------------------------------------------
# 2. Fuzz=1 allows first/last context mismatch
# ---------------------------------------------------------------------------

def test_fuzz_1_allows_first_last_context_mismatch():
    """With fuzz=1, the first and last context lines may differ."""
    original = "FIRST_ORIGINAL\nsecond\nthird\nLAST_ORIGINAL\n"
    # Diff has slightly different first and last context lines
    diff = (
        "@@ -1,4 +1,4 @@\n"
        " FIRST_DIFF\n"
        "-second\n"
        "+SECOND\n"
        " third\n"
        " LAST_DIFF\n"
    )
    result = apply_diff(original, diff, fuzz=1)
    # Fuzz lines taken from original, not diff
    assert result == "FIRST_ORIGINAL\nSECOND\nthird\nLAST_ORIGINAL\n"


# ---------------------------------------------------------------------------
# 3. Fuzz=0 strict context mismatch fails
# ---------------------------------------------------------------------------

def test_fuzz_0_strict_context_mismatch_fails():
    """Same scenario as fuzz=1 test but with fuzz=0 must raise ValueError."""
    original = "FIRST_ORIGINAL\nsecond\nthird\nLAST_ORIGINAL\n"
    diff = (
        "@@ -1,4 +1,4 @@\n"
        " FIRST_DIFF\n"
        "-second\n"
        "+SECOND\n"
        " third\n"
        " LAST_DIFF\n"
    )
    with pytest.raises(ValueError):
        apply_diff(original, diff, fuzz=0)


# ---------------------------------------------------------------------------
# 4. Empty original with additions
# ---------------------------------------------------------------------------

def test_empty_original_with_additions():
    original = ""
    diff = (
        "@@ -0,0 +1,2 @@\n"
        "+line1\n"
        "+line2\n"
    )
    result = apply_diff(original, diff)
    assert result == "line1\nline2\n"


# ---------------------------------------------------------------------------
# 5. Deletion-only diff
# ---------------------------------------------------------------------------

def test_deletion_only_diff():
    original = "aaa\nbbb\nccc\n"
    diff = (
        "@@ -1,3 +1,2 @@\n"
        " aaa\n"
        "-bbb\n"
        " ccc\n"
    )
    result = apply_diff(original, diff)
    assert result == "aaa\nccc\n"


# ---------------------------------------------------------------------------
# 6. File header lines (--- / +++) are skipped
# ---------------------------------------------------------------------------

def test_file_header_lines_skipped():
    original = "aaa\nbbb\nccc\n"
    diff = (
        "--- a/file.py\n"
        "+++ b/file.py\n"
        "@@ -1,3 +1,3 @@\n"
        " aaa\n"
        "-bbb\n"
        "+BBB\n"
        " ccc\n"
    )
    result = apply_diff(original, diff)
    assert result == "aaa\nBBB\nccc\n"


# ---------------------------------------------------------------------------
# 7. Hunk header with trailing context text
# ---------------------------------------------------------------------------

def test_hunk_header_with_context_text():
    original = "def my_function():\n    x = 1\n    y = 2\n    return x + y\n"
    diff = (
        "@@ -1,3 +1,4 @@ def my_function():\n"
        " def my_function():\n"
        "     x = 1\n"
        "+    z = 3\n"
        "     y = 2\n"
    )
    result = apply_diff(original, diff)
    assert result == "def my_function():\n    x = 1\n    z = 3\n    y = 2\n    return x + y\n"


# ---------------------------------------------------------------------------
# 8. Windows line endings normalized
# ---------------------------------------------------------------------------

def test_windows_line_endings_normalized():
    original = "aaa\r\nbbb\r\nccc\r\n"
    diff = (
        "@@ -1,3 +1,3 @@\n"
        " aaa\n"
        "-bbb\n"
        "+BBB\n"
        " ccc\n"
    )
    result = apply_diff(original, diff)
    assert result == "aaa\nBBB\nccc\n"


# ---------------------------------------------------------------------------
# 9. "No newline at end of file" marker
# ---------------------------------------------------------------------------

def test_no_newline_at_end_of_file_marker():
    original = "aaa\nbbb\n"
    diff = (
        "@@ -1,2 +1,2 @@\n"
        " aaa\n"
        "-bbb\n"
        "+bbb\n"
        "\\ No newline at end of file\n"
    )
    result = apply_diff(original, diff)
    # The marker means the preceding "+" line has no trailing newline
    assert result == "aaa\nbbb"


# ---------------------------------------------------------------------------
# 10. Hunk targets beyond file raises ValueError
# ---------------------------------------------------------------------------

def test_hunk_targets_beyond_file_raises():
    original = "aaa\nbbb\nccc\n"
    diff = (
        "@@ -10,1 +10,1 @@\n"
        "-xxx\n"
        "+yyy\n"
    )
    with pytest.raises(ValueError):
        apply_diff(original, diff)


# ---------------------------------------------------------------------------
# 11. Overlapping hunks raise ValueError
# ---------------------------------------------------------------------------

def test_overlapping_hunks_raise():
    original = "aaa\nbbb\nccc\nddd\neee\n"
    # First hunk covers lines 1-3, second hunk starts at line 2 (overlapping)
    diff = (
        "@@ -1,3 +1,3 @@\n"
        " aaa\n"
        "-bbb\n"
        "+BBB\n"
        " ccc\n"
        "@@ -2,2 +2,2 @@\n"
        "-bbb\n"
        "+BBB2\n"
        " ccc\n"
    )
    with pytest.raises(ValueError):
        apply_diff(original, diff)


# ---------------------------------------------------------------------------
# 12. Context mismatch in middle raises even with fuzz=1
# ---------------------------------------------------------------------------

def test_context_mismatch_in_middle_raises():
    original = "aaa\nbbb\nccc\nddd\neee\n"
    # Five context lines, middle one (ccc) doesn't match
    diff = (
        "@@ -1,5 +1,5 @@\n"
        " aaa\n"
        " bbb\n"
        " WRONG_MIDDLE\n"
        " ddd\n"
        " eee\n"
    )
    with pytest.raises(ValueError):
        apply_diff(original, diff, fuzz=1)


# ---------------------------------------------------------------------------
# 13. Hunk count omitted implies 1
# ---------------------------------------------------------------------------

def test_hunk_count_omitted_implies_one():
    original = "aaa\nbbb\nccc\nddd\neee\n"
    # "@@ -3 +3,2 @@" means orig_count=1
    diff = (
        "@@ -3 +3,2 @@\n"
        "-ccc\n"
        "+CCC\n"
        "+CCC2\n"
    )
    hunks = parse_diff(diff)
    assert len(hunks) == 1
    assert hunks[0].orig_start == 3
    assert hunks[0].orig_count == 1
    assert hunks[0].mod_count == 2

    result = apply_diff(original, diff)
    assert result == "aaa\nbbb\nCCC\nCCC2\nddd\neee\n"


# ---------------------------------------------------------------------------
# 14. Large diff with three hunks and offsets
# ---------------------------------------------------------------------------

def test_large_diff_three_hunks_with_offsets():
    """A realistic 10-line file with 3 hunks making various changes."""
    original = (
        "import os\n"
        "import sys\n"
        "\n"
        "def main():\n"
        "    x = 1\n"
        "    y = 2\n"
        "    print(x)\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )
    # Hunk 1: add an import after line 2 (+1 net)
    # Hunk 2: change the body (lines 5-6), replace 2 with 3 (+1 net), offset=+1
    # Hunk 3: modify the guard (line 9), same count (0 net), offset=+2
    diff = (
        "@@ -1,3 +1,4 @@\n"
        " import os\n"
        " import sys\n"
        "+import json\n"
        " \n"
        "@@ -5,2 +6,3 @@\n"
        "-    x = 1\n"
        "-    y = 2\n"
        "+    x = 10\n"
        "+    y = 20\n"
        "+    z = 30\n"
        "@@ -9,2 +11,2 @@\n"
        "-if __name__ == '__main__':\n"
        "+if __name__ == \"__main__\":\n"
        "     main()\n"
    )
    result = apply_diff(original, diff)
    expected = (
        "import os\n"
        "import sys\n"
        "import json\n"
        "\n"
        "def main():\n"
        "    x = 10\n"
        "    y = 20\n"
        "    z = 30\n"
        "    print(x)\n"
        "\n"
        "if __name__ == \"__main__\":\n"
        "    main()\n"
    )
    assert result == expected


# ---------------------------------------------------------------------------
# 15. Diff removes all lines leaving empty string
# ---------------------------------------------------------------------------

def test_diff_removes_all_lines_leaving_empty():
    original = "aaa\nbbb\n"
    diff = (
        "@@ -1,2 +0,0 @@\n"
        "-aaa\n"
        "-bbb\n"
    )
    result = apply_diff(original, diff)
    assert result == ""
