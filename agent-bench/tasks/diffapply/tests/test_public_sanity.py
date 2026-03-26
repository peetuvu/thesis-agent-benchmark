"""Public sanity tests for the diffapply task."""

from diffapply import apply_diff, parse_diff


def test_single_hunk_add_remove():
    original = "apple\nbanana\ncherry\ndate\n"
    diff = (
        "@@ -2,2 +2,3 @@\n"
        " banana\n"
        "-cherry\n"
        "+CHERRY\n"
        "+cranberry\n"
    )
    result = apply_diff(original, diff)
    assert result == "apple\nbanana\nCHERRY\ncranberry\ndate\n"


def test_context_lines_match():
    original = "aaa\nbbb\nccc\n"
    diff = (
        "@@ -1,3 +1,3 @@\n"
        " aaa\n"
        "-bbb\n"
        "+BBB\n"
        " ccc\n"
    )
    result = apply_diff(original, diff)
    assert result == "aaa\nBBB\nccc\n"


def test_multi_hunk_basic():
    original = "one\ntwo\nthree\nfour\nfive\nsix\n"
    diff = (
        "@@ -1,3 +1,4 @@\n"
        " one\n"
        "-two\n"
        "+TWO\n"
        "+TWO-B\n"
        " three\n"
        "@@ -5,2 +6,2 @@\n"
        " five\n"
        "-six\n"
        "+SIX\n"
    )
    result = apply_diff(original, diff)
    assert result == "one\nTWO\nTWO-B\nthree\nfour\nfive\nSIX\n"


def test_additions_only():
    original = "line1\nline2\n"
    diff = (
        "@@ -1,2 +1,3 @@\n"
        " line1\n"
        "+inserted\n"
        " line2\n"
    )
    result = apply_diff(original, diff)
    assert result == "line1\ninserted\nline2\n"


def test_empty_diff_returns_original():
    original = "hello\nworld\n"
    diff = ""
    result = apply_diff(original, diff)
    assert result == "hello\nworld\n"
