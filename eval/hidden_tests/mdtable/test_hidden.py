import textwrap

import pytest

from mdtable import format_markdown
from mdtable.formatter import parse_table, format_table


def test_table_inside_fenced_code_block_not_reformatted():
    src = textwrap.dedent("""\
        Some text

        ```
        | A | B |
        |---|---|
        | 1 | 2 |
        ```

        More text
    """).rstrip("\n")

    assert format_markdown(src) == src


def test_table_inside_indented_code_block_not_reformatted():
    src = textwrap.dedent("""\
        Some text

            | A | B |
            |---|---|
            | 1 | 2 |

        More text
    """).rstrip("\n")

    assert format_markdown(src) == src


def test_escaped_pipes_in_cells():
    src = textwrap.dedent("""\
        | Name \\| Surname | Age |
        |---|---|
        | Alice \\| Smith | 30 |
    """).rstrip("\n")

    expected = textwrap.dedent("""\
        | Name \\| Surname | Age |
        |-----------------|-----|
        | Alice \\| Smith  | 30  |
    """).rstrip("\n")

    assert format_markdown(src) == expected


def test_empty_cells_preserved():
    src = textwrap.dedent("""\
        | A | B | C |
        |---|---|---|
        | | data | |
    """).rstrip("\n")

    expected = textwrap.dedent("""\
        | A | B    | C |
        |---|------|---|
        |   | data |   |
    """).rstrip("\n")

    assert format_markdown(src) == expected


def test_single_column_table():
    src = textwrap.dedent("""\
        | Header |
        |--------|
        | data   |
    """).rstrip("\n")

    expected = textwrap.dedent("""\
        | Header |
        |--------|
        | data   |
    """).rstrip("\n")

    assert format_markdown(src) == expected


def test_table_with_no_data_rows():
    src = textwrap.dedent("""\
        | A | B |
        |---|---|
    """).rstrip("\n")

    expected = textwrap.dedent("""\
        | A | B |
        |---|---|
    """).rstrip("\n")

    assert format_markdown(src) == expected


def test_multiple_tables_in_document():
    src = textwrap.dedent("""\
        # First section

        |A|B|
        |---|---|
        |1|2|

        Some text between tables.

        |X|Y|Z|
        |---|---|---|
        |a|b|c|
    """).rstrip("\n")

    expected = textwrap.dedent("""\
        # First section

        | A | B |
        |---|---|
        | 1 | 2 |

        Some text between tables.

        | X | Y | Z |
        |---|---|---|
        | a | b | c |
    """).rstrip("\n")

    assert format_markdown(src) == expected


def test_blank_lines_between_tables_preserved():
    src = textwrap.dedent("""\
        |A|B|
        |---|---|
        |1|2|


        |X|Y|
        |---|---|
        |a|b|
    """).rstrip("\n")

    expected = textwrap.dedent("""\
        | A | B |
        |---|---|
        | 1 | 2 |


        | X | Y |
        |---|---|
        | a | b |
    """).rstrip("\n")

    assert format_markdown(src) == expected


def test_center_aligned_padding_odd_width():
    """When center-aligning and padding is odd, extra space goes to the right."""
    src = textwrap.dedent("""\
        | Header | Col |
        |:------:|-----|
        | A | data |
    """).rstrip("\n")

    # "Header" = 6 chars, "A" = 1 char. Column width = 6.
    # Center padding for "A": total = 6 - 1 = 5, left = 5 // 2 = 2, right = 3
    # Cell content: "  A   " (6 chars)
    # "Col" = 3, "data" = 4. Column width = 4.
    expected = textwrap.dedent("""\
        | Header | Col  |
        |:------:|------|
        |   A    | data |
    """).rstrip("\n")

    assert format_markdown(src) == expected


def test_mixed_alignment_table():
    src = textwrap.dedent("""\
        | Default | Left | Center | Right |
        |---------|:-----|:------:|------:|
        | aa | bb | cc | dd |
    """).rstrip("\n")

    expected = textwrap.dedent("""\
        | Default | Left | Center | Right |
        |---------|:-----|:------:|------:|
        | aa      | bb   |   cc   |    dd |
    """).rstrip("\n")

    assert format_markdown(src) == expected


def test_table_at_start_of_document():
    src = textwrap.dedent("""\
        |A|B|
        |---|---|
        |1|2|

        Some text after.
    """).rstrip("\n")

    expected = textwrap.dedent("""\
        | A | B |
        |---|---|
        | 1 | 2 |

        Some text after.
    """).rstrip("\n")

    assert format_markdown(src) == expected


def test_table_at_end_of_document():
    src = textwrap.dedent("""\
        Some text before.

        |A|B|
        |---|---|
        |1|2|
    """).rstrip("\n")

    expected = textwrap.dedent("""\
        Some text before.

        | A | B |
        |---|---|
        | 1 | 2 |
    """).rstrip("\n")

    assert format_markdown(src) == expected


def test_rows_with_fewer_columns_padded():
    src = textwrap.dedent("""\
        | A | B | C |
        |---|---|---|
        | 1 |
        | x | y | z |
    """).rstrip("\n")

    expected = textwrap.dedent("""\
        | A | B | C |
        |---|---|---|
        | 1 |   |   |
        | x | y | z |
    """).rstrip("\n")

    assert format_markdown(src) == expected
