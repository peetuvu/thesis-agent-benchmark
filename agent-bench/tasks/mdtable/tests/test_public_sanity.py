import textwrap

import pytest

from mdtable import format_markdown


def test_simple_pipe_table_reformatted():
    src = textwrap.dedent("""\
        |Name|Age|City|
        |---|---|---|
        |Alice|30|Helsinki|
        |Bob|25|Espoo|
    """).rstrip("\n")

    expected = textwrap.dedent("""\
        | Name  | Age | City     |
        |-------|-----|----------|
        | Alice | 30  | Helsinki |
        | Bob   | 25  | Espoo    |
    """).rstrip("\n")

    assert format_markdown(src) == expected


def test_misaligned_table_fixed():
    src = textwrap.dedent("""\
        |  Name  |Age|   City   |
        |---|---|---|
        | Alice|  30 |Helsinki  |
        |Bob  |25|  Espoo|
    """).rstrip("\n")

    expected = textwrap.dedent("""\
        | Name  | Age | City     |
        |-------|-----|----------|
        | Alice | 30  | Helsinki |
        | Bob   | 25  | Espoo    |
    """).rstrip("\n")

    assert format_markdown(src) == expected


def test_non_table_text_preserved():
    src = textwrap.dedent("""\
        # Heading

        This is a paragraph with no tables.

        Another paragraph here.
    """).rstrip("\n")

    assert format_markdown(src) == src


def test_table_without_leading_pipes():
    src = textwrap.dedent("""\
        Name | Age | City
        ---|---|---
        Alice | 30 | Helsinki
    """).rstrip("\n")

    expected = textwrap.dedent("""\
        | Name  | Age | City     |
        |-------|-----|----------|
        | Alice | 30  | Helsinki |
    """).rstrip("\n")

    assert format_markdown(src) == expected


def test_right_aligned_column():
    src = textwrap.dedent("""\
        | Score | Grade |
        |------:|:-----:|
        | 95    | A     |
        | 82    | B     |
    """).rstrip("\n")

    expected = textwrap.dedent("""\
        | Score | Grade |
        |------:|:-----:|
        |    95 |   A   |
        |    82 |   B   |
    """).rstrip("\n")

    assert format_markdown(src) == expected
