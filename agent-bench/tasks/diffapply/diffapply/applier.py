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
        self.orig_start = orig_start
        self.orig_count = orig_count
        self.mod_start = mod_start
        self.mod_count = mod_count
        self.lines = lines


def parse_diff(diff_text: str) -> list[Hunk]:
    """Parse a unified diff string into a list of Hunk objects.

    Raises ValueError if the diff is malformed.
    """
    raise NotImplementedError("Not implemented")


def apply_diff(original: str, diff_text: str, fuzz: int = 0) -> str:
    """Apply a unified diff to the original file contents.

    Raises ValueError if the diff cannot be applied.
    """
    raise NotImplementedError("Not implemented")
