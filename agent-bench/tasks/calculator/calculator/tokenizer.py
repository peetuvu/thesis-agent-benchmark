class TokenType:
    NUMBER = "NUMBER"
    PLUS = "PLUS"
    MINUS = "MINUS"
    STAR = "STAR"
    SLASH = "SLASH"
    PERCENT = "PERCENT"
    POWER = "POWER"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    IDENT = "IDENT"
    EOF = "EOF"


class Token:
    def __init__(self, type: str, value: str):
        self.type = type
        self.value = value


def tokenize(expression: str) -> list:
    """Convert an expression string into a list of tokens.

    Raises ValueError for invalid characters.
    The last token must always be Token(TokenType.EOF, "").
    """
    raise NotImplementedError("Not implemented")
