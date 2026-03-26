class ASTNode:
    """Base class for AST nodes."""
    pass


class NumberNode(ASTNode):
    def __init__(self, value: float):
        self.value = value


class VariableNode(ASTNode):
    def __init__(self, name: str):
        self.name = name


class BinaryOpNode(ASTNode):
    def __init__(self, op: str, left: ASTNode, right: ASTNode):
        self.op = op
        self.left = left
        self.right = right


class UnaryOpNode(ASTNode):
    def __init__(self, op: str, operand: ASTNode):
        self.op = op
        self.operand = operand


def parse(tokens: list) -> ASTNode:
    """Parse a token list into an AST.

    Implements operator precedence and associativity as specified.
    Raises ValueError for syntax errors (mismatched parens, unexpected tokens).
    """
    raise NotImplementedError("Not implemented")
