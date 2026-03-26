# Calculator — Benchmark Task

Implement a mathematical expression calculator in:

- `calculator/tokenizer.py`
- `calculator/parser.py`
- `calculator/evaluator.py`

You must implement **exactly** these functions across three files. Do not change the file structure. Do not add dependencies. Do not modify tests.

The evaluator will run both public sanity tests and separate hidden tests.

---

## Overview

Build a calculator that tokenizes, parses, and evaluates mathematical expressions given as strings. The implementation is split across three files that work together:

1. `tokenizer.py` — converts a string into a list of tokens
2. `parser.py` — converts a token list into an abstract syntax tree (AST)
3. `evaluator.py` — evaluates an AST to produce a numeric result

---

## Supported features

### Operators and precedence (lowest to highest)

| Precedence | Operators | Associativity | Description |
|------------|-----------|---------------|-------------|
| 1 (lowest) | `+`, `-`  | Left          | Addition, subtraction |
| 2          | `*`, `/`, `%` | Left      | Multiplication, division, modulo |
| 3          | `**`      | Right         | Exponentiation |
| 4 (highest)| unary `-` | Right         | Negation |

Parentheses `(` `)` override precedence as usual.

### Numbers

- Integers: `42`, `0`, `-3`
- Floats: `3.14`, `0.5`, `.5`
- Negative numbers via unary minus: `-3`, `-(2+1)`

### Variables

- Variable names are alphabetic strings: `x`, `foo`, `myVar`
- Case-sensitive: `x` and `X` are different variables
- Variables are passed as a `dict[str, float]` to the evaluator

### Whitespace

- Arbitrary whitespace between tokens is allowed and ignored
- `2 + 3` and `2+3` are equivalent

---

## Tokenizer — `tokenizer.py`

```python
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

def tokenize(expression: str) -> list[Token]:
    """Convert an expression string into a list of tokens.

    Raises ValueError for invalid characters.
    The last token must always be Token(TokenType.EOF, "").
    """
    ...
```

Token mapping:
- `+` → `PLUS`, `-` → `MINUS`, `*` → `STAR`, `/` → `SLASH`
- `%` → `PERCENT`, `**` → `POWER` (two characters, not two STAR tokens)
- `(` → `LPAREN`, `)` → `RPAREN`
- Sequences of digits (with optional single `.`) → `NUMBER`
- Sequences of alphabetic characters → `IDENT`
- Any other non-whitespace character → raise `ValueError`

---

## Parser — `parser.py`

```python
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
    ...
```

Precedence climbing or recursive descent are both acceptable approaches. The parser must handle:
- Left-associative binary operators: `1 - 2 - 3` = `(1 - 2) - 3`
- Right-associative exponentiation: `2 ** 3 ** 2` = `2 ** (3 ** 2)` = `2 ** 9` = `512`
- Unary minus: `-3 ** 2` = `-(3 ** 2)` = `-9` (unary minus binds looser than exponentiation)
- Parenthesized expressions: `(-3) ** 2` = `9`

---

## Evaluator — `evaluator.py`

```python
def evaluate(expression: str, variables: dict[str, float] | None = None) -> float:
    """Evaluate a mathematical expression string.

    Args:
        expression: The expression to evaluate.
        variables: Optional mapping of variable names to values.

    Returns:
        The numeric result as a float.

    Raises:
        ValueError: For syntax errors, undefined variables, or division by zero.
    """
    ...
```

This is the main entry point. It should:
1. Tokenize the expression using `tokenize()`
2. Parse the tokens using `parse()`
3. Walk the AST and compute the result

Evaluation rules:
- `+`, `-`, `*` — standard arithmetic
- `/` — float division (not integer division). `7 / 2` = `3.5`
- `%` — modulo. `7 % 3` = `1.0`
- `**` — exponentiation. `2 ** 10` = `1024.0`
- Division by zero → raise `ValueError` with a message containing "division by zero"
- Modulo by zero → raise `ValueError` with a message containing "division by zero"
- Undefined variable → raise `ValueError` with a message containing the variable name
- Empty expression → raise `ValueError`

---

## Examples

```python
evaluate("2 + 3")           # 5.0
evaluate("2 + 3 * 4")       # 14.0  (precedence)
evaluate("(2 + 3) * 4")     # 20.0  (parens)
evaluate("2 ** 3 ** 2")     # 512.0 (right-associative)
evaluate("-3 ** 2")          # -9.0  (unary minus binds looser)
evaluate("(-3) ** 2")        # 9.0
evaluate("x + 1", {"x": 5}) # 6.0
evaluate("10 / 3")           # 3.333...
evaluate("10 % 3")           # 1.0
evaluate("1 / 0")            # raises ValueError
evaluate("x + 1")            # raises ValueError (x undefined)
```

---

## Output format

Return complete file contents in labeled code blocks:

```calculator/tokenizer.py
# complete file contents here
```

```calculator/parser.py
# complete file contents here
```

```calculator/evaluator.py
# complete file contents here
```
