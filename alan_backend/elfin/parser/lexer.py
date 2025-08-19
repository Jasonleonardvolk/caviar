"""
ELFIN DSL Lexer

This module provides lexical analysis for the ELFIN DSL, tokenizing the input string
into a sequence of tokens that can be processed by the parser.
"""

import re
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict


class TokenType(Enum):
    """Token types for the ELFIN DSL lexer."""
    # Keywords
    CONCEPT = auto()
    RELATION = auto()
    PSI_MODE = auto()
    STABILITY = auto()
    LYAPUNOV = auto()
    KOOPMAN = auto()
    AGENT = auto()
    GOAL = auto()
    ASSUME = auto()
    REQUIRE = auto()
    IS = auto()
    HAS = auto()
    WITH = auto()
    
    # Stability-related keywords
    STABLE = auto()
    UNSTABLE = auto()
    ASYMPTOTIC = auto()
    EXPONENTIAL = auto()
    MARGINAL = auto()
    CONDITIONAL = auto()
    
    # Oscillator types
    KURAMOTO = auto()
    WINFREE = auto()
    STUART_LANDAU = auto()
    HOPF = auto()
    PRC = auto()  # Phase Response Curve
    
    # Symbols
    COLON = auto()
    SEMICOLON = auto()
    COMMA = auto()
    DOT = auto()
    ARROW = auto()  # ->
    DOUBLE_ARROW = auto()  # <->
    CURLY_OPEN = auto()  # {
    CURLY_CLOSE = auto()  # }
    PAREN_OPEN = auto()  # (
    PAREN_CLOSE = auto()  # )
    BRACKET_OPEN = auto()  # [
    BRACKET_CLOSE = auto()  # ]
    EQUALS = auto()  # =
    PLUS = auto()  # +
    MINUS = auto()  # -
    STAR = auto()  # *
    SLASH = auto()  # /
    CARET = auto()  # ^
    LESS_THAN = auto()  # <
    GREATER_THAN = auto()  # >
    LESS_EQUAL = auto()  # <=
    GREATER_EQUAL = auto()  # >=
    
    # Literals
    IDENTIFIER = auto()
    STRING = auto()
    NUMBER = auto()
    FLOAT = auto()
    PSI_SYMBOL = auto()  # ψ or \psi
    PHI_SYMBOL = auto()  # φ or \phi
    
    # Special
    EOF = auto()
    ERROR = auto()


@dataclass
class Token:
    """
    Represents a token in the ELFIN DSL.
    
    Attributes:
        type: The token type
        lexeme: The original text of the token
        literal: The literal value (for numbers, strings, etc.)
        line: The line number in the source code
        column: The column number in the source code
    """
    type: TokenType
    lexeme: str
    literal: Optional[object] = None
    line: int = 0
    column: int = 0


class Lexer:
    """
    Lexical analyzer for the ELFIN DSL.
    
    This class transforms a string of ELFIN DSL source code into a sequence
    of tokens that can be parsed by the ELFIN parser.
    """
    
    # Keywords mapping
    KEYWORDS = {
        "concept": TokenType.CONCEPT,
        "relation": TokenType.RELATION,
        "psi_mode": TokenType.PSI_MODE,
        "stability": TokenType.STABILITY,
        "lyapunov": TokenType.LYAPUNOV,
        "koopman": TokenType.KOOPMAN,
        "agent": TokenType.AGENT,
        "goal": TokenType.GOAL,
        "assume": TokenType.ASSUME,
        "require": TokenType.REQUIRE,
        "is": TokenType.IS,
        "has": TokenType.HAS,
        "with": TokenType.WITH,
        "stable": TokenType.STABLE,
        "unstable": TokenType.UNSTABLE,
        "asymptotic": TokenType.ASYMPTOTIC,
        "exponential": TokenType.EXPONENTIAL,
        "marginal": TokenType.MARGINAL,
        "conditional": TokenType.CONDITIONAL,
        "kuramoto": TokenType.KURAMOTO,
        "winfree": TokenType.WINFREE,
        "stuart_landau": TokenType.STUART_LANDAU,
        "hopf": TokenType.HOPF,
        "prc": TokenType.PRC,
    }
    
    def __init__(self, source: str):
        """
        Initialize the lexer with source code.
        
        Args:
            source: The ELFIN DSL source code to tokenize
        """
        self.source = source
        self.tokens = []
        self.start = 0
        self.current = 0
        self.line = 1
        self.column = 1
    
    def tokenize(self) -> List[Token]:
        """
        Tokenize the source code.
        
        Returns:
            A list of tokens
        """
        while not self.is_at_end():
            self.start = self.current
            self.scan_token()
        
        # Add EOF token
        self.tokens.append(Token(TokenType.EOF, "", None, self.line, self.column))
        return self.tokens
    
    def is_at_end(self) -> bool:
        """Check if we've reached the end of the source code."""
        return self.current >= len(self.source)
    
    def advance(self) -> str:
        """
        Advance to the next character and return the current one.
        
        Returns:
            The current character
        """
        char = self.source[self.current]
        self.current += 1
        self.column += 1
        return char
    
    def peek(self) -> str:
        """
        Look at the current character without advancing.
        
        Returns:
            The current character, or '\0' if at the end
        """
        if self.is_at_end():
            return '\0'
        return self.source[self.current]
    
    def peek_next(self) -> str:
        """
        Look at the next character without advancing.
        
        Returns:
            The next character, or '\0' if at the end
        """
        if self.current + 1 >= len(self.source):
            return '\0'
        return self.source[self.current + 1]
    
    def match(self, expected: str) -> bool:
        """
        Check if the current character matches the expected one.
        If it does, advance and return True, otherwise return False.
        
        Args:
            expected: The expected character
            
        Returns:
            True if the current character matches, False otherwise
        """
        if self.is_at_end() or self.source[self.current] != expected:
            return False
        self.current += 1
        self.column += 1
        return True
    
    def add_token(self, type: TokenType, literal: Optional[object] = None):
        """
        Add a token to the list of tokens.
        
        Args:
            type: The token type
            literal: The literal value (optional)
        """
        text = self.source[self.start:self.current]
        self.tokens.append(Token(type, text, literal, self.line, self.column - len(text)))
    
    def scan_token(self):
        """Scan the next token from the source code."""
        char = self.advance()
        
        # Handle single-character tokens
        if char == ':':
            self.add_token(TokenType.COLON)
        elif char == ';':
            self.add_token(TokenType.SEMICOLON)
        elif char == ',':
            self.add_token(TokenType.COMMA)
        elif char == '.':
            self.add_token(TokenType.DOT)
        elif char == '{':
            self.add_token(TokenType.CURLY_OPEN)
        elif char == '}':
            self.add_token(TokenType.CURLY_CLOSE)
        elif char == '(':
            self.add_token(TokenType.PAREN_OPEN)
        elif char == ')':
            self.add_token(TokenType.PAREN_CLOSE)
        elif char == '[':
            self.add_token(TokenType.BRACKET_OPEN)
        elif char == ']':
            self.add_token(TokenType.BRACKET_CLOSE)
        elif char == '+':
            self.add_token(TokenType.PLUS)
        elif char == '*':
            self.add_token(TokenType.STAR)
        elif char == '/':
            self.add_token(TokenType.SLASH)
        elif char == '^':
            self.add_token(TokenType.CARET)
        
        # Handle potentially two-character tokens
        elif char == '-':
            if self.match('>'):
                self.add_token(TokenType.ARROW)
            else:
                self.add_token(TokenType.MINUS)
        elif char == '<':
            if self.match('-'):
                if self.match('>'):
                    self.add_token(TokenType.DOUBLE_ARROW)
                else:
                    # Error: incomplete double arrow
                    self.add_token(TokenType.ERROR, "Incomplete double arrow '<-'")
            elif self.match('='):
                self.add_token(TokenType.LESS_EQUAL)
            else:
                self.add_token(TokenType.LESS_THAN)
        elif char == '>':
            if self.match('='):
                self.add_token(TokenType.GREATER_EQUAL)
            else:
                self.add_token(TokenType.GREATER_THAN)
        elif char == '=':
            self.add_token(TokenType.EQUALS)
        
        # Handle whitespace
        elif char == ' ' or char == '\r' or char == '\t':
            # Ignore whitespace
            pass
        elif char == '\n':
            self.line += 1
            self.column = 1
        
        # Handle string literals
        elif char == '"':
            self.string()
        
        # Handle psi and phi symbols
        elif char == 'ψ' or (char == '\\' and self.peek() == 'p' and self.peek_next() == 's'):
            if char == '\\':
                # Skip "psi"
                self.advance()
                self.advance()
                self.advance()
            self.add_token(TokenType.PSI_SYMBOL, "ψ")
        elif char == 'φ' or (char == '\\' and self.peek() == 'p' and self.peek_next() == 'h'):
            if char == '\\':
                # Skip "phi"
                self.advance()
                self.advance()
                self.advance()
            self.add_token(TokenType.PHI_SYMBOL, "φ")
        
        # Handle numbers
        elif self.is_digit(char):
            self.number()
        
        # Handle identifiers and keywords
        elif self.is_alpha(char):
            self.identifier()
        
        else:
            # Error: unexpected character
            self.add_token(TokenType.ERROR, f"Unexpected character: {char}")
    
    def string(self):
        """Process a string literal."""
        while self.peek() != '"' and not self.is_at_end():
            if self.peek() == '\n':
                self.line += 1
                self.column = 1
            self.advance()
        
        if self.is_at_end():
            # Error: unterminated string
            self.add_token(TokenType.ERROR, "Unterminated string")
            return
        
        # Consume the closing "
        self.advance()
        
        # Extract the string value (without the quotes)
        value = self.source[self.start + 1:self.current - 1]
        self.add_token(TokenType.STRING, value)
    
    def number(self):
        """Process a number literal."""
        is_float = False
        
        # Consume digits
        while self.is_digit(self.peek()):
            self.advance()
        
        # Check for decimal point
        if self.peek() == '.' and self.is_digit(self.peek_next()):
            is_float = True
            # Consume the decimal point
            self.advance()
            
            # Consume digits after the decimal point
            while self.is_digit(self.peek()):
                self.advance()
        
        # Check for scientific notation
        if self.peek().lower() == 'e':
            is_float = True
            # Consume the 'e' or 'E'
            self.advance()
            
            # Consume optional sign
            if self.peek() in ['+', '-']:
                self.advance()
            
            # Must have at least one digit in the exponent
            if not self.is_digit(self.peek()):
                self.add_token(TokenType.ERROR, "Invalid scientific notation")
                return
            
            # Consume digits in the exponent
            while self.is_digit(self.peek()):
                self.advance()
        
        # Parse the number
        number_str = self.source[self.start:self.current]
        if is_float:
            value = float(number_str)
            self.add_token(TokenType.FLOAT, value)
        else:
            value = int(number_str)
            self.add_token(TokenType.NUMBER, value)
    
    def identifier(self):
        """Process an identifier or keyword."""
        while self.is_alphanumeric(self.peek()):
            self.advance()
        
        # Check if the identifier is a keyword
        text = self.source[self.start:self.current]
        token_type = self.KEYWORDS.get(text.lower(), TokenType.IDENTIFIER)
        
        self.add_token(token_type)
    
    def is_digit(self, char: str) -> bool:
        """Check if a character is a digit."""
        return '0' <= char <= '9'
    
    def is_alpha(self, char: str) -> bool:
        """Check if a character is alphabetic."""
        return ('a' <= char.lower() <= 'z') or char == '_'
    
    def is_alphanumeric(self, char: str) -> bool:
        """Check if a character is alphanumeric."""
        return self.is_digit(char) or self.is_alpha(char)


def tokenize(source: str) -> List[Token]:
    """
    Tokenize ELFIN DSL source code.
    
    Args:
        source: The ELFIN DSL source code
        
    Returns:
        A list of tokens
    """
    lexer = Lexer(source)
    return lexer.tokenize()
