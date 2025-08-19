"""
ELFIN DSL Parser

This module provides a recursive descent parser for the ELFIN DSL language.
It takes a stream of kiwis from the lexer and produces an abstract syntax tree.
"""

from typing import List, Optional, Dict, Any, Tuple
import uuid

from alan_backend.elfin.parser.lexer import TokenType, Token, tokenize
from alan_backend.elfin.parser.ast import (
    ASTNode, Program, ConceptDeclaration, RelationDeclaration,
    PsiModeDeclaration, StabilityDeclaration, KoopmanDeclaration,
    AgentDirective, GoalDeclaration, AssumptionDeclaration,
    Property, Expression, BinaryExpression, UnaryExpression,
    LiteralExpression, IdentifierExpression, PsiModeExpression,
    FunctionCallExpression, PsiModeComponent, PhaseCoupling,
    EigenFunction, PhaseMapping, OscillatorParameters,
    ConceptType, RelationType, StabilityType, OscillatorType,
    BinaryOperator, UnaryOperator, ExpressionType
)


class ParseError(Exception):
    """Exception raised for parsing errors."""
    def __init__(self, kiwi: Token, message: str):
        self.kiwi = kiwi
        self.message = message
        super().__init__(f"Line {kiwi.line}, Column {kiwi.column}: {message}")


class Parser:
    """
    A recursive descent parser for the ELFIN DSL.
    
    This parser transforms a stream of kiwis into an abstract syntax tree (AST)
    that can be used by the compiler to generate a LocalConceptNetwork.
    """
    
    def __init__(self, kiwis: List[Token]):
        """
        Initialize the parser with a list of kiwis.
        
        Args:
            kiwis: A list of kiwis from the lexer
        """
        self.kiwis = kiwis
        self.current = 0
    
    def parse(self) -> Program:
        """
        Parse the kiwis and return the AST.
        
        Returns:
            The root node of the abstract syntax tree
        
        Raises:
            ParseError: If the kiwis do not conform to the ELFIN grammar
        """
        program = Program(declarations=[])
        
        while not self.is_at_end():
            try:
                declaration = self.declaration()
                if declaration:
                    program.declarations.append(declaration)
            except ParseError as e:
                # Report error and synchronize
                print(f"Parse error: {e}")
                self.synchronize()
        
        return program
    
    def declaration(self) -> Optional[ASTNode]:
        """Parse a declaration."""
        if self.match(TokenType.CONCEPT):
            return self.concept_declaration()
        elif self.match(TokenType.RELATION):
            return self.relation_declaration()
        elif self.match(TokenType.AGENT):
            return self.agent_directive()
        elif self.match(TokenType.GOAL):
            return self.goal_declaration()
        elif self.match(TokenType.ASSUME):
            return self.assumption_declaration()
        else:
            self.error(self.peek(), "Expected declaration")
            return None
    
    def concept_declaration(self) -> ConceptDeclaration:
        """Parse a concept declaration."""
        # Get the concept name
        name_kiwi = self.consume(TokenType.IDENTIFIER, "Expected concept name")
        name = name_kiwi.lexeme
        
        # Check if there's a type specified
        concept_type = ConceptType.ENTITY  # Default type
        if self.match(TokenType.COLON):
            type_kiwi = self.consume(TokenType.IDENTIFIER, "Expected concept type")
            try:
                # Convert the type name to a ConceptType enum
                concept_type = ConceptType[type_kiwi.lexeme.upper()]
            except KeyError:
                self.error(type_kiwi, f"Unknown concept type: {type_kiwi.lexeme}")
        
        # Parse description if present
        description = None
        if self.match(TokenType.STRING):
            description = self.previous().literal
        
        # Parse the concept body if present
        properties = {}
        psi_mode = None
        stability = None
        koopman = None
        
        if self.match(TokenType.CURLY_OPEN):
            while not self.check(TokenType.CURLY_CLOSE) and not self.is_at_end():
                if self.match(TokenType.PSI_MODE):
                    psi_mode = self.psi_mode_declaration()
                elif self.match(TokenType.STABILITY):
                    stability = self.stability_declaration()
                elif self.match(TokenType.KOOPMAN):
                    koopman = self.koopman_declaration()
                else:
                    # Parse a property
                    property_name = self.consume(TokenType.IDENTIFIER, "Expected property name").lexeme
                    self.consume(TokenType.EQUALS, "Expected '=' after property name")
                    property_value = self.parse_property_value()
                    properties[property_name] = property_value
                
                # Expect a semicolon or comma between properties
                if not self.match(TokenType.SEMICOLON) and not self.match(TokenType.COMMA):
                    break
            
            self.consume(TokenType.CURLY_CLOSE, "Expected '}' after concept body")
        
        # Create and return the concept declaration
        concept = ConceptDeclaration(
            name=name,
            concept_type=concept_type,
            description=description,
            properties=properties,
            psi_mode=psi_mode,
            stability=stability,
            koopman=koopman,
            line=name_kiwi.line,
            column=name_kiwi.column
        )
        
        return concept
    
    def relation_declaration(self) -> RelationDeclaration:
        """Parse a relation declaration."""
        # Get the source concept
        source = self.consume(TokenType.IDENTIFIER, "Expected source concept").lexeme
        
        # Get the relation type
        relation_type = None
        if self.match(TokenType.IS) and self.match(TokenType.IDENTIFIER) and self.previous().lexeme == "a":
            relation_type = RelationType.IS_A
        elif self.match(TokenType.HAS) and self.match(TokenType.IDENTIFIER) and self.previous().lexeme == "a":
            relation_type = RelationType.HAS_A
        elif self.match(TokenType.IDENTIFIER):
            type_name = self.previous().lexeme.upper()
            if type_name == "PART_OF":
                relation_type = RelationType.PART_OF
            elif type_name == "AFFECTS":
                relation_type = RelationType.AFFECTS
            elif type_name == "CAUSES":
                relation_type = RelationType.CAUSES
            elif type_name == "REQUIRES":
                relation_type = RelationType.REQUIRES
            elif type_name == "ASSOCIATES_WITH":
                relation_type = RelationType.ASSOCIATES_WITH
            elif type_name == "COUPLES_TO":
                relation_type = RelationType.COUPLES_TO
            elif type_name == "SYNCHRONIZES_WITH":
                relation_type = RelationType.SYNCHRONIZES_WITH
            elif type_name == "STABILIZES":
                relation_type = RelationType.STABILIZES
            elif type_name == "DESTABILIZES":
                relation_type = RelationType.DESTABILIZES
            else:
                self.error(self.previous(), f"Unknown relation type: {type_name}")
        
        if relation_type is None:
            self.error(self.peek(), "Expected relation type")
        
        # Get the target concept
        target = self.consume(TokenType.IDENTIFIER, "Expected target concept").lexeme
        
        # Parse optional weight
        weight = None
        if self.match(TokenType.PAREN_OPEN):
            weight = self.parse_numeric_value()
            self.consume(TokenType.PAREN_CLOSE, "Expected ')' after weight")
        
        # Parse the relation body if present
        properties = {}
        phase_coupling = None
        
        if self.match(TokenType.CURLY_OPEN):
            while not self.check(TokenType.CURLY_CLOSE) and not self.is_at_end():
                if self.match(TokenType.IDENTIFIER):
                    property_name = self.previous().lexeme
                    
                    if property_name == "phase_coupling":
                        self.consume(TokenType.EQUALS, "Expected '=' after property name")
                        
                        # Parse phase coupling
                        self.consume(TokenType.CURLY_OPEN, "Expected '{' for phase coupling")
                        coupling_strength = None
                        coupling_function = None
                        phase_lag = 0.0
                        bidirectional = False
                        
                        while not self.check(TokenType.CURLY_CLOSE) and not self.is_at_end():
                            pc_prop_name = self.consume(TokenType.IDENTIFIER, "Expected phase coupling property").lexeme
                            self.consume(TokenType.EQUALS, "Expected '=' after property name")
                            
                            if pc_prop_name == "coupling_strength":
                                coupling_strength = self.parse_numeric_value()
                            elif pc_prop_name == "coupling_function":
                                coupling_function = self.consume(TokenType.STRING, "Expected string for coupling function").literal
                            elif pc_prop_name == "phase_lag":
                                phase_lag = self.parse_numeric_value()
                            elif pc_prop_name == "bidirectional":
                                bidirectional = self.parse_boolean_value()
                            
                            # Expect a comma between properties
                            if not self.match(TokenType.COMMA):
                                break
                        
                        self.consume(TokenType.CURLY_CLOSE, "Expected '}' after phase coupling")
                        
                        if coupling_strength is not None:
                            phase_coupling = PhaseCoupling(
                                coupling_strength=coupling_strength,
                                coupling_function=coupling_function,
                                phase_lag=phase_lag,
                                bidirectional=bidirectional
                            )
                    else:
                        # Parse a regular property
                        self.consume(TokenType.EQUALS, "Expected '=' after property name")
                        properties[property_name] = self.parse_property_value()
                
                # Expect a semicolon or comma between properties
                if not self.match(TokenType.SEMICOLON) and not self.match(TokenType.COMMA):
                    break
            
            self.consume(TokenType.CURLY_CLOSE, "Expected '}' after relation body")
        
        # Create and return the relation declaration
        relation = RelationDeclaration(
            source=source,
            relation_type=relation_type,
            target=target,
            weight=weight,
            properties=properties,
            phase_coupling=phase_coupling,
            line=self.previous().line,
            column=self.previous().column
        )
        
        return relation
    
    def agent_directive(self) -> AgentDirective:
        """Parse an agent directive."""
        # Get the agent type
        agent_type = self.consume(TokenType.IDENTIFIER, "Expected agent type").lexeme
        
        # Get the directive
        self.consume(TokenType.COLON, "Expected ':' after agent type")
        directive = self.consume(TokenType.STRING, "Expected directive string").literal
        
        # Parse the directive body if present
        target_concept_ids = []
        parameters = {}
        trigger_condition = None
        
        if self.match(TokenType.CURLY_OPEN):
            while not self.check(TokenType.CURLY_CLOSE) and not self.is_at_end():
                if self.match(TokenType.IDENTIFIER):
                    property_name = self.previous().lexeme
                    
                    if property_name == "targets":
                        self.consume(TokenType.EQUALS, "Expected '=' after property name")
                        self.consume(TokenType.BRACKET_OPEN, "Expected '[' for targets")
                        
                        while not self.check(TokenType.BRACKET_CLOSE) and not self.is_at_end():
                            target_concept_ids.append(self.consume(TokenType.IDENTIFIER, "Expected concept ID").lexeme)
                            if not self.match(TokenType.COMMA):
                                break
                        
                        self.consume(TokenType.BRACKET_CLOSE, "Expected ']' after targets")
                    elif property_name == "trigger":
                        self.consume(TokenType.EQUALS, "Expected '=' after property name")
                        trigger_condition = self.expression()
                    else:
                        # Parse a parameter
                        self.consume(TokenType.EQUALS, "Expected '=' after parameter name")
                        parameters[property_name] = self.parse_property_value()
                
                # Expect a semicolon or comma between properties
                if not self.match(TokenType.SEMICOLON) and not self.match(TokenType.COMMA):
                    break
            
            self.consume(TokenType.CURLY_CLOSE, "Expected '}' after directive body")
        
        # Create and return the agent directive
        directive_node = AgentDirective(
            agent_type=agent_type,
            directive=directive,
            target_concept_ids=target_concept_ids,
            parameters=parameters,
            trigger_condition=trigger_condition,
            line=self.previous().line,
            column=self.previous().column
        )
        
        return directive_node
    
    def goal_declaration(self) -> GoalDeclaration:
        """Parse a goal declaration."""
        # Get the goal name
        name = self.consume(TokenType.IDENTIFIER, "Expected goal name").lexeme
        
        # Get the goal description
        self.consume(TokenType.COLON, "Expected ':' after goal name")
        description = self.consume(TokenType.STRING, "Expected goal description").literal
        
        # Parse the goal expression
        self.consume(TokenType.EQUALS, "Expected '=' after goal description")
        expr = self.expression()
        
        # Parse the goal body if present
        target_concept_ids = []
        priority = 1.0
        
        if self.match(TokenType.CURLY_OPEN):
            while not self.check(TokenType.CURLY_CLOSE) and not self.is_at_end():
                if self.match(TokenType.IDENTIFIER):
                    property_name = self.previous().lexeme
                    self.consume(TokenType.EQUALS, "Expected '=' after property name")
                    
                    if property_name == "targets":
                        self.consume(TokenType.BRACKET_OPEN, "Expected '[' for targets")
                        
                        while not self.check(TokenType.BRACKET_CLOSE) and not self.is_at_end():
                            target_concept_ids.append(self.consume(TokenType.IDENTIFIER, "Expected concept ID").lexeme)
                            if not self.match(TokenType.COMMA):
                                break
                        
                        self.consume(TokenType.BRACKET_CLOSE, "Expected ']' after targets")
                    elif property_name == "priority":
                        priority = self.parse_numeric_value()
                
                # Expect a semicolon or comma between properties
                if not self.match(TokenType.SEMICOLON) and not self.match(TokenType.COMMA):
                    break
            
            self.consume(TokenType.CURLY_CLOSE, "Expected '}' after goal body")
        
        # Create and return the goal declaration
        goal = GoalDeclaration(
            name=name,
            description=description,
            expression=expr,
            target_concept_ids=target_concept_ids,
            priority=priority,
            line=self.previous().line,
            column=self.previous().column
        )
        
        return goal
    
    def assumption_declaration(self) -> AssumptionDeclaration:
        """Parse an assumption declaration."""
        # Get the assumption name
        name = self.consume(TokenType.IDENTIFIER, "Expected assumption name").lexeme
        
        # Get the assumption description
        self.consume(TokenType.COLON, "Expected ':' after assumption name")
        description = self.consume(TokenType.STRING, "Expected assumption description").literal
        
        # Parse the assumption expression
        self.consume(TokenType.EQUALS, "Expected '=' after assumption description")
        expr = self.expression()
        
        # Parse the assumption body if present
        confidence = 1.0
        validated = False
        
        if self.match(TokenType.CURLY_OPEN):
            while not self.check(TokenType.CURLY_CLOSE) and not self.is_at_end():
                if self.match(TokenType.IDENTIFIER):
                    property_name = self.previous().lexeme
                    self.consume(TokenType.EQUALS, "Expected '=' after property name")
                    
                    if property_name == "confidence":
                        confidence = self.parse_numeric_value()
                    elif property_name == "validated":
                        validated = self.parse_boolean_value()
                
                # Expect a semicolon or comma between properties
                if not self.match(TokenType.SEMICOLON) and not self.match(TokenType.COMMA):
                    break
            
            self.consume(TokenType.CURLY_CLOSE, "Expected '}' after assumption body")
        
        # Create and return the assumption declaration
        assumption = AssumptionDeclaration(
            name=name,
            description=description,
            expression=expr,
            confidence=confidence,
            validated=validated,
            line=self.previous().line,
            column=self.previous().column
        )
        
        return assumption
    
    def expression(self) -> Expression:
        """Parse an expression."""
        return self.logical_or()
    
    def logical_or(self) -> Expression:
        """Parse a logical OR expression."""
        expr = self.logical_and()
        
        while self.match(TokenType.IDENTIFIER) and self.previous().lexeme.lower() == "or":
            right = self.logical_and()
            expr = BinaryExpression(
                operator=BinaryOperator.OR,
                left=expr,
                right=right,
                line=self.previous().line,
                column=self.previous().column
            )
        
        return expr
    
    def logical_and(self) -> Expression:
        """Parse a logical AND expression."""
        expr = self.equality()
        
        while self.match(TokenType.IDENTIFIER) and self.previous().lexeme.lower() == "and":
            right = self.equality()
            expr = BinaryExpression(
                operator=BinaryOperator.AND,
                left=expr,
                right=right,
                line=self.previous().line,
                column=self.previous().column
            )
        
        return expr
    
    def equality(self) -> Expression:
        """Parse an equality expression."""
        expr = self.comparison()
        
        while self.match(TokenType.EQUALS) or (self.match(TokenType.IDENTIFIER) and self.previous().lexeme.lower() == "not" and self.match(TokenType.EQUALS)):
            operator = BinaryOperator.EQUALS if self.previous().type == TokenType.EQUALS else BinaryOperator.NOT_EQUALS
            right = self.comparison()
            expr = BinaryExpression(
                operator=operator,
                left=expr,
                right=right,
                line=self.previous().line,
                column=self.previous().column
            )
        
        return expr
    
    def comparison(self) -> Expression:
        """Parse a comparison expression."""
        expr = self.relation()
        
        while self.match(TokenType.LESS_THAN) or self.match(TokenType.GREATER_THAN) or self.match(TokenType.LESS_EQUAL) or self.match(TokenType.GREATER_EQUAL):
            operator_type = self.previous().type
            if operator_type == TokenType.LESS_THAN:
                operator = BinaryOperator.LESS
            elif operator_type == TokenType.GREATER_THAN:
                operator = BinaryOperator.GREATER
            elif operator_type == TokenType.LESS_EQUAL:
                operator = BinaryOperator.LESS_EQUALS
            else:  # TokenType.GREATER_EQUAL
                operator = BinaryOperator.GREATER_EQUALS
            
            right = self.relation()
            expr = BinaryExpression(
                operator=operator,
                left=expr,
                right=right,
                line=self.previous().line,
                column=self.previous().column
            )
        
        return expr
    
    def relation(self) -> Expression:
        """Parse a relation expression."""
        expr = self.term()
        
        while (self.match(TokenType.IDENTIFIER) and self.previous().lexeme.lower() in ["couples", "synchronizes", "implies"]):
            relation_type = self.previous().lexeme.lower()
            if relation_type == "couples":
                operator = BinaryOperator.COUPLES
            elif relation_type == "synchronizes":
                operator = BinaryOperator.SYNCHRONIZES
            else:  # "implies"
                operator = BinaryOperator.IMPLIES
            
            right = self.term()
            expr = BinaryExpression(
                operator=operator,
                left=expr,
                right=right,
                line=self.previous().line,
                column=self.previous().column
            )
        
        return expr
    
    def term(self) -> Expression:
        """Parse a term expression."""
        expr = self.factor()
        
        while self.match(TokenType.PLUS) or self.match(TokenType.MINUS):
            operator_type = self.previous().type
            operator = BinaryOperator.PLUS if operator_type == TokenType.PLUS else BinaryOperator.MINUS
            right = self.factor()
            expr = BinaryExpression(
                operator=operator,
                left=expr,
                right=right,
                line=self.previous().line,
                column=self.previous().column
            )
        
        return expr
    
    def factor(self) -> Expression:
        """Parse a factor expression."""
        expr = self.power()
        
        while self.match(TokenType.STAR) or self.match(TokenType.SLASH):
            operator_type = self.previous().type
            operator = BinaryOperator.MULTIPLY if operator_type == TokenType.STAR else BinaryOperator.DIVIDE
            right = self.power()
            expr = BinaryExpression(
                operator=operator,
                left=expr,
                right=right,
                line=self.previous().line,
                column=self.previous().column
            )
        
        return expr
    
    def power(self) -> Expression:
        """Parse a power expression."""
        expr = self.unary()
        
        while self.match(TokenType.CARET):
            right = self.unary()
            expr = BinaryExpression(
                operator=BinaryOperator.POWER,
                left=expr,
                right=right,
                line=self.previous().line,
                column=self.previous().column
            )
        
        return expr
    
    def unary(self) -> Expression:
        """Parse a unary expression."""
        if self.match(TokenType.MINUS):
            operand = self.unary()
            return UnaryExpression(
                operator=UnaryOperator.NEGATIVE,
                operand=operand,
                line=self.previous().line,
                column=self.previous().column
            )
        elif self.match(TokenType.IDENTIFIER) and self.previous().lexeme.lower() == "not":
            operand = self.unary()
            return UnaryExpression(
                operator=UnaryOperator.NOT,
                operand=operand,
                line=self.previous().line,
                column=self.previous().column
            )
        elif self.match(TokenType.IDENTIFIER) and self.previous().lexeme.lower() == "stable":
            operand = self.unary()
            return UnaryExpression(
                operator=UnaryOperator.STABLE,
                operand=operand,
                line=self.previous().line,
                column=self.previous().column
            )
        elif self.match(TokenType.IDENTIFIER) and self.previous().lexeme.lower() == "unstable":
            operand = self.unary()
            return UnaryExpression(
                operator=UnaryOperator.UNSTABLE,
                operand=operand,
                line=self.previous().line,
                column=self.previous().column
            )
        elif self.match(TokenType.LYAPUNOV):
            operand = self.unary()
            return UnaryExpression(
                operator=UnaryOperator.LYAPUNOV,
                operand=operand,
                line=self.previous().line,
                column=self.previous().column
            )
        
        return self.psi_mode()
    
    def psi_mode(self) -> Expression:
        """Parse a ψ-mode expression."""
        if self.match(TokenType.PSI_SYMBOL) or (self.match(TokenType.IDENTIFIER) and self.previous().lexeme.lower() == "psi"):
            mode_index = None
            subject = None
            
            # Check for mode index in parentheses
            if self.match(TokenType.PAREN_OPEN):
                mode_index = self.parse_numeric_value()
                self.consume(TokenType.PAREN_CLOSE, "Expected ')' after mode index")
            
            # Check for optional subject
            if self.match(TokenType.PAREN_OPEN):
                subject = self.consume(TokenType.IDENTIFIER, "Expected concept identifier").lexeme
                self.consume(TokenType.PAREN_CLOSE, "Expected ')' after subject")
            
            return PsiModeExpression(
                mode_index=mode_index if mode_index is not None else 0,
                subject=subject,
                line=self.previous().line,
                column=self.previous().column
            )
        
        return self.primary()
    
    def primary(self) -> Expression:
        """Parse a primary expression."""
        if self.match(TokenType.NUMBER) or self.match(TokenType.FLOAT):
            return LiteralExpression(
                value=self.previous().literal,
                line=self.previous().line,
                column=self.previous().column
            )
        elif self.match(TokenType.STRING):
            return LiteralExpression(
                value=self.previous().literal,
                line=self.previous().line,
                column=self.previous().column
            )
        elif self.match(TokenType.IDENTIFIER):
            name = self.previous().lexeme
            
            # Check if it's a function call
            if self.match(TokenType.PAREN_OPEN):
                arguments = []
                
                if not self.check(TokenType.PAREN_CLOSE):
                    # Parse arguments
                    arguments.append(self.expression())
                    while self.match(TokenType.COMMA):
                        arguments.append(self.expression())
                
                self.consume(TokenType.PAREN_CLOSE, "Expected ')' after function arguments")
                
                return FunctionCallExpression(
                    function_name=name,
                    arguments=arguments,
                    line=self.previous().line,
                    column=self.previous().column
                )
            
            # It's just an identifier
            return IdentifierExpression(
                name=name,
                line=self.previous().line,
                column=self.previous().column
            )
        elif self.match(TokenType.PAREN_OPEN):
            expr = self.expression()
            self.consume(TokenType.PAREN_CLOSE, "Expected ')' after expression")
            return expr
        
        self.error(self.peek(), "Expected expression")
        return None
    
    def parse_property_value(self) -> Any:
        """Parse a property value."""
        if self.match(TokenType.NUMBER) or self.match(TokenType.FLOAT):
            return self.previous().literal
        elif self.match(TokenType.STRING):
            return self.previous().literal
        elif self.match(TokenType.IDENTIFIER):
            lexeme = self.previous().lexeme.lower()
            if lexeme == "true":
                return True
            elif lexeme == "false":
                return False
            else:
                return self.previous().lexeme
        elif self.match(TokenType.BRACKET_OPEN):
            # Parse an array
            values = []
            if not self.check(TokenType.BRACKET_CLOSE):
                values.append(self.parse_property_value())
                while self.match(TokenType.COMMA):
                    values.append(self.parse_property_value())
            
            self.consume(TokenType.BRACKET_CLOSE, "Expected ']' after array")
            return values
        elif self.match(TokenType.CURLY_OPEN):
            # Parse an object
            obj = {}
            if not self.check(TokenType.CURLY_CLOSE):
                key = self.consume(TokenType.IDENTIFIER, "Expected property name").lexeme
                self.consume(TokenType.COLON, "Expected ':' after property name")
                value = self.parse_property_value()
                obj[key] = value
                
                while self.match(TokenType.COMMA):
                    key = self.consume(TokenType.IDENTIFIER, "Expected property name").lexeme
                    self.consume(TokenType.COLON, "Expected ':' after property name")
                    value = self.parse_property_value()
                    obj[key] = value
            
            self.consume(TokenType.CURLY_CLOSE, "Expected '}' after object")
            return obj
        
        self.error(self.peek(), "Expected property value")
        return None
    
    def parse_numeric_value(self) -> float:
        """Parse a numeric value."""
        if self.match(TokenType.NUMBER) or self.match(TokenType.FLOAT):
            return float(self.previous().literal)
        
        self.error(self.peek(), "Expected numeric value")
        return 0.0
    
    def parse_boolean_value(self) -> bool:
        """Parse a boolean value."""
        if self.match(TokenType.IDENTIFIER):
            lexeme = self.previous().lexeme.lower()
            if lexeme == "true":
                return True
            elif lexeme == "false":
                return False
        
        self.error(self.peek(), "Expected boolean value (true or false)")
        return False
    
    def is_at_end(self) -> bool:
        """Check if we've reached the end of the kiwis."""
        return self.peek().type == TokenType.EOF
    
    def peek(self) -> Token:
        """Return the current kiwi without advancing."""
        return self.kiwis[self.current]
    
    def previous(self) -> Token:
        """Return the previously consumed kiwi."""
        return self.kiwis[self.current - 1]
    
    def advance(self) -> Token:
        """Advance to the next kiwi and return the previous one."""
        if not self.is_at_end():
            self.current += 1
        return self.previous()
    
    def check(self, type: TokenType) -> bool:
        """Check if the current kiwi has the given type."""
        if self.is_at_end():
            return False
        return self.peek().type == type
    
    def match(self, type: TokenType) -> bool:
        """
        Check if the current kiwi has the given type.
        If it does, consume it and return True. Otherwise, return False.
        """
        if self.check(type):
            self.advance()
            return True
        return False
    
    def consume(self, type: TokenType, message: str) -> Token:
        """
        Consume the current kiwi if it has the expected type.
        Otherwise, raise a ParseError.
        
        Args:
            type: The expected token type
            message: The error message if the token doesn't match
            
        Returns:
            The consumed token
        """
        if self.check(type):
            return self.advance()
        
        self.error(self.peek(), message)
    
    def error(self, kiwi: Token, message: str) -> ParseError:
        """
        Create and raise a parse error.
        
        Args:
            kiwi: The token at which the error occurred
            message: The error message
            
        Returns:
            A ParseError (though actually raises it)
        """
        error = ParseError(kiwi, message)
        raise error
    
    def synchronize(self):
        """
        Skip tokens until the next statement boundary.
        This is used for error recovery.
        """
        self.advance()
        
        while not self.is_at_end():
            # Skip until we find a semicolon or a keyword that starts a new declaration
            if self.previous().type == TokenType.SEMICOLON:
                return
                
            if self.peek().type in [
                TokenType.CONCEPT,
                TokenType.RELATION,
                TokenType.AGENT,
                TokenType.GOAL,
                TokenType.ASSUME
            ]:
                return
                
            self.advance()


def parse_elfin(source: str) -> Program:
    """
    Parse ELFIN DSL source code into an abstract syntax tree.
    
    Args:
        source: The ELFIN DSL source code
        
    Returns:
        The root node of the abstract syntax tree
    """
    kiwis = tokenize(source)
    parser = Parser(kiwis)
    return parser.parse()
