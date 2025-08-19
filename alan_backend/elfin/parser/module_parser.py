"""
ELFIN Module-Aware Parser.

This module provides an extended parser that supports the ELFIN module system,
including imports and templates.
"""

from typing import List, Dict, Optional, Any, Tuple, Set
from pathlib import Path

from alan_backend.elfin.parser.lexer import TokenType, Token
from alan_backend.elfin.parser.parser import Parser, ParseError
from alan_backend.elfin.parser.module_ast import (
    ImportDecl, TemplateParamDecl, TemplateDecl, TemplateArgument,
    TemplateInstantiation, ModuleNode
)
from alan_backend.elfin.modules.resolver import ImportResolver
from alan_backend.elfin.modules.symbols import SymbolTable, Symbol, Scope
from alan_backend.elfin.modules.templates import TemplateRegistry, TemplateDefinition, TemplateParameter


class ModuleAwareParser(Parser):
    """
    A parser that is aware of modules and can resolve imports.
    
    This parser extends the base ELFIN parser with support for imports,
    templates, and symbol resolution.
    """
    
    def __init__(
        self,
        kiwis: List[Token],
        resolver: Optional[ImportResolver] = None,
        symbol_table: Optional[SymbolTable] = None,
        template_registry: Optional[TemplateRegistry] = None,
        file_path: Optional[str] = None
    ):
        """
        Initialize a module-aware parser.
        
        Args:
            kiwis: A list of kiwis from the lexer
            resolver: An import resolver (optional)
            symbol_table: A symbol table (optional)
            template_registry: A template registry (optional)
            file_path: The path to the file being parsed (optional)
        """
        super().__init__(kiwis)
        self.resolver = resolver or ImportResolver()
        self.symbol_table = symbol_table or SymbolTable()
        self.template_registry = template_registry or TemplateRegistry()
        self.file_path = file_path
        self.imports = []
        self.templates = []
        
        # Keep track of import sources to avoid duplicates
        self.imported_sources = set()
        
        # Stack of scopes for tracking the current parsing context
        self.scope_stack = []
        
        # Keep track of symbol references for validation
        self.symbol_references = []
    
    def parse(self) -> ModuleNode:
        """
        Parse the kiwis and return the module AST.
        
        Returns:
            The root node of the module AST
        """
        # Create a new scope for the module
        self.begin_scope("module")
        
        # Parse declarations
        declarations = []
        
        while not self.is_at_end():
            try:
                if self.match(TokenType.IMPORT):
                    # Parse import declaration
                    import_decl = self.import_declaration()
                    self.imports.append(import_decl)
                    
                    # Process the import immediately
                    self._process_import(import_decl)
                elif self.match(TokenType.TEMPLATE):
                    # Parse template declaration
                    template_decl = self.template_declaration()
                    self.templates.append(template_decl)
                    
                    # Register the template immediately
                    self._register_template(template_decl)
                else:
                    # Parse regular declaration
                    decl = self.declaration()
                    if decl:
                        declarations.append(decl)
            except ParseError as e:
                # Report error and synchronize
                print(f"Parse error: {e}")
                self.synchronize()
        
        # End the module scope
        self.end_scope()
        
        # Create and return the module node
        module = ModuleNode(
            path=self.file_path or "unknown",
            declarations=declarations,
            imports=self.imports,
            templates=self.templates
        )
        
        return module
    
    def begin_scope(self, name: str) -> None:
        """
        Begin a new scope for symbol declarations.
        
        Args:
            name: The name of the scope
        """
        self.symbol_table.enter_scope(name)
        self.scope_stack.append(name)
    
    def end_scope(self) -> None:
        """End the current scope."""
        if self.scope_stack:
            self.scope_stack.pop()
        self.symbol_table.exit_scope()
    
    def declare_symbol(self, name: str, symbol_type: str, value: Any = None, 
                     source: Optional[str] = None) -> Symbol:
        """
        Declare a symbol in the current scope.
        
        Args:
            name: The name of the symbol
            symbol_type: The type of the symbol
            value: The value of the symbol (optional)
            source: The source of the symbol (optional)
            
        Returns:
            The declared symbol
        """
        return self.symbol_table.define(name, symbol_type, value, source)
    
    def look_up_symbol(self, name: str, location: Tuple[int, int] = (0, 0)) -> Optional[Symbol]:
        """
        Look up a symbol by name.
        
        Args:
            name: The name of the symbol
            location: The (line, column) location for error reporting
            
        Returns:
            The symbol if found, None otherwise
            
        Note:
            This also adds a reference to the symbol for later validation.
        """
        symbol = self.symbol_table.lookup(name)
        
        # Record the reference for validation
        if symbol:
            self.symbol_references.append({
                "name": name,
                "symbol": symbol,
                "location": location,
                "scope": self.current_scope_name()
            })
        
        return symbol
    
    def current_scope_name(self) -> str:
        """Get the name of the current scope."""
        return self.scope_stack[-1] if self.scope_stack else "global"
    
    def _process_import(self, import_decl: ImportDecl) -> None:
        """
        Process an import declaration by resolving the imported module.
        
        Args:
            import_decl: The import declaration to process
        """
        # Skip if we've already imported this source
        if import_decl.source in self.imported_sources:
            return
        
        try:
            # Resolve the import
            source_path = Path(import_decl.source)
            resolved_path, module = self.resolver.resolve(source_path)
            
            # Import symbols
            for import_item in import_decl.imports:
                name = import_item["name"]
                alias = import_item["alias"] or name
                
                # Get the symbol from the module
                # In a real implementation, we would get this from the parsed module
                # Here we're just creating a placeholder
                symbol = Symbol(
                    name=name,
                    symbol_type="imported",
                    value=None,
                    source=str(resolved_path)
                )
                
                # Define the symbol in the current scope
                try:
                    self.declare_symbol(
                        name=alias,
                        symbol_type="imported",
                        value=symbol,
                        source=str(resolved_path)
                    )
                except ValueError as e:
                    # Symbol already defined, log a warning
                    print(f"Warning: {e}")
            
            # Mark as imported
            self.imported_sources.add(import_decl.source)
            
        except Exception as e:
            # Log the error but continue parsing
            print(f"Import error: {e}")
    
    def _register_template(self, template_decl: TemplateDecl) -> None:
        """
        Register a template declaration with the template registry.
        
        Args:
            template_decl: The template declaration to register
        """
        try:
            # Convert template parameters
            parameters = []
            for param in template_decl.parameters:
                parameters.append(
                    TemplateParameter(
                        name=param.name,
                        default_value=param.default_value,
                        parameter_type=param.param_type
                    )
                )
            
            # Create template definition
            template_def = TemplateDefinition(
                name=template_decl.name,
                parameters=parameters,
                body=template_decl.body,
                source=self.file_path
            )
            
            # Register with the template registry
            self.template_registry.register(template_def)
            
            # Also define the template in the symbol table
            self.declare_symbol(
                name=template_decl.name,
                symbol_type="template",
                value=template_def,
                source=self.file_path
            )
            
        except ValueError as e:
            # Template already registered, log a warning
            print(f"Warning: {e}")
    
    def import_declaration(self) -> ImportDecl:
        """
        Parse an import declaration.
        
        Returns:
            The parsed import declaration
        """
        # Get the location
        line = self.previous().line
        column = self.previous().column
        
        # Parse import items (single or multiple)
        imports = []
        
        # Handle single import
        if not self.check(TokenType.CURLY_OPEN):
            name = self.consume(TokenType.IDENTIFIER, "Expected import name").lexeme
            alias = None
            
            # Check for alias
            if self.match(TokenType.IDENTIFIER) and self.previous().lexeme.lower() == "as":
                alias = self.consume(TokenType.IDENTIFIER, "Expected alias").lexeme
            
            imports.append({"name": name, "alias": alias})
        else:
            # Handle multiple imports
            self.consume(TokenType.CURLY_OPEN, "Expected '{' for multiple imports")
            
            while not self.check(TokenType.CURLY_CLOSE) and not self.is_at_end():
                name = self.consume(TokenType.IDENTIFIER, "Expected import name").lexeme
                alias = None
                
                # Check for alias
                if self.match(TokenType.IDENTIFIER) and self.previous().lexeme.lower() == "as":
                    alias = self.consume(TokenType.IDENTIFIER, "Expected alias").lexeme
                
                imports.append({"name": name, "alias": alias})
                
                # Check for comma between imports
                if not self.match(TokenType.COMMA):
                    break
            
            self.consume(TokenType.CURLY_CLOSE, "Expected '}' after imports")
        
        # Parse the source module
        self.consume(TokenType.IDENTIFIER, "Expected 'from' keyword")
        if self.previous().lexeme.lower() != "from":
            self.error(self.previous(), "Expected 'from' keyword")
            
        source = self.consume(TokenType.STRING, "Expected source module path").literal
        
        # Create the import declaration
        import_decl = ImportDecl(
            imports=imports,
            source=source,
            location=(line, column)
        )
        
        return import_decl
    
    def template_declaration(self) -> TemplateDecl:
        """
        Parse a template declaration.
        
        Returns:
            The parsed template declaration
        """
        # Get the location
        line = self.previous().line
        column = self.previous().column
        
        # Get the template name
        name = self.consume(TokenType.IDENTIFIER, "Expected template name").lexeme
        
        # Create a new scope for the template
        self.begin_scope(f"template:{name}")
        
        # Parse parameters
        self.consume(TokenType.PAREN_OPEN, "Expected '(' after template name")
        parameters = []
        
        if not self.check(TokenType.PAREN_CLOSE):
            # Parse the first parameter
            param = self.parse_template_parameter()
            parameters.append(param)
            
            # Declare the parameter in the template scope
            self.declare_symbol(
                name=param.name,
                symbol_type="parameter",
                value=param.default_value,
                source=None
            )
            
            # Parse additional parameters
            while self.match(TokenType.COMMA):
                param = self.parse_template_parameter()
                parameters.append(param)
                
                # Declare the parameter in the template scope
                self.declare_symbol(
                    name=param.name,
                    symbol_type="parameter",
                    value=param.default_value,
                    source=None
                )
        
        self.consume(TokenType.PAREN_CLOSE, "Expected ')' after parameters")
        
        # Parse the template body
        self.consume(TokenType.CURLY_OPEN, "Expected '{' after template declaration")
        
        # In a real implementation, we would parse the body properly
        # For now, we'll just create a placeholder and skip the content
        body = self._parse_template_body()
        
        # End the template scope
        self.end_scope()
        
        # Create the template declaration
        template_decl = TemplateDecl(
            name=name,
            parameters=parameters,
            body=body,
            location=(line, column)
        )
        
        return template_decl
    
    def _parse_template_body(self) -> ModuleNode:
        """
        Parse a template body.
        
        Returns:
            A module node representing the template body
        """
        # Get the path or use a placeholder
        path = f"{self.file_path}#template" if self.file_path else "unknown#template"
        
        # For now, we'll just skip the content and create a placeholder
        body_declarations = []
        
        # Skip the body content
        nesting = 1
        while nesting > 0 and not self.is_at_end():
            if self.match(TokenType.CURLY_OPEN):
                nesting += 1
            elif self.match(TokenType.CURLY_CLOSE):
                nesting -= 1
            else:
                self.advance()
        
        # Create and return the body node
        return ModuleNode(
            path=path,
            declarations=body_declarations
        )
    
    def parse_template_parameter(self) -> TemplateParamDecl:
        """
        Parse a template parameter.
        
        Returns:
            The parsed template parameter
        """
        # Get the location
        line = self.peek().line
        column = self.peek().column
        
        # Get the parameter name
        name = self.consume(TokenType.IDENTIFIER, "Expected parameter name").lexeme
        
        # Check for type annotation
        param_type = None
        if self.match(TokenType.COLON):
            param_type = self.consume(TokenType.IDENTIFIER, "Expected type").lexeme
        
        # Check for default value
        default_value = None
        if self.match(TokenType.EQUALS):
            default_value = self.parse_property_value()
        
        # Create the parameter declaration
        param_decl = TemplateParamDecl(
            name=name,
            param_type=param_type,
            default_value=default_value,
            location=(line, column)
        )
        
        return param_decl
    
    def primary(self) -> Any:
        """
        Parse a primary expression. Override to handle template instantiation.
        
        Returns:
            The parsed expression
        """
        # Check for numbers, strings, etc.
        if self.match(TokenType.NUMBER) or self.match(TokenType.FLOAT):
            return super().primary()
        elif self.match(TokenType.STRING):
            return super().primary()
        elif self.match(TokenType.IDENTIFIER):
            name = self.previous().lexeme
            line = self.previous().line
            column = self.previous().column
            
            # Look up the symbol
            symbol = self.look_up_symbol(name, (line, column))
            
            # Check if it's a template instantiation
            if self.match(TokenType.PAREN_OPEN):
                # Check if the identifier refers to a template
                is_template = symbol and symbol.symbol_type == "template"
                
                # Parse arguments
                arguments = []
                
                if not self.check(TokenType.PAREN_CLOSE):
                    # Parse the first argument
                    arguments.append(self.parse_template_argument())
                    
                    # Parse additional arguments
                    while self.match(TokenType.COMMA):
                        arguments.append(self.parse_template_argument())
                
                self.consume(TokenType.PAREN_CLOSE, "Expected ')' after arguments")
                
                # Check for instance name
                instance_name = None
                if self.match(TokenType.COLON):
                    instance_name = self.consume(TokenType.IDENTIFIER, "Expected instance name").lexeme
                
                # If it's a template, create a template instantiation
                if is_template:
                    instance = TemplateInstantiation(
                        template_name=name,
                        arguments=arguments,
                        instance_name=instance_name,
                        location=(line, column)
                    )
                    
                    # Validate the arguments against the template
                    self._validate_template_instantiation(instance, symbol.value)
                    
                    return instance
                else:
                    # It's a function call
                    return super().primary()
            else:
                # It's just an identifier
                return super().primary()
        elif self.match(TokenType.PAREN_OPEN):
            return super().primary()
        
        # If we get here, we couldn't parse a primary expression
        return super().primary()
    
    def parse_template_argument(self) -> TemplateArgument:
        """
        Parse a template argument.
        
        Returns:
            The parsed template argument
        """
        # Get the location
        line = self.peek().line
        column = self.peek().column
        
        # Check if it's a named argument
        if self.match(TokenType.IDENTIFIER):
            name = self.previous().lexeme
            
            if self.match(TokenType.EQUALS):
                # It's a named argument
                value = self.parse_property_value()
                return TemplateArgument(
                    value=value,
                    name=name,
                    location=(line, column)
                )
            
            # It's not a named argument, backtrack
            self.current -= 1
        
        # It's a positional argument
        value = self.parse_property_value()
        return TemplateArgument(
            value=value,
            location=(line, column)
        )
    
    def _validate_template_instantiation(self, 
                                      instance: TemplateInstantiation, 
                                      template_def: TemplateDefinition) -> None:
        """
        Validate that a template instantiation matches the template definition.
        
        Args:
            instance: The template instantiation
            template_def: The template definition
            
        Raises:
            ParseError: If the instantiation is invalid
        """
        if not template_def:
            return
        
        # Get the template parameters
        params = template_def.parameters
        
        # Track which parameters have been provided
        provided_params = set()
        
        # Check positional arguments
        positional_args = [arg for arg in instance.arguments if arg.name is None]
        
        for i, arg in enumerate(positional_args):
            if i < len(params):
                param = params[i]
                provided_params.add(param.name)
                
                # Check type compatibility (in a real implementation)
                # For now, we'll just assume they're compatible
            else:
                self.error(self.previous(), f"Too many positional arguments for template '{template_def.name}'")
        
        # Check named arguments
        for arg in instance.arguments:
            if arg.name:
                # Check if the parameter exists
                param = template_def.get_parameter(arg.name)
                if not param:
                    self.error(self.previous(), f"Unknown parameter '{arg.name}' in template '{template_def.name}'")
                
                provided_params.add(arg.name)
                
                # Check type compatibility (in a real implementation)
                # For now, we'll just assume they're compatible
        
        # Check that all required parameters are provided
        for param in params:
            if param.default_value is None and param.name not in provided_params:
                self.error(self.previous(), f"Missing required parameter '{param.name}' in template '{template_def.name}'")


def parse_elfin_module(
    source: str,
    file_path: Optional[str] = None,
    resolver: Optional[ImportResolver] = None,
    symbol_table: Optional[SymbolTable] = None,
    template_registry: Optional[TemplateRegistry] = None
) -> ModuleNode:
    """
    Parse ELFIN DSL source code into a module AST.
    
    Args:
        source: The ELFIN DSL source code
        file_path: The path to the file being parsed (optional)
        resolver: An import resolver (optional)
        symbol_table: A symbol table (optional)
        template_registry: A template registry (optional)
        
    Returns:
        The root node of the module AST
    """
    from alan_backend.elfin.parser.lexer import tokenize
    
    kiwis = tokenize(source)
    parser = ModuleAwareParser(
        kiwis,
        resolver=resolver,
        symbol_table=symbol_table,
        template_registry=template_registry,
        file_path=file_path
    )
    return parser.parse()
