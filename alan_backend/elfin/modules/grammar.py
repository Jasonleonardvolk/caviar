"""
ELFIN Module System Grammar

This module defines the grammar extensions for the module system, including
import statements and template instantiation.
"""

# Import statement grammar
IMPORT_GRAMMAR = r"""
// Import statement
import_statement = "import" whitespace import_items whitespace "from" whitespace string_literal ";"

// Import items can be a single module or a list of items
import_items = single_import / multi_import

// Single import with optional alias
single_import = identifier (whitespace "as" whitespace identifier)?

// Multiple imports in curly braces
multi_import = "{" whitespace? import_list whitespace? "}"

// Comma-separated list of identifiers with optional aliases
import_list = import_item (whitespace? "," whitespace? import_item)*

// Single import item
import_item = identifier (whitespace "as" whitespace identifier)?
"""

# Grammar for template instantiation
TEMPLATE_GRAMMAR = r"""
// Template instantiation
template_instantiation = identifier "(" whitespace? template_args? whitespace? ")"

// Template arguments
template_args = template_arg (whitespace? "," whitespace? template_arg)*

// Single template argument
template_arg = named_arg / positional_arg

// Named argument (name=value)
named_arg = identifier whitespace? "=" whitespace? expression

// Positional argument (just value)
positional_arg = expression
"""

# Combined grammar for module system
MODULE_SYSTEM_GRAMMAR = f"""
{IMPORT_GRAMMAR}

{TEMPLATE_GRAMMAR}
"""


def integrate_with_main_grammar(main_grammar):
    """
    Integrate the module system grammar with the main ELFIN grammar.
    
    Args:
        main_grammar: The main ELFIN grammar string
        
    Returns:
        The combined grammar string
    """
    # This is a placeholder for now - when we have the actual main grammar,
    # we'll need to add import_statement to the top-level statements and
    # add template_instantiation to the expression types.
    return f"""
{main_grammar}

// Module system extensions
{MODULE_SYSTEM_GRAMMAR}
"""


# Parser for import statements
def parse_import_statement(statement):
    """
    Parse an import statement and extract the imported modules and source.
    
    Args:
        statement: The import statement string
        
    Returns:
        A dictionary with the following keys:
        - 'type': 'import'
        - 'imports': List of imported items as dictionaries with 'name' and 'alias'
        - 'source': The source module path
        
    Example:
        >>> parse_import_statement('import Controller from "controller.elfin";')
        {
            'type': 'import',
            'imports': [{'name': 'Controller', 'alias': None}],
            'source': 'controller.elfin'
        }
        
        >>> parse_import_statement('import { Vector3, Matrix3 as Mat3 } from "math/linear.elfin";')
        {
            'type': 'import',
            'imports': [
                {'name': 'Vector3', 'alias': None},
                {'name': 'Matrix3', 'alias': 'Mat3'}
            ],
            'source': 'math/linear.elfin'
        }
    """
    # This is a placeholder implementation - in a real parser we would use
    # a proper parsing library like PEG or LALR. For now, we'll use a simple
    # regex-based approach for demonstration purposes.
    import re
    
    # Match single import with optional alias
    single_import_pattern = r'import\s+(\w+)(?:\s+as\s+(\w+))?\s+from\s+"([^"]+)"\s*;'
    single_match = re.match(single_import_pattern, statement)
    if single_match:
        name, alias, source = single_match.groups()
        return {
            'type': 'import',
            'imports': [{'name': name, 'alias': alias}],
            'source': source
        }
    
    # Match multi-import with optional aliases
    multi_import_pattern = r'import\s+{\s*(.*?)\s*}\s+from\s+"([^"]+)"\s*;'
    multi_match = re.match(multi_import_pattern, statement)
    if multi_match:
        items_str, source = multi_match.groups()
        
        # Parse individual items
        items = []
        for item in re.split(r'\s*,\s*', items_str):
            item_match = re.match(r'(\w+)(?:\s+as\s+(\w+))?', item)
            if item_match:
                name, alias = item_match.groups()
                items.append({'name': name, 'alias': alias})
        
        return {
            'type': 'import',
            'imports': items,
            'source': source
        }
    
    # No match
    return None


def extract_imports_from_module(content):
    """
    Extract import statements from a module's content.
    
    Args:
        content: The module content string
        
    Returns:
        A list of parsed import statements
    """
    # This is a placeholder implementation - in a real parser we would extract
    # imports as part of the full parsing process.
    import re
    
    imports = []
    
    # Find import statements
    import_pattern = r'import\s+(?:{\s*.*?\s*}|\w+(?:\s+as\s+\w+)?)\s+from\s+"[^"]+"\s*;'
    for match in re.finditer(import_pattern, content):
        import_stmt = match.group(0)
        parsed = parse_import_statement(import_stmt)
        if parsed:
            imports.append(parsed)
    
    return imports
