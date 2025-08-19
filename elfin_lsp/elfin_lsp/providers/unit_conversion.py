"""
Unit conversion quick-fix for the ELFIN language server.

This module provides code actions for unit conversion, such as:
- Converting centimeters to meters when added together
- Converting degrees to radians when used in trig functions
"""

import logging
import re
from typing import List, Optional, Dict

from pygls.server import LanguageServer
from elfin_lsp.protocol import Position, Range
from pygls.capabilities import types

# Import types directly from the public API
CodeAction = types.CodeAction
CodeActionKind = types.CodeActionKind
TextEdit = types.TextEdit
WorkspaceEdit = types.WorkspaceEdit
CodeActionParams = types.CodeActionParams

logger = logging.getLogger(__name__)

# Common unit conversion relationships
UNIT_CONVERSIONS = {
    # Length
    ("cm", "m"): ("toMeters", 0.01),
    ("mm", "m"): ("toMeters", 0.001),
    ("km", "m"): ("toMeters", 1000.0),
    # Angle
    ("deg", "rad"): ("toRadians", 0.017453292519943295),  # pi/180
    # Time
    ("min", "s"): ("toSeconds", 60.0),
    ("hr", "s"): ("toSeconds", 3600.0),
    # Mass
    ("g", "kg"): ("toKilograms", 0.001),
    # Temperature
    ("c", "k"): ("toKelvin", None),  # Non-linear conversion
    # Force
    ("N", "lb"): ("toNewtons", 4.4482216152605),
}

def _detect_unit_conversion_issues(diag: types.Diagnostic) -> Optional[Dict]:
    """
    Detect unit conversion issues from dimensional mismatch diagnostics.
    
    Args:
        diag: The diagnostic to check
        
    Returns:
        Information about the unit conversion needed, or None if not applicable
    """
    # Check if the diagnostic is related to dimensional mismatches
    if not (hasattr(diag, 'code') and diag.code == "DIM_MISMATCH"):
        return None
    
    # Extract units from the diagnostic message
    # Expected message format: "Cannot add [cm] and [m]" or similar
    message = diag.message
    match = re.search(r"Cannot (?:add|compare|mix) \[([a-zA-Z]+)\] and \[([a-zA-Z]+)\]", message)
    
    if not match:
        return None
    
    unit1 = match.group(1)
    unit2 = match.group(2)
    
    # Check if this is a known unit conversion
    conversion = None
    if (unit1, unit2) in UNIT_CONVERSIONS:
        conversion = UNIT_CONVERSIONS[(unit1, unit2)]
        from_unit, to_unit = unit1, unit2
    elif (unit2, unit1) in UNIT_CONVERSIONS:
        conversion = UNIT_CONVERSIONS[(unit2, unit1)]
        from_unit, to_unit = unit2, unit1
    
    if not conversion:
        return None
    
    converter_name, factor = conversion
    
    return {
        "from_unit": from_unit,
        "to_unit": to_unit,
        "converter": converter_name,
        "factor": factor,
        "range": diag.range
    }

def _create_conversion_edit(document, issue: Dict) -> Optional[WorkspaceEdit]:
    """
    Create a workspace edit for unit conversion.
    
    Args:
        document: The document to edit
        issue: Information about the unit conversion needed
        
    Returns:
        A workspace edit for unit conversion, or None if not possible
    """
    try:
        # Get the range where the conversion is needed
        rng = issue["range"]
        line = document.lines[rng.start.line]
        
        # Find the token to wrap with the converter
        # This is a simplified approach - a real implementation would use
        # the AST to find the exact token to convert
        
        # For demonstration, we'll assume the token with unit mismatch is
        # a simple numeric literal or variable near the diagnostic position
        token_match = re.search(r'(\d+(\.\d+)?|\w+)', line[rng.start.character:])
        if not token_match:
            return None
        
        token = token_match.group(1)
        token_start = rng.start.character + token_match.start()
        token_end = rng.start.character + token_match.end()
        
        # Create the replacement text
        converter = issue["converter"]
        replacement = f"{converter}({token})"
        
        # Create a text edit
        text_edit = TextEdit(
            range=Range(
                start=Position(line=rng.start.line, character=token_start),
                end=Position(line=rng.start.line, character=token_end)
            ),
            new_text=replacement
        )
        
        # Create a workspace edit
        return WorkspaceEdit(changes={document.uri: [text_edit]})
    except Exception as e:
        logger.error(f"Error creating conversion edit: {e}")
        return None

def _get_unit_conversion_actions(ls: LanguageServer, params: CodeActionParams) -> List[CodeAction]:
    """
    Get code actions for unit conversion.
    
    Args:
        ls: The language server
        params: The code action parameters
        
    Returns:
        A list of code actions for unit conversion
    """
    uri = params.textDocument.uri
    document = ls.workspace.get_document(uri)
    if not document:
        return []
    
    actions = []
    
    # Check each diagnostic for unit conversion issues
    for diag in params.context.diagnostics:
        issue = _detect_unit_conversion_issues(diag)
        if not issue:
            continue
        
        # Create an edit for the conversion
        edit = _create_conversion_edit(document, issue)
        if not edit:
            continue
        
        # Create the code action
        converter = issue["converter"]
        from_unit = issue["from_unit"]
        to_unit = issue["to_unit"]
        
        action = CodeAction(
            title=f"Convert {from_unit} to {to_unit}",
            kind=CodeActionKind.QuickFix,
            edit=edit,
            diagnostics=[diag],
            is_preferred=True
        )
        
        actions.append(action)
    
    return actions

async def unit_conversion_code_action(ls: LanguageServer, params: CodeActionParams) -> List[CodeAction]:
    """
    Handle code action request for unit conversion.
    
    Args:
        ls: The language server
        params: The code action parameters
        
    Returns:
        A list of code actions for unit conversion
    """
    try:
        return _get_unit_conversion_actions(ls, params)
    except Exception as e:
        logger.error(f"Error getting unit conversion code actions: {e}")
        return []

def register(server: LanguageServer):
    """
    Register the unit conversion code action provider with the language server.
    
    Args:
        server: The language server to register with
    """
    # This provider hooks into the existing code action mechanism
    # It doesn't need a separate registration, as it will be called
    # by the code_action.py provider
    pass
