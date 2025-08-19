# ELFIN Troubleshooting & Debugging Guide

## Common Circular Reference Issues

Circular references are one of the most common issues in ELFIN files. They occur when a variable directly or indirectly depends on itself, creating an algebraic loop that cannot be resolved at runtime.

### Direct Circular References

Direct circular references happen when a variable directly references itself in its own definition:

```
v_dot = v_dot + a / m;  // ERROR: v_dot depends on itself
```

These are usually straightforward to spot and fix.

### Indirect Circular References

Indirect circular references form a chain of dependencies that loop back on themselves:

```
a = b;
b = c;
c = a;  // ERROR: Creates a cycle a -> b -> c -> a
```

These can be harder to detect, especially in large files with many variables.

## Best Practices for Avoiding Circular References

### 1. Use Clear Dependency Ordering

Structure your equations to have a clear forward direction:

```
// Good: Clear dependency chain
a = f(x, y);
b = g(a);
c = h(b);
```

### 2. Break Circular Dependencies with State

Use explicit state variables rather than direct feedback:

```
// Bad
x_dot = f(x_dot, x);

// Good
x_dot = f(x_dot_prev, x);  // Use previous timestep value
```

### 3. Use Helper Functions

Extract common calculations into helper functions to clarify dependencies:

```
helpers MyHelpers {
    calculate_position(theta, r) = r * cos(theta);
}

system Robot {
    flow_dynamics {
        x = calculate_position(theta, r);  // Clearer dependencies
    }
}
```

### 4. Regular Static Analysis

Run the circular reference checker regularly during development:

```bash
python check_circular_refs.py path/to/file.elfin
```

## Debugging Approaches

### Direct String Parsing vs. Complex AST

When we implemented the circular reference analyzer, we found two approaches:

1. **Complex AST-based Parsing**: Initially tried with `reference_analyzer.py`, built a complex parser that extracts a full AST and analyzes it.
   - Pro: Can potentially provide more detailed analysis
   - Con: More prone to parsing errors and edge cases

2. **Direct String Parsing**: Implemented in `circular_analyzer.py`, uses simple line-by-line parsing to extract assignments.
   - Pro: More robust, handles various syntax variations better
   - Con: Less detailed analysis capabilities

The direct string parsing approach proved more effective for circular reference detection because:
- It's more robust against syntax variations
- It focuses on just the information needed (assignments)
- It handles comments and whitespace naturally
- It's simpler to understand and maintain

### Effective Debugging Techniques

1. **Visualize the Dependency Graph**: When debugging complex relationships, sketch the dependency graph to visualize cycles.

2. **Simplify**: Create minimal test cases that isolate the problematic pattern.

3. **Incremental Building**: Start with the simplest possible analyzer that works, then enhance:
   ```python
   # Start with just detecting assignments
   for line in lines:
       if '=' in line and ';' in line:
           # Extract variable and expression
           ...
   ```

4. **Use Multiple Test Cases**: Develop test cases for each pattern you want to detect.

## Common Issues and Solutions

### Problem: False Positive Circular References

**Example**: Function calls mistaken for variables
```
position(t) = position(t-1) + velocity * dt;  // Not actually circular
```

**Solution**: Skip function definitions by checking for parentheses in variable names
```python
# Skip function definitions
if '(' in var_name:
    continue
```

### Problem: Complex Expressions Confuse the Parser

**Example**: Conditional expressions or multi-line statements
```
x = if (condition) then
       expr1
     else
       expr2;
```

**Solution**: Use line-by-line parsing with specific pattern checks rather than trying to parse the full grammar

### Problem: Dependencies Not Found

**Example**: References to undefined variables or functions
```
x_dot = f(y);  // But f or y may not be defined
```

**Solution**: Build a dependency graph and validate all references

## Summary: Effective Analyzer Design

1. **Focus on Specific Tasks**: Build analyzers that do one thing well
2. **Start Simple**: Begin with direct string manipulation for robustness
3. **Test Thoroughly**: Create tests for all edge cases
4. **Prioritize Robustness**: Prefer simpler approaches that work reliably
5. **Error Clearly**: Provide specific, actionable error messages
