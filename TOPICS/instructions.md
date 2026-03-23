# Instructions for Creating Topic Equations

This document provides guidelines for creating physics equations that can be used to generate diverse word problems.

## Basic Formatting Rules

- **Spacing**: Each element in an equation must be separated by a space.
- **Operators**: Use standard mathematical operators: `*` for multiplication, `/` for division, `^` for exponentiation, `+` for addition, `-` for subtraction.
- **Parentheses**: Use parentheses to group terms and ensure correct order of operations.

### Example
```
"F = ( G * m1 * m2 ) / ( r ^ 2 )"
```

## Expanding Formulas for Problem Variations

To maximize the diversity of generated problems, expand formulas to include additional variables that might appear in different problem scenarios, even if they cancel out in the final equation.

### Example
```
"F = ( G * m1 * m2 ) / ( ( r - ( r1 + r2 ) + ( r1 + r2 ) ) ^ 2 )"
```

In this expanded form:
- `r` is the distance from center to center between two bodies
- `r1` and `r2` are the radii of the two bodies

Although `r1` and `r2` cancel out in the simplified equation, including them allows the problem generator to create scenarios where these values are relevant. For instance, this enables problems involving gravitational potential when two bodies are about to collide, providing additional context and complexity.

## Best Practices

- Include variables that could be known or unknown in different problem types
- Use expanded forms to support edge cases and specific physics scenarios
- Ensure equations are mathematically correct and follow standard notation
- Test equations with the graph chain system to verify they generate solvable problems