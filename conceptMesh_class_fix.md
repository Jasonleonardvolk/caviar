# Fix for conceptMesh.ts

## Issue: Methods are outside the class

The `removeConceptDiff` and `clearConceptMesh` methods are currently placed OUTSIDE the ConceptMeshStore class (after the closing brace on line 769).

They need to be INSIDE the class, before the closing brace.

## Quick Fix:

1. Find line 769 where the ConceptMeshStore class ends with `}`
2. Move both methods (`removeConceptDiff` and `clearConceptMesh`) to BEFORE that closing brace
3. Make sure they're at the same indentation level as other methods like `getStatistics()`

## Correct placement:

```typescript
  // ... other methods ...

  private emit(event: string, data: any) {
    // ... existing code ...
  }

  /**
   * Remove a previously recorded ConceptDiff by its id and broadcast an event.
   */
  public removeConceptDiff(diffId: string): void {
    // ... method body ...
  }

  /**
   * Hard‑reset the in‑memory concept mesh, diffs, and events.
   */
  public clearConceptMesh(): void {
    // ... method body ...
  }

  // Store access methods
  public get conceptsStore(): Readable<Concept[]> {
    return { subscribe: this.concepts.subscribe };
  }
  
  // ... other getter methods ...
  
} // <-- This is the closing brace of ConceptMeshStore class
```

The exports at the bottom are correct and should work once the methods are inside the class.
