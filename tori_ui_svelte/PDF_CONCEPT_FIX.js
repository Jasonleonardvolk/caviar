// Simple patch file to fix PDF concept extraction
// Apply this to line 370 in +page.svelte

// REPLACE THIS LINE:
currentConcepts: [...new Set($conceptMesh.flatMap(d => d.concepts))],

// WITH THIS BLOCK:
currentConcepts: (() => {
  const allConcepts = [];
  let pdfCount = 0;
  $conceptMesh.forEach(diff => {
    if (diff.concepts && Array.isArray(diff.concepts)) {
      if (diff.metadata?.source === 'scholarsphere_enhanced_server') {
        pdfCount += diff.concepts.length;
      }
      allConcepts.push(...diff.concepts);
    }
  });
  const uniqueConcepts = [...new Set(allConcepts)];
  console.log('📚 CONCEPT MESH STATUS:', {
    totalConcepts: allConcepts.length,
    uniqueConcepts: uniqueConcepts.length,
    pdfConcepts: pdfCount
  });
  return uniqueConcepts;
})(),
