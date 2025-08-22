import fs from 'fs';
import path from 'path';

// Read the file
const filePath = path.join(process.cwd(), 'src/routes/+page.svelte');
let content = fs.readFileSync(filePath, 'utf-8');

// Find and replace the specific line
const oldLine = 'currentConcepts: [...new Set($conceptMesh.flatMap(d => d.concepts))],';
const newLine = `currentConcepts: (() => {
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
          console.log('📚 PDF CONCEPTS AVAILABLE:', pdfCount, 'of', uniqueConcepts.length, 'total');
          return uniqueConcepts;
        })(),`;

// Replace
content = content.replace(oldLine, newLine);

// Write back
fs.writeFileSync(filePath, content, 'utf-8');
console.log('✅ PDF concept extraction fix applied!');
