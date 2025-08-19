<!-- ScholarSphere Panel - Real document upload and library -->
<script lang="ts">
  import { createEventDispatcher, onMount } from 'svelte';
  import { conceptMesh, addConceptDiff } from '$lib/stores/conceptMesh';
  import { fade, fly } from 'svelte/transition';
  
  const dispatch = createEventDispatcher();
  
  let isDragOver = false;
  let isUploading = false;
  let uploadProgress = 0;
  let uploadedDocuments: any[] = [];
  let uploadStatus = '';
  let uploadError = '';
  let showDocuments = true;
  
  onMount(() => {
    // Load uploaded documents from localStorage
    const saved = localStorage.getItem('tori-scholarsphere-documents');
    if (saved) {
      try {
        uploadedDocuments = JSON.parse(saved);
      } catch (e) {
        console.warn('Failed to load ScholarSphere documents:', e);
      }
    }
  });
  
  // Auto-save documents list
  $: if (uploadedDocuments.length > 0) {
    localStorage.setItem('tori-scholarsphere-documents', JSON.stringify(uploadedDocuments));
  }
  
  function handleDragOver(event: DragEvent) {
    event.preventDefault();
    isDragOver = true;
  }
  
  function handleDragLeave(event: DragEvent) {
    event.preventDefault();
    isDragOver = false;
  }
  
  async function handleDrop(event: DragEvent) {
    event.preventDefault();
    isDragOver = false;
    
    const files = event.dataTransfer?.files;
    if (files && files.length > 0) {
      await processFiles(files);
    }
  }
  
  function handleFileSelect(event: Event) {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      processFiles(input.files);
    }
  }
  
  async function processFiles(files: FileList) {
    if (files.length === 0) return;
    
    isUploading = true;
    uploadProgress = 0;
    uploadError = '';
    uploadStatus = 'Preparing upload...';
    
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      
      try {
        uploadStatus = `Processing ${file.name}...`;
        uploadProgress = 20;
        
        // Create form data
        const formData = new FormData();
        formData.append('file', file);
        
        uploadProgress = 40;
        uploadStatus = 'Uploading to ScholarSphere...';
        
        // Call upload endpoint
        const response = await fetch('/upload', {
          method: 'POST',
          body: formData
        });
        
        uploadProgress = 70;
        
        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(errorData.message || `Upload failed: ${response.status}`);
        }
        
        uploadStatus = 'Processing document...';
        uploadProgress = 90;
        
        const result = await response.json();
        
        if (result.success && result.document) {
          // Add to uploaded documents list
          uploadedDocuments = [result.document, ...uploadedDocuments];
          
          // Add to concept mesh with ScholarSphere source
          addConceptDiff({
            type: 'document',
            title: result.document.filename,
            concepts: result.document.concepts,
            summary: result.document.summary,
            metadata: {
              source: 'scholarsphere',
              documentId: result.document.id,
              filename: result.document.filename,
              size: result.document.size,
              uploadedAt: result.document.uploadedAt,
              uploadedBy: result.document.uploadedBy,
              extractedText: result.document.extractedText,
              elfinTriggered: result.document.elfinTriggered,
              processingMethod: 'scholarsphere_ingest'
            }
          });
          
          uploadStatus = `✅ ${file.name} processed successfully`;
          uploadProgress = 100;
          
          console.log('📚 Document added to ScholarSphere:', result.document);
          
          // Dispatch completion event
          dispatch('upload-complete', { 
            document: result.document,
            conceptsAdded: result.document.concepts.length 
          });
          
        } else {
          throw new Error('Upload succeeded but document data missing');
        }
        
      } catch (error) {
        console.error(`❌ Failed to process ${file.name}:`, error);
        uploadError = error instanceof Error ? error.message : 'Upload failed';
        uploadStatus = `❌ Failed: ${file.name}`;
        break;
      }
    }
    
    // Reset upload state after delay
    setTimeout(() => {
      isUploading = false;
      uploadProgress = 0;
      uploadStatus = '';
      uploadError = '';
    }, 2000);
  }
  
  function browseFiles() {
    const input = document.createElement('input');
    input.type = 'file';
    input.multiple = true;
    input.accept = '.pdf,.txt,.json';
    input.onchange = handleFileSelect;
    input.click();
  }
  
  // A11y: Handle keyboard events for upload area
  function handleUploadKeydown(event: KeyboardEvent) {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      browseFiles();
    }
  }
  
  function formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }
  
  function formatDate(dateString: string): string {
    return new Date(dateString).toLocaleDateString() + ' ' + 
           new Date(dateString).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }
  
  function getDocumentIcon(filename: string): string {
    const ext = filename.split('.').pop()?.toLowerCase();
    switch (ext) {
      case 'pdf': return '📕';
      case 'txt': return '📄';
      case 'json': return '🔧';
      default: return '📄';
    }
  }
  
  function removeDocument(docId: string) {
    if (confirm('Remove this document from ScholarSphere?')) {
      uploadedDocuments = uploadedDocuments.filter(doc => doc.id !== docId);
      // Note: In production, this should also call a DELETE endpoint to remove the file
    }
  }
  
  function clearError() {
    uploadError = '';
  }
</script>

<div class="h-full flex flex-col bg-white">
  <!-- Header -->
  <div class="p-4 border-b border-gray-200 bg-gray-50">
    <div class="flex items-center justify-between">
      <div>
        <h3 class="font-semibold text-gray-900 flex items-center">
          📚 ScholarSphere
          <span class="ml-2 px-2 py-1 text-xs bg-purple-100 text-purple-700 rounded-full">Admin</span>
        </h3>
        <p class="text-xs text-gray-600 mt-1">
          {uploadedDocuments.length} documents • Revolutionary AI knowledge vault
        </p>
      </div>
      
      <button
        type="button"
        on:click={() => showDocuments = !showDocuments}
        class="p-1 hover:bg-gray-200 rounded transition-colors"
        title={showDocuments ? 'Hide documents' : 'Show documents'}
      >
        {showDocuments ? '🔽' : '▶️'}
      </button>
    </div>
  </div>
  
  <!-- Upload Area -->
  <div class="p-4 border-b border-gray-200">
    <!-- Error Display -->
    {#if uploadError}
      <div class="mb-3 p-3 bg-red-50 border border-red-200 rounded-lg" transition:fade>
        <div class="flex items-center justify-between">
          <div class="flex items-center">
            <span class="text-red-600 mr-2">❌</span>
            <span class="text-sm text-red-700">{uploadError}</span>
          </div>
          <button type="button" on:click={clearError} class="text-red-400 hover:text-red-600">✕</button>
        </div>
      </div>
    {/if}
    
    <!-- Upload Drop Zone -->
    <div 
      class="border-2 border-dashed rounded-lg p-4 text-center transition-all duration-200 {
        isDragOver ? 'border-purple-400 bg-purple-50' : 
        isUploading ? 'border-blue-400 bg-blue-50' : 
        'border-gray-300 hover:border-gray-400 cursor-pointer'
      }"
      class:pointer-events-none={isUploading}
      on:dragover={handleDragOver}
      on:dragleave={handleDragLeave}
      on:drop={handleDrop}
      on:click={browseFiles}
      on:keydown={handleUploadKeydown}
      role="button"
      tabindex="0"
      aria-label="Upload documents by clicking or dragging files here"
    >
      {#if isUploading}
        <div class="space-y-3" transition:fly={{y: 10, duration: 200}}>
          <div class="text-blue-600 text-2xl">📤</div>
          <div class="text-sm font-medium text-blue-700">{uploadStatus}</div>
          <div class="w-full bg-blue-200 rounded-full h-2 max-w-xs mx-auto">
            <div 
              class="bg-blue-600 h-2 rounded-full transition-all duration-300"
              style="width: {uploadProgress}%"
            ></div>
          </div>
          <div class="text-xs text-blue-600">{uploadProgress}%</div>
        </div>
      {:else}
        <div class="space-y-2">
          <div class="text-2xl text-gray-400">📚</div>
          <div class="text-sm font-medium text-gray-700">
            Drop PDFs here or <span class="text-purple-600">browse to upload</span>
          </div>
          <div class="text-xs text-gray-500">
            Documents will be processed by TORI's cognitive engine
          </div>
        </div>
      {/if}
    </div>
    
    <!-- File type support -->
    <div class="text-xs text-gray-500 text-center mt-2">
      Supports: PDF, TXT, JSON • Max 50MB per file
    </div>
  </div>
  
  <!-- Document Library -->
  {#if showDocuments}
    <div class="flex-1 overflow-y-auto" transition:fly={{y: -20, duration: 200}}>
      {#if uploadedDocuments.length === 0}
        <div class="p-6 text-center text-gray-500">
          <div class="text-3xl mb-2">📖</div>
          <div class="text-sm">No documents uploaded yet</div>
          <div class="text-xs mt-1">Upload your first document to begin</div>
        </div>
      {:else}
        <div class="p-2 space-y-2">
          {#each uploadedDocuments as doc (doc.id)}
            <div class="border border-gray-200 rounded-lg p-3 hover:border-gray-300 hover:shadow-sm transition-all"
                 transition:fly={{x: -20, duration: 200}}>
              <!-- Document header -->
              <div class="flex items-start justify-between mb-2">
                <div class="flex items-center space-x-2 flex-1 min-w-0">
                  <div class="text-lg flex-shrink-0">
                    {getDocumentIcon(doc.filename)}
                  </div>
                  <div class="min-w-0 flex-1">
                    <h4 class="font-medium text-gray-900 truncate text-sm" title={doc.filename}>
                      {doc.filename}
                    </h4>
                    <p class="text-xs text-gray-500">
                      {formatDate(doc.uploadedAt)} • {formatFileSize(doc.size)}
                    </p>
                  </div>
                </div>
                
                <!-- Actions -->
                <button
                  type="button"
                  on:click={() => removeDocument(doc.id)}
                  class="text-gray-400 hover:text-red-500 text-xs p-1"
                  title="Remove document"
                >
                  🗑️
                </button>
              </div>
              
              <!-- Concepts -->
              {#if doc.concepts && doc.concepts.length > 0}
                <div class="flex flex-wrap gap-1 mb-2">
                  {#each doc.concepts.slice(0, 3) as concept}
                    <span class="px-2 py-1 text-xs bg-purple-100 text-purple-700 rounded-full">
                      {concept}
                    </span>
                  {/each}
                  {#if doc.concepts.length > 3}
                    <span class="px-2 py-1 text-xs bg-gray-100 text-gray-600 rounded-full">
                      +{doc.concepts.length - 3}
                    </span>
                  {/if}
                </div>
              {/if}
              
              <!-- Status indicators -->
              <div class="flex items-center justify-between text-xs">
                <div class="text-gray-500">
                  {doc.concepts?.length || 0} concepts extracted
                </div>
                
                <div class="flex items-center space-x-2">
                  {#if doc.elfinTriggered}
                    <span class="text-green-600" title="ELFIN++ processed">🧬</span>
                  {/if}
                  <span class="text-purple-600" title="In concept mesh">🧠</span>
                </div>
              </div>
            </div>
          {/each}
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  [role="button"] {
    outline: none;
  }
  
  [role="button"]:focus {
    box-shadow: 0 0 0 2px rgba(147, 51, 234, 0.5);
  }
</style>
