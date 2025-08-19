<!-- Clean UploadPanel with accessibility fixes -->
<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { conceptMesh, addConceptDiff } from '$lib/stores/conceptMesh';
  import { userSession, updateUserStats } from '$lib/stores/user';
  
  const dispatch = createEventDispatcher();
  
  let isDragOver = false;
  let isUploading = false;
  let uploadProgress = 0;
  
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
    if (!$userSession.isAuthenticated) {
      alert('Please sign in to upload documents');
      return;
    }
    
    isUploading = true;
    uploadProgress = 0;
    
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      
      try {
        // Simulate upload progress
        for (let progress = 0; progress <= 100; progress += 20) {
          uploadProgress = progress;
          await new Promise(resolve => setTimeout(resolve, 100));
        }
        
        // Extract concepts from filename and file type
        const concepts = extractConceptsFromFile(file);
        
        // Add to concept mesh
        addConceptDiff({
          type: 'document',
          title: file.name,
          concepts: concepts,
          summary: `${file.type || 'Unknown type'} document (${formatFileSize(file.size)})`,
          metadata: {
            fileName: file.name,
            fileSize: file.size,
            fileType: file.type,
            uploadDate: new Date(),
            processingMethod: 'file_upload'
          }
        });
        
        // Update user stats
        updateUserStats({ documentsUploaded: 1 });
        
        console.log(`✅ Document processed: ${file.name}`);
        
      } catch (error) {
        console.error(`❌ Failed to process ${file.name}:`, error);
      }
    }
    
    isUploading = false;
    uploadProgress = 0;
    
    // Dispatch completion event
    dispatch('upload-complete', { fileCount: files.length });
  }
  
  function extractConceptsFromFile(file: File): string[] {
    const concepts: string[] = [];
    const fileName = file.name.toLowerCase();
    const fileType = file.type.toLowerCase();
    
    // Concepts based on file type
    if (fileType.includes('pdf')) concepts.push('Document', 'PDF');
    if (fileType.includes('image')) concepts.push('Image', 'Visual');
    if (fileType.includes('text')) concepts.push('Text', 'Notes');
    if (fileType.includes('spreadsheet') || fileName.includes('.csv') || fileName.includes('.xlsx')) {
      concepts.push('Data', 'Spreadsheet');
    }
    if (fileType.includes('presentation')) concepts.push('Presentation', 'Slides');
    
    // Concepts based on filename keywords
    if (fileName.includes('report')) concepts.push('Report');
    if (fileName.includes('research')) concepts.push('Research');
    if (fileName.includes('analysis')) concepts.push('Analysis');
    if (fileName.includes('design')) concepts.push('Design');
    if (fileName.includes('plan')) concepts.push('Planning');
    if (fileName.includes('meeting')) concepts.push('Meeting');
    if (fileName.includes('project')) concepts.push('Project');
    if (fileName.includes('budget')) concepts.push('Finance');
    if (fileName.includes('strategy')) concepts.push('Strategy');
    
    // Academic/research concepts
    if (fileName.includes('paper') || fileName.includes('thesis')) concepts.push('Academic');
    if (fileName.includes('study')) concepts.push('Study');
    if (fileName.includes('journal')) concepts.push('Journal');
    
    return concepts.length > 0 ? concepts : ['Document'];
  }
  
  function formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }
  
  function browseFiles() {
    const input = document.createElement('input');
    input.type = 'file';
    input.multiple = true;
    input.accept = '.pdf,.doc,.docx,.txt,.md,.csv,.xlsx,.ppt,.pptx,.jpg,.jpeg,.png,.gif';
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
</script>

<div class="space-y-3">
  <!-- Upload area -->
  <div 
    class="border-2 border-dashed rounded-lg p-6 text-center transition-colors {
      isDragOver ? 'border-blue-400 bg-blue-50' : 'border-gray-300 hover:border-gray-400'
    } {isUploading ? 'pointer-events-none opacity-50' : 'cursor-pointer'}"
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
      <div class="space-y-2">
        <div class="text-blue-600">📤</div>
        <div class="text-sm text-blue-700">Processing documents...</div>
        <div class="w-full bg-blue-200 rounded-full h-2">
          <div 
            class="bg-blue-600 h-2 rounded-full transition-all duration-300"
            style="width: {uploadProgress}%"
          ></div>
        </div>
        <div class="text-xs text-blue-600">{uploadProgress}%</div>
      </div>
    {:else}
      <div class="space-y-2">
        <div class="text-3xl text-gray-400">📄</div>
        <div class="text-sm font-medium text-gray-700">
          Drop files here or <span class="text-blue-600">browse to upload</span>
        </div>
        <div class="text-xs text-gray-500">
          Documents will be processed by Scholar ghost via revolutionary AI
        </div>
      </div>
    {/if}
  </div>
  
  <!-- Supported formats -->
  <div class="text-xs text-gray-500 text-center">
    Supports: PDF, DOC, TXT, MD, CSV, XLSX, PPT, Images
  </div>
  
  <!-- Quick stats -->
  {#if $userSession.user}
    <div class="text-xs text-gray-600 text-center">
      {$userSession.user.stats.documentsUploaded} documents uploaded • Ready for revolutionary processing
    </div>
  {/if}
</div>

<style>
  [role="button"] {
    outline: none;
  }
  
  [role="button"]:focus {
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.5);
  }
</style>
