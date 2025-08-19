<!-- UserSettings.svelte - Privacy-focused settings panel -->
<script lang="ts">
  import { userSession, updateUserPreferences, updateUserMemory } from '$lib/stores/user';
  import { conceptMesh, clearConceptMesh } from '$lib/stores/conceptMesh';
  
  let showSettings = false;
  let activeTab = 'profile'; // 'profile', 'privacy', 'data'
  
  // Privacy settings
  let personalizedGreetings = $userSession.user?.preferences.personalizedGreetings ?? true;
  let showRecentActivity = $userSession.user?.preferences.showRecentActivity ?? true;
  let privacyLevel = $userSession.user?.preferences.privacyLevel ?? 'balanced';
  
  function toggleSettings() {
    showSettings = !showSettings;
  }
  
  function updatePrivacySettings() {
    if ($userSession.user) {
      updateUserPreferences({
        personalizedGreetings,
        showRecentActivity,
        privacyLevel
      });
    }
  }
  
  function exportData() {
    const data = {
      user: $userSession.user,
      conceptMesh: $conceptMesh,
      exportDate: new Date().toISOString()
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `tori-data-export-${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }
  
  function clearAllData() {
    if (confirm('This will delete all your data including conversations, documents, and settings. This cannot be undone. Continue?')) {
      clearConceptMesh();
      localStorage.clear();
      location.reload();
    }
  }
</script>

<!-- Settings trigger - subtle gear icon -->
{#if $userSession.isAuthenticated}
  <button 
    class="text-gray-400 hover:text-gray-600 transition-colors p-2 rounded-lg hover:bg-gray-100"
    on:click={toggleSettings}
    title="Settings"
  >
    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
            d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
            d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
    </svg>
  </button>
{/if}

<!-- Settings Modal -->
{#if showSettings}
  <div class="fixed inset-0 bg-black/20 backdrop-blur-sm flex items-center justify-center z-50 p-4">
    <div class="bg-white rounded-xl shadow-2xl max-w-2xl w-full max-h-[80vh] overflow-hidden">
      <!-- Header -->
      <div class="border-b border-gray-200 p-6 flex items-center justify-between">
        <h2 class="text-xl font-semibold text-gray-800">Settings</h2>
        <button 
          class="text-gray-400 hover:text-gray-600 transition-colors"
          on:click={() => showSettings = false}
        >
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
      
      <!-- Content -->
      <div class="flex h-[500px]">
        <!-- Sidebar -->
        <div class="w-48 border-r border-gray-200 p-4 space-y-1">
          <button 
            class="w-full text-left px-3 py-2 rounded-lg transition-colors {activeTab === 'profile' ? 'bg-blue-50 text-blue-700' : 'text-gray-700 hover:bg-gray-100'}"
            on:click={() => activeTab = 'profile'}
          >
            👤 Profile
          </button>
          <button 
            class="w-full text-left px-3 py-2 rounded-lg transition-colors {activeTab === 'privacy' ? 'bg-blue-50 text-blue-700' : 'text-gray-700 hover:bg-gray-100'}"
            on:click={() => activeTab = 'privacy'}
          >
            🔒 Privacy
          </button>
          <button 
            class="w-full text-left px-3 py-2 rounded-lg transition-colors {activeTab === 'data' ? 'bg-blue-50 text-blue-700' : 'text-gray-700 hover:bg-gray-100'}"
            on:click={() => activeTab = 'data'}
          >
            💾 Data
          </button>
        </div>
        
        <!-- Tab content -->
        <div class="flex-1 p-6 overflow-y-auto">
          {#if activeTab === 'profile'}
            <!-- Profile settings -->
            <div class="space-y-6">
              <div>
                <h3 class="text-lg font-medium text-gray-800 mb-4">Profile Information</h3>
                
                <div class="space-y-4">
                  <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">Name</label>
                    <input 
                      type="text"
                      value={$userSession.user?.name}
                      disabled
                      class="w-full px-3 py-2 border border-gray-300 rounded-lg bg-gray-50"
                    />
                  </div>
                  
                  <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">Email</label>
                    <input 
                      type="email"
                      value={$userSession.user?.email}
                      disabled
                      class="w-full px-3 py-2 border border-gray-300 rounded-lg bg-gray-50"
                    />
                  </div>
                </div>
              </div>
              
              <div>
                <h3 class="text-lg font-medium text-gray-800 mb-4">Statistics</h3>
                <div class="grid grid-cols-2 gap-4">
                  <div class="bg-gray-50 rounded-lg p-4">
                    <div class="text-2xl font-semibold text-gray-800">
                      {$userSession.user?.stats.conversationsCount || 0}
                    </div>
                    <div class="text-sm text-gray-600">Conversations</div>
                  </div>
                  <div class="bg-gray-50 rounded-lg p-4">
                    <div class="text-2xl font-semibold text-gray-800">
                      {$userSession.user?.stats.documentsUploaded || 0}
                    </div>
                    <div class="text-sm text-gray-600">Documents</div>
                  </div>
                  <div class="bg-gray-50 rounded-lg p-4">
                    <div class="text-2xl font-semibold text-gray-800">
                      {$userSession.user?.stats.conceptsCreated || 0}
                    </div>
                    <div class="text-sm text-gray-600">Concepts</div>
                  </div>
                  <div class="bg-gray-50 rounded-lg p-4">
                    <div class="text-2xl font-semibold text-gray-800">
                      {$conceptMesh.length}
                    </div>
                    <div class="text-sm text-gray-600">Memory Entries</div>
                  </div>
                </div>
              </div>
            </div>
            
          {:else if activeTab === 'privacy'}
            <!-- Privacy settings -->
            <div class="space-y-6">
              <div>
                <h3 class="text-lg font-medium text-gray-800 mb-4">Privacy Settings</h3>
                <p class="text-sm text-gray-600 mb-6">
                  Control how TORI uses your information. All data is stored locally on your device.
                </p>
                
                <div class="space-y-4">
                  <label class="flex items-center justify-between p-4 bg-gray-50 rounded-lg cursor-pointer hover:bg-gray-100 transition-colors">
                    <div>
                      <div class="font-medium text-gray-800">Personalized Greetings</div>
                      <div class="text-sm text-gray-600">Show your name in welcome messages</div>
                    </div>
                    <input 
                      type="checkbox"
                      bind:checked={personalizedGreetings}
                      on:change={updatePrivacySettings}
                      class="w-4 h-4 text-blue-600 rounded"
                    />
                  </label>
                  
                  <label class="flex items-center justify-between p-4 bg-gray-50 rounded-lg cursor-pointer hover:bg-gray-100 transition-colors">
                    <div>
                      <div class="font-medium text-gray-800">Show Recent Activity</div>
                      <div class="text-sm text-gray-600">Display your recent documents and conversations</div>
                    </div>
                    <input 
                      type="checkbox"
                      bind:checked={showRecentActivity}
                      on:change={updatePrivacySettings}
                      class="w-4 h-4 text-blue-600 rounded"
                    />
                  </label>
                  
                  <div class="p-4 bg-gray-50 rounded-lg">
                    <div class="font-medium text-gray-800 mb-3">Privacy Level</div>
                    <div class="space-y-2">
                      <label class="flex items-center">
                        <input 
                          type="radio"
                          bind:group={privacyLevel}
                          value="minimal"
                          on:change={updatePrivacySettings}
                          class="mr-2"
                        />
                        <span class="text-sm">
                          <span class="font-medium">Minimal</span> - Basic functionality only
                        </span>
                      </label>
                      <label class="flex items-center">
                        <input 
                          type="radio"
                          bind:group={privacyLevel}
                          value="balanced"
                          on:change={updatePrivacySettings}
                          class="mr-2"
                        />
                        <span class="text-sm">
                          <span class="font-medium">Balanced</span> - Recommended settings
                        </span>
                      </label>
                      <label class="flex items-center">
                        <input 
                          type="radio"
                          bind:group={privacyLevel}
                          value="full"
                          on:change={updatePrivacySettings}
                          class="mr-2"
                        />
                        <span class="text-sm">
                          <span class="font-medium">Full</span> - All features enabled
                        </span>
                      </label>
                    </div>
                  </div>
                </div>
              </div>
              
              <div class="p-4 bg-blue-50 rounded-lg">
                <div class="flex items-start space-x-3">
                  <div class="text-2xl">🔒</div>
                  <div>
                    <h4 class="font-medium text-blue-900">Your Privacy Matters</h4>
                    <p class="text-sm text-blue-800 mt-1">
                      All your data is stored locally on your device. TORI never sends your personal information, 
                      conversations, or documents to any external servers. You have complete control over your data.
                    </p>
                  </div>
                </div>
              </div>
            </div>
            
          {:else if activeTab === 'data'}
            <!-- Data management -->
            <div class="space-y-6">
              <div>
                <h3 class="text-lg font-medium text-gray-800 mb-4">Data Management</h3>
                <p class="text-sm text-gray-600 mb-6">
                  Export your data or clear your local storage. All actions are performed locally.
                </p>
                
                <div class="space-y-4">
                  <div class="p-4 border border-gray-200 rounded-lg">
                    <h4 class="font-medium text-gray-800 mb-2">Export Your Data</h4>
                    <p class="text-sm text-gray-600 mb-3">
                      Download all your conversations, documents, and settings as a JSON file.
                    </p>
                    <button 
                      class="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors text-sm font-medium"
                      on:click={exportData}
                    >
                      Export Data
                    </button>
                  </div>
                  
                  <div class="p-4 border border-red-200 bg-red-50 rounded-lg">
                    <h4 class="font-medium text-red-800 mb-2">Clear All Data</h4>
                    <p class="text-sm text-red-700 mb-3">
                      Permanently delete all your data from this device. This action cannot be undone.
                    </p>
                    <button 
                      class="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors text-sm font-medium"
                      on:click={clearAllData}
                    >
                      Clear All Data
                    </button>
                  </div>
                </div>
              </div>
              
              <div>
                <h3 class="text-lg font-medium text-gray-800 mb-4">Storage Usage</h3>
                <div class="space-y-2">
                  <div class="flex items-center justify-between text-sm">
                    <span class="text-gray-600">Conversations</span>
                    <span class="font-medium">{(JSON.stringify(localStorage.getItem('tori-conversation-history') || '').length / 1024).toFixed(1)} KB</span>
                  </div>
                  <div class="flex items-center justify-between text-sm">
                    <span class="text-gray-600">Concept Mesh</span>
                    <span class="font-medium">{(JSON.stringify($conceptMesh).length / 1024).toFixed(1)} KB</span>
                  </div>
                  <div class="flex items-center justify-between text-sm">
                    <span class="text-gray-600">User Data</span>
                    <span class="font-medium">{(JSON.stringify($userSession).length / 1024).toFixed(1)} KB</span>
                  </div>
                </div>
              </div>
            </div>
          {/if}
        </div>
      </div>
    </div>
  </div>
{/if}
