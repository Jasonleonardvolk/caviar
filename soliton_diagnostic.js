// Quick test to see what's happening with Soliton Memory
// Paste this in your browser console (F12)

console.log("=== SOLITON MEMORY DIAGNOSTIC ===");

// Test 1: Check if the endpoint is accessible
fetch('/api/soliton/init', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user: 'Admin User' })
})
.then(r => {
    console.log("Init response status:", r.status);
    return r.json();
})
.then(data => {
    console.log("Init response data:", data);
})
.catch(err => {
    console.error("Init error:", err);
});

// Test 2: Check health endpoint
fetch('/api/soliton/health')
.then(r => r.json())
.then(data => console.log("Health check:", data))
.catch(err => console.error("Health error:", err));

// Test 3: Check if user ID is set
console.log("Current user data:", localStorage.getItem('tori_user'));

// Test 4: Try to get the soliton memory state
const memoryKey = 'tori_soliton_memory_Admin User';
console.log("Memory state:", localStorage.getItem(memoryKey));

// Test 5: Check if the backend is responding
fetch('/api/health')
.then(r => r.json())
.then(data => console.log("Backend health:", data))
.catch(err => console.error("Backend error:", err));
