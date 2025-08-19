// Quick test to check what's happening with soliton memory
// Run this in browser console

console.log("=== TORI Frontend Diagnostic ===");

// Check if the page loaded
console.log("Page loaded:", !!window);

// Check for ELFIN
console.log("ELFIN available:", typeof window.ELFIN !== 'undefined');

// Try to call soliton init directly
async function testSolitonDirect() {
    console.log("\n--- Testing Soliton Init Directly ---");
    
    try {
        const response = await fetch('/api/soliton/init', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user: 'console_test_user' })
        });
        
        console.log("Response status:", response.status);
        const data = await response.json();
        console.log("Response data:", data);
        
        if (response.ok) {
            console.log("✅ Soliton init worked!");
        } else {
            console.log("❌ Soliton init failed");
        }
    } catch (error) {
        console.error("❌ Network error:", error);
    }
}

// Try to test the answer endpoint
async function testAnswerDirect() {
    console.log("\n--- Testing Answer Endpoint Directly ---");
    
    try {
        const response = await fetch('/api/answer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                user_query: 'Hello from console test',
                persona: {}
            })
        });
        
        console.log("Response status:", response.status);
        
        if (response.ok) {
            const data = await response.json();
            console.log("✅ Answer received:", data.answer?.substring(0, 100) + "...");
        } else {
            const error = await response.text();
            console.log("❌ Answer failed:", error);
        }
    } catch (error) {
        console.error("❌ Network error:", error);
    }
}

// Check what's in the stores
console.log("\n--- Checking Svelte Stores ---");
console.log("Checking window.__svelte__:", !!window.__svelte__);

// Run the tests
console.log("\n🧪 Running tests...\n");
testSolitonDirect().then(() => testAnswerDirect());

// Also check localStorage
console.log("\n--- Local Storage ---");
console.log("Conversation history:", localStorage.getItem('tori-conversation-history') ? 'Present' : 'Empty');
console.log("Welcome seen:", localStorage.getItem('tori-welcome-seen'));

// Network check
console.log("\n--- Quick Network Check ---");
fetch('http://localhost:8002/api/health')
    .then(r => r.json())
    .then(d => console.log("✅ Direct backend call works:", d))
    .catch(e => console.log("❌ Direct backend call failed:", e));

fetch('/api/health')
    .then(r => r.json())
    .then(d => console.log("✅ Proxied call works:", d))
    .catch(e => console.log("❌ Proxied call failed:", e));
