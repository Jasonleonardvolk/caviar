// Test the API endpoints directly
console.log("Testing Soliton API endpoints...");

// Test 1: Health check
fetch('/api/soliton/health')
  .then(r => r.json())
  .then(d => console.log("✅ Health:", d))
  .catch(e => console.error("❌ Health failed:", e));

// Test 2: Initialize
fetch('/api/soliton/init', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({user: 'Admin User'})
})
  .then(r => r.json())
  .then(d => console.log("✅ Init:", d))
  .catch(e => console.error("❌ Init failed:", e));

// Test 3: Check if chat API works
fetch('/api/answer', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({query: 'Hello'})
})
  .then(r => r.json())
  .then(d => console.log("✅ Chat API:", d))
  .catch(e => console.error("❌ Chat failed:", e));
