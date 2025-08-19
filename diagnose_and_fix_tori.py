#!/usr/bin/env python3
"""
TORI Comprehensive Diagnostic and Fix Script
============================================
This script will systematically diagnose and fix the TORI homepage issues.
"""

import os
import sys
import time
import json
import requests
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

class TORIDiagnostic:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.frontend_dir = self.base_dir / "tori_ui_svelte"
        self.issues_found = []
        self.fixes_applied = []
        
    def log(self, message, level="INFO"):
        """Log with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = {
            "INFO": "📋",
            "SUCCESS": "✅",
            "WARNING": "⚠️",
            "ERROR": "❌",
            "FIX": "🔧"
        }.get(level, "📌")
        print(f"[{timestamp}] {prefix} {message}")
        
    def check_backend_health(self):
        """Check if backend is responding"""
        self.log("Checking backend health...")
        
        try:
            # Check main health endpoint
            response = requests.get("http://localhost:8002/api/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.log(f"Backend health: {data.get('status', 'unknown')}", "SUCCESS")
                self.log(f"Prajna loaded: {data.get('prajna_loaded', False)}")
                return True
            else:
                self.log(f"Backend returned status {response.status_code}", "ERROR")
                self.issues_found.append("Backend unhealthy")
                return False
        except requests.exceptions.ConnectionError:
            self.log("Backend not responding - is it running?", "ERROR")
            self.issues_found.append("Backend not running")
            return False
        except Exception as e:
            self.log(f"Backend check failed: {e}", "ERROR")
            self.issues_found.append(f"Backend error: {e}")
            return False
            
    def test_soliton_endpoints(self):
        """Test soliton memory endpoints"""
        self.log("Testing soliton memory endpoints...")
        
        endpoints = [
            ("POST", "/api/soliton/init", {"user": "test_user"}),
            ("GET", "/api/soliton/health", None),
            ("GET", "/api/soliton/stats/test_user", None)
        ]
        
        working_endpoints = 0
        
        for method, endpoint, data in endpoints:
            try:
                url = f"http://localhost:8002{endpoint}"
                if method == "POST":
                    response = requests.post(url, json=data, timeout=5)
                else:
                    response = requests.get(url, timeout=5)
                    
                if response.status_code in [200, 201]:
                    self.log(f"{endpoint} - OK", "SUCCESS")
                    working_endpoints += 1
                else:
                    self.log(f"{endpoint} - Failed ({response.status_code})", "WARNING")
                    self.issues_found.append(f"Endpoint {endpoint} returned {response.status_code}")
            except Exception as e:
                self.log(f"{endpoint} - Error: {e}", "ERROR")
                self.issues_found.append(f"Endpoint {endpoint} error: {e}")
                
        return working_endpoints == len(endpoints)
        
    def test_prajna_endpoint(self):
        """Test the main chat endpoint"""
        self.log("Testing Prajna chat endpoint...")
        
        try:
            response = requests.post(
                "http://localhost:8002/api/answer",
                json={"user_query": "Hello TORI", "persona": {}},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                self.log(f"Prajna response received: {len(data.get('answer', ''))} chars", "SUCCESS")
                return True
            else:
                self.log(f"Prajna returned status {response.status_code}", "ERROR")
                self.issues_found.append(f"Prajna endpoint failed: {response.status_code}")
                return False
        except Exception as e:
            self.log(f"Prajna test failed: {e}", "ERROR")
            self.issues_found.append(f"Prajna error: {e}")
            return False
            
    def check_frontend_cache(self):
        """Check for Svelte/Vite cache issues"""
        self.log("Checking frontend cache...")
        
        cache_dirs = [
            self.frontend_dir / ".svelte-kit",
            self.frontend_dir / "node_modules" / ".vite",
            self.frontend_dir / ".vite"
        ]
        
        caches_found = []
        
        for cache_dir in cache_dirs:
            if cache_dir.exists():
                self.log(f"Found cache: {cache_dir.name}", "WARNING")
                caches_found.append(cache_dir)
                self.issues_found.append(f"Cache found: {cache_dir.name}")
                
        return caches_found
        
    def clear_frontend_cache(self, force=False):
        """Clear Svelte/Vite caches"""
        if not force:
            response = input("\n🔧 Clear frontend caches? This will force a rebuild. (y/n): ")
            if response.lower() != 'y':
                self.log("Skipping cache clear", "INFO")
                return False
                
        self.log("Clearing frontend caches...", "FIX")
        
        # Stop any running dev server first
        self.log("Note: Make sure to stop the dev server (Ctrl+C) before clearing cache", "WARNING")
        input("Press Enter when dev server is stopped...")
        
        cache_dirs = [
            self.frontend_dir / ".svelte-kit",
            self.frontend_dir / "node_modules" / ".vite",
            self.frontend_dir / ".vite"
        ]
        
        for cache_dir in cache_dirs:
            if cache_dir.exists():
                try:
                    shutil.rmtree(cache_dir)
                    self.log(f"Cleared {cache_dir.name}", "SUCCESS")
                    self.fixes_applied.append(f"Cleared {cache_dir.name}")
                except Exception as e:
                    self.log(f"Failed to clear {cache_dir.name}: {e}", "ERROR")
                    
        return True
        
    def verify_file_changes(self):
        """Verify our changes were actually saved"""
        self.log("Verifying file changes...")
        
        soliton_file = self.frontend_dir / "src" / "lib" / "services" / "solitonMemory.ts"
        
        if not soliton_file.exists():
            self.log("solitonMemory.ts not found!", "ERROR")
            self.issues_found.append("solitonMemory.ts missing")
            return False
            
        content = soliton_file.read_text(encoding='utf-8')
        
        # Check for our fixes
        checks = [
            ("init endpoint", '/api/soliton/init"', content),
            ("user field", '"user": uid', content),
            ("stats endpoint", '/api/soliton/stats/${uid}', content)
        ]
        
        all_good = True
        for name, expected, text in checks:
            if expected in text:
                self.log(f"Fix verified: {name}", "SUCCESS")
            else:
                self.log(f"Fix NOT found: {name}", "ERROR")
                self.issues_found.append(f"Fix missing: {name}")
                all_good = False
                
        return all_good
        
    def check_proxy_actual(self):
        """Check if proxy is actually working"""
        self.log("Testing proxy forwarding...")
        
        # This tests if frontend proxy forwards to backend
        frontend_url = "http://localhost:5173/api/health"
        
        try:
            response = requests.get(frontend_url, timeout=5)
            if response.status_code == 200:
                self.log("Proxy forwarding works!", "SUCCESS")
                return True
            else:
                self.log(f"Proxy returned {response.status_code}", "WARNING")
                self.issues_found.append("Proxy not forwarding correctly")
                return False
        except Exception as e:
            self.log(f"Proxy test failed: {e}", "WARNING")
            # This might be normal if frontend isn't running
            return None
            
    def generate_test_html(self):
        """Generate a standalone test HTML to isolate issues"""
        self.log("Generating standalone test page...", "FIX")
        
        test_html = """<!DOCTYPE html>
<html>
<head>
    <title>TORI API Test Page</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .test { margin: 10px 0; padding: 10px; border: 1px solid #ccc; }
        .success { background-color: #d4edda; }
        .error { background-color: #f8d7da; }
        button { margin: 5px; padding: 10px; cursor: pointer; }
        #log { background: #f0f0f0; padding: 10px; margin-top: 20px; }
        .log-entry { margin: 5px 0; font-family: monospace; }
    </style>
</head>
<body>
    <h1>TORI API Direct Test Page</h1>
    <p>This page tests the API endpoints directly without Svelte/Vite interference.</p>
    
    <div class="test">
        <h3>1. Backend Health Check</h3>
        <button onclick="testHealth()">Test Health</button>
        <div id="health-result"></div>
    </div>
    
    <div class="test">
        <h3>2. Soliton Init Test</h3>
        <button onclick="testSolitonInit()">Test Soliton Init</button>
        <div id="soliton-result"></div>
    </div>
    
    <div class="test">
        <h3>3. Chat Test</h3>
        <input type="text" id="chat-input" placeholder="Type a message..." value="Hello TORI">
        <button onclick="testChat()">Send Message</button>
        <div id="chat-result"></div>
    </div>
    
    <div class="test">
        <h3>4. Upload Test</h3>
        <input type="file" id="file-input" accept=".pdf">
        <button onclick="testUpload()">Test Upload</button>
        <div id="upload-result"></div>
    </div>
    
    <div id="log">
        <h3>Log:</h3>
        <div id="log-entries"></div>
    </div>
    
    <script>
        const API_BASE = 'http://localhost:8002';
        
        function log(message, type = 'info') {
            const logDiv = document.getElementById('log-entries');
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            entry.style.color = type === 'error' ? 'red' : type === 'success' ? 'green' : 'black';
            entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
            logDiv.appendChild(entry);
            console.log(message);
        }
        
        async function testHealth() {
            log('Testing backend health...');
            const resultDiv = document.getElementById('health-result');
            
            try {
                const response = await fetch(`${API_BASE}/api/health`);
                const data = await response.json();
                
                resultDiv.className = 'success';
                resultDiv.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
                log('Health check successful!', 'success');
            } catch (error) {
                resultDiv.className = 'error';
                resultDiv.textContent = `Error: ${error.message}`;
                log(`Health check failed: ${error.message}`, 'error');
            }
        }
        
        async function testSolitonInit() {
            log('Testing soliton init...');
            const resultDiv = document.getElementById('soliton-result');
            
            try {
                const response = await fetch(`${API_BASE}/api/soliton/init`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ user: 'test_user' })
                });
                const data = await response.json();
                
                resultDiv.className = response.ok ? 'success' : 'error';
                resultDiv.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
                log(`Soliton init: ${response.ok ? 'success' : 'failed'}`, response.ok ? 'success' : 'error');
            } catch (error) {
                resultDiv.className = 'error';
                resultDiv.textContent = `Error: ${error.message}`;
                log(`Soliton init error: ${error.message}`, 'error');
            }
        }
        
        async function testChat() {
            log('Testing chat...');
            const resultDiv = document.getElementById('chat-result');
            const input = document.getElementById('chat-input').value;
            
            try {
                const response = await fetch(`${API_BASE}/api/answer`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        user_query: input,
                        persona: {}
                    })
                });
                const data = await response.json();
                
                resultDiv.className = response.ok ? 'success' : 'error';
                resultDiv.innerHTML = `<strong>Response:</strong><br>${data.answer || data.error || 'No response'}`;
                log(`Chat test: ${response.ok ? 'success' : 'failed'}`, response.ok ? 'success' : 'error');
            } catch (error) {
                resultDiv.className = 'error';
                resultDiv.textContent = `Error: ${error.message}`;
                log(`Chat error: ${error.message}`, 'error');
            }
        }
        
        async function testUpload() {
            log('Testing upload...');
            const resultDiv = document.getElementById('upload-result');
            const fileInput = document.getElementById('file-input');
            
            if (!fileInput.files[0]) {
                resultDiv.className = 'error';
                resultDiv.textContent = 'Please select a file first';
                return;
            }
            
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            
            try {
                const progressId = `test_${Date.now()}`;
                const response = await fetch(`${API_BASE}/api/upload?progress_id=${progressId}`, {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                
                resultDiv.className = response.ok ? 'success' : 'error';
                resultDiv.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
                log(`Upload test: ${response.ok ? 'success' : 'failed'}`, response.ok ? 'success' : 'error');
            } catch (error) {
                resultDiv.className = 'error';
                resultDiv.textContent = `Error: ${error.message}`;
                log(`Upload error: ${error.message}`, 'error');
            }
        }
        
        // Test health on load
        window.onload = () => {
            log('Test page loaded. Click buttons to test each endpoint.');
            testHealth();
        };
    </script>
</body>
</html>
"""
        
        test_file = self.base_dir / "tori_api_test.html"
        test_file.write_text(test_html)
        self.log(f"Test page created: {test_file}", "SUCCESS")
        self.fixes_applied.append("Created API test page")
        
        return test_file
        
    def run_diagnostics(self):
        """Run all diagnostics"""
        self.log("=" * 60)
        self.log("TORI COMPREHENSIVE DIAGNOSTICS", "INFO")
        self.log("=" * 60)
        
        # 1. Check backend
        backend_ok = self.check_backend_health()
        
        if backend_ok:
            # 2. Test endpoints
            self.test_soliton_endpoints()
            self.test_prajna_endpoint()
        
        # 3. Check frontend
        self.verify_file_changes()
        cache_dirs = self.check_frontend_cache()
        
        # 4. Test proxy
        proxy_result = self.check_proxy_actual()
        
        # 5. Generate test page
        test_page = self.generate_test_html()
        
        # Report
        self.log("\n" + "=" * 60)
        self.log("DIAGNOSTIC SUMMARY", "INFO")
        self.log("=" * 60)
        
        if self.issues_found:
            self.log(f"Issues found: {len(self.issues_found)}", "WARNING")
            for issue in self.issues_found:
                self.log(f"  • {issue}", "WARNING")
        else:
            self.log("No major issues found!", "SUCCESS")
            
        if self.fixes_applied:
            self.log(f"\nFixes applied: {len(self.fixes_applied)}", "FIX")
            for fix in self.fixes_applied:
                self.log(f"  • {fix}", "FIX")
                
        # Recommendations
        self.log("\n" + "=" * 60)
        self.log("RECOMMENDATIONS", "INFO")
        self.log("=" * 60)
        
        if not backend_ok:
            self.log("1. Restart the backend:", "FIX")
            self.log("   cd C:\\Users\\jason\\Desktop\\tori\\kha", "INFO")
            self.log("   python enhanced_launcher.py", "INFO")
            
        if cache_dirs:
            self.log("2. Clear frontend caches (run this script with --clear-cache)", "FIX")
            
        self.log(f"3. Open the test page in your browser:", "FIX")
        self.log(f"   file:///{test_page}", "INFO")
        self.log("   This will test the API directly without Svelte interference", "INFO")
        
        if self.issues_found:
            self.log("4. After fixing issues, restart both backend and frontend", "FIX")
            
        self.log("\n✅ Diagnostics complete!", "SUCCESS")
        

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="TORI Diagnostic and Fix Tool")
    parser.add_argument("--clear-cache", action="store_true", help="Clear frontend caches")
    parser.add_argument("--force", action="store_true", help="Force actions without prompting")
    
    args = parser.parse_args()
    
    diagnostic = TORIDiagnostic()
    
    if args.clear_cache:
        diagnostic.clear_frontend_cache(force=args.force)
    else:
        diagnostic.run_diagnostics()
        

if __name__ == "__main__":
    main()
