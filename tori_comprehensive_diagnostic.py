#!/usr/bin/env python3
"""
🔬 TORI COMPREHENSIVE DIAGNOSTIC SCRIPT
=======================================
The "Big Juicy" diagnostic that gives you EVERY insight into your TORI system.

This script will:
- Test every single endpoint
- Capture detailed request/response data
- Monitor system performance
- Check file system state
- Analyze configuration
- Write comprehensive report to file
- Give actionable insights

Usage: python tori_comprehensive_diagnostic.py
"""

import requests
import json
import time
import traceback
import os
import sys
from pathlib import Path
from datetime import datetime
import subprocess
import psutil

class TORIDiagnostic:
    def __init__(self):
        self.base_url = "http://localhost:8002"
        self.frontend_url = "http://localhost:5173"
        self.results = []
        self.start_time = time.time()
        self.report_file = f"TORI_DIAGNOSTIC_REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
    def log(self, message, level="INFO"):
        """Log message to both console and results"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        formatted = f"[{timestamp}] {level}: {message}"
        print(formatted)
        self.results.append(formatted)
        
    def test_endpoint(self, method, url, data=None, timeout=10, description=""):
        """Test an endpoint with comprehensive logging"""
        self.log(f"🧪 TESTING: {description or url}", "TEST")
        self.log(f"   Method: {method}")
        self.log(f"   URL: {url}")
        if data:
            self.log(f"   Data: {json.dumps(data, indent=2)}")
            
        start_time = time.time()
        
        try:
            if method.upper() == 'GET':
                response = requests.get(url, timeout=timeout)
            elif method.upper() == 'POST':
                if data:
                    response = requests.post(url, json=data, timeout=timeout)
                else:
                    response = requests.post(url, timeout=timeout)
                    
            duration = time.time() - start_time
            
            # Detailed response logging
            self.log(f"   ✅ Response Time: {duration:.3f}s")
            self.log(f"   📊 Status Code: {response.status_code}")
            self.log(f"   📋 Headers: {dict(response.headers)}")
            
            if response.headers.get('content-type', '').startswith('application/json'):
                try:
                    json_response = response.json()
                    self.log(f"   📄 JSON Response: {json.dumps(json_response, indent=2)}")
                    return {
                        "status": "SUCCESS",
                        "status_code": response.status_code,
                        "response": json_response,
                        "duration": duration,
                        "description": description
                    }
                except json.JSONDecodeError:
                    self.log(f"   ⚠️ JSON Decode Error")
                    
            text_response = response.text[:500] + "..." if len(response.text) > 500 else response.text
            self.log(f"   📝 Text Response: {text_response}")
            
            return {
                "status": "SUCCESS",
                "status_code": response.status_code,
                "response": response.text,
                "duration": duration,
                "description": description
            }
            
        except requests.exceptions.ConnectionError as e:
            self.log(f"   ❌ CONNECTION ERROR: {str(e)}", "ERROR")
            return {"status": "CONNECTION_ERROR", "error": str(e), "description": description}
        except requests.exceptions.Timeout as e:
            self.log(f"   ⏰ TIMEOUT: {str(e)}", "ERROR")
            return {"status": "TIMEOUT", "error": str(e), "description": description}
        except Exception as e:
            self.log(f"   💥 EXCEPTION: {str(e)}", "ERROR")
            self.log(f"   📚 Traceback: {traceback.format_exc()}", "ERROR")
            return {"status": "ERROR", "error": str(e), "traceback": traceback.format_exc(), "description": description}
    
    def check_system_resources(self):
        """Check system resource usage"""
        self.log("🖥️ SYSTEM RESOURCE CHECK", "SYSTEM")
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.log(f"   CPU Usage: {cpu_percent}%")
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.log(f"   Memory Usage: {memory.percent}% ({memory.used / 1024**3:.1f}GB / {memory.total / 1024**3:.1f}GB)")
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.log(f"   Disk Usage: {disk.percent}% ({disk.used / 1024**3:.1f}GB / {disk.total / 1024**3:.1f}GB)")
            
            # Running processes related to TORI
            self.log("   🔍 TORI-related processes:")
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
                try:
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    if any(keyword in cmdline.lower() for keyword in ['tori', 'prajna', 'enhanced_launcher', 'uvicorn', 'node', 'svelte']):
                        self.log(f"     PID {proc.info['pid']}: {proc.info['name']} - {cmdline[:100]}")
                        self.log(f"       CPU: {proc.info['cpu_percent']}%, Memory: {proc.info['memory_info'].rss / 1024**2:.1f}MB")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
        except Exception as e:
            self.log(f"   ❌ System check failed: {e}", "ERROR")
    
    def check_file_system(self):
        """Check TORI file system state"""
        self.log("📁 FILE SYSTEM CHECK", "SYSTEM")
        
        # Key directories to check
        tori_root = Path(__file__).parent
        key_paths = [
            tori_root / "prajna" / "api" / "prajna_api.py",
            tori_root / "tori_ui_svelte" / "vite.config.js",
            tori_root / "enhanced_launcher_improved.py",
            tori_root / "enhanced_launcher.py",
            tori_root / "tmp",
            tori_root / "soliton_concept_memory.json",
        ]
        
        for path in key_paths:
            if path.exists():
                if path.is_file():
                    size = path.stat().st_size
                    modified = datetime.fromtimestamp(path.stat().st_mtime)
                    self.log(f"   ✅ {path.name}: {size} bytes, modified {modified}")
                else:
                    files_count = len(list(path.iterdir())) if path.is_dir() else 0
                    self.log(f"   ✅ {path.name}/: {files_count} items")
            else:
                self.log(f"   ❌ {path.name}: NOT FOUND")
    
    def run_comprehensive_tests(self):
        """Run all diagnostic tests"""
        self.log("🚀 STARTING COMPREHENSIVE TORI DIAGNOSTIC", "START")
        self.log(f"   Report will be saved to: {self.report_file}")
        self.log("=" * 80)
        
        # System checks first
        self.check_system_resources()
        self.log("=" * 80)
        
        self.check_file_system()
        self.log("=" * 80)
        
        # Test backend health
        self.log("🏥 BACKEND HEALTH TESTS", "SECTION")
        health_tests = [
            ("GET", f"{self.base_url}/api/health", None, "Basic Health Check"),
            ("GET", f"{self.base_url}/api/stats", None, "Basic Stats"),
            ("GET", f"{self.base_url}/api/prajna/stats", None, "Prajna Stats"),
        ]
        
        backend_healthy = False
        for method, url, data, desc in health_tests:
            result = self.test_endpoint(method, url, data, description=desc)
            if result["status"] == "SUCCESS" and result["status_code"] == 200:
                backend_healthy = True
        
        self.log("=" * 80)
        
        # Test soliton endpoints (the problematic ones)
        self.log("🌊 SOLITON MEMORY TESTS", "SECTION")
        soliton_tests = [
            ("GET", f"{self.base_url}/api/soliton/health", None, "Soliton Health Check"),
            ("POST", f"{self.base_url}/api/soliton/init", {"userId": "test_user"}, "Soliton Initialization"),
            ("POST", f"{self.base_url}/api/soliton/init", {}, "Soliton Init with Empty Body"),
            ("POST", f"{self.base_url}/api/soliton/init", None, "Soliton Init with No Body"),
            ("POST", f"{self.base_url}/api/soliton/store", {
                "userId": "test_user",
                "conceptId": "test_concept",
                "content": "test memory content",
                "importance": 0.8
            }, "Soliton Memory Store"),
            ("GET", f"{self.base_url}/api/soliton/recall/test_user/test_concept", None, "Soliton Memory Recall"),
            ("GET", f"{self.base_url}/api/soliton/stats/test_user", None, "Soliton Stats"),
        ]
        
        soliton_working = 0
        for method, url, data, desc in soliton_tests:
            result = self.test_endpoint(method, url, data, description=desc)
            if result["status"] == "SUCCESS" and result["status_code"] in [200, 201]:
                soliton_working += 1
        
        self.log("=" * 80)
        
        # Test upload endpoint
        self.log("📤 UPLOAD ENDPOINT TESTS", "SECTION")
        upload_tests = [
            ("GET", f"{self.base_url}/upload", None, "Upload Endpoint GET (should fail)"),
            ("GET", f"{self.base_url}/api/upload", None, "API Upload Endpoint GET (should fail)"),
        ]
        
        for method, url, data, desc in upload_tests:
            result = self.test_endpoint(method, url, data, description=desc)
        
        self.log("=" * 80)
        
        # Test frontend connectivity
        self.log("🌐 FRONTEND CONNECTIVITY TESTS", "SECTION")
        frontend_tests = [
            ("GET", f"{self.frontend_url}/", None, "Frontend Root"),
        ]
        
        frontend_healthy = False
        for method, url, data, desc in frontend_tests:
            result = self.test_endpoint(method, url, data, timeout=5, description=desc)
            if result["status"] == "SUCCESS":
                frontend_healthy = True
        
        self.log("=" * 80)
        
        # Generate insights and recommendations
        self.generate_insights(backend_healthy, soliton_working, len(soliton_tests), frontend_healthy)
        
        # Write report to file
        self.write_report()
        
    def generate_insights(self, backend_healthy, soliton_working, soliton_total, frontend_healthy):
        """Generate actionable insights"""
        self.log("🧠 DIAGNOSTIC INSIGHTS & RECOMMENDATIONS", "INSIGHTS")
        
        total_time = time.time() - self.start_time
        self.log(f"   Total diagnostic time: {total_time:.2f} seconds")
        
        # Backend analysis
        if backend_healthy:
            self.log("   ✅ Backend is responding to basic health checks")
        else:
            self.log("   ❌ Backend is not responding - check if enhanced_launcher.py is running")
            self.log("   💡 SOLUTION: Run 'python enhanced_launcher.py' in TORI directory")
        
        # Soliton analysis
        soliton_success_rate = (soliton_working / soliton_total) * 100
        self.log(f"   📊 Soliton endpoint success rate: {soliton_success_rate:.1f}% ({soliton_working}/{soliton_total})")
        
        if soliton_success_rate < 50:
            self.log("   ❌ CRITICAL: Soliton endpoints are mostly failing")
            self.log("   💡 SOLUTIONS:")
            self.log("     1. Check backend logs for 500 errors")
            self.log("     2. Verify Pydantic model compatibility")
            self.log("     3. Check request body parsing in FastAPI")
            self.log("     4. Consider fallback to Dict[str, Any] for problematic endpoints")
        elif soliton_success_rate < 80:
            self.log("   ⚠️ WARNING: Some soliton endpoints are failing")
            self.log("   💡 SOLUTIONS:")
            self.log("     1. Check specific failing endpoints")
            self.log("     2. Verify request format compatibility")
        else:
            self.log("   ✅ Soliton endpoints are mostly working")
        
        # Frontend analysis
        if frontend_healthy:
            self.log("   ✅ Frontend is accessible")
        else:
            self.log("   ❌ Frontend is not responding")
            self.log("   💡 SOLUTIONS:")
            self.log("     1. Check if 'npm run dev' is running in tori_ui_svelte/")
            self.log("     2. Verify port 5173 is not blocked")
            self.log("     3. Check Vite configuration")
        
        # Integration analysis
        self.log("   🔗 INTEGRATION ANALYSIS:")
        if backend_healthy and frontend_healthy:
            if soliton_success_rate > 80:
                self.log("     ✅ Full system integration looks good")
                self.log("     🎯 NEXT STEPS: Test PDF upload in browser")
            else:
                self.log("     ⚠️ Backend/Frontend OK, but soliton issues may cause frontend errors")
                self.log("     🎯 PRIORITY: Fix soliton 500 errors first")
        else:
            self.log("     ❌ Basic connectivity issues need to be resolved first")
        
        # Performance analysis
        self.log("   ⚡ PERFORMANCE NOTES:")
        self.log("     - Backend response times should be < 1 second for health checks")
        self.log("     - Soliton endpoints should respond in < 500ms")
        self.log("     - Frontend should load in < 3 seconds")
        
    def write_report(self):
        """Write comprehensive report to file"""
        try:
            with open(self.report_file, 'w', encoding='utf-8') as f:
                f.write("TORI COMPREHENSIVE DIAGNOSTIC REPORT\n")
                f.write("=" * 50 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Duration: {time.time() - self.start_time:.2f} seconds\n")
                f.write("=" * 50 + "\n\n")
                
                for line in self.results:
                    f.write(line + "\n")
                
                f.write("\n" + "=" * 50 + "\n")
                f.write("END OF DIAGNOSTIC REPORT\n")
            
            self.log(f"📝 Report written to: {self.report_file}", "SUCCESS")
            self.log(f"   File size: {os.path.getsize(self.report_file)} bytes")
            
        except Exception as e:
            self.log(f"❌ Failed to write report: {e}", "ERROR")

def main():
    """Run the comprehensive diagnostic"""
    print("🔬 TORI Comprehensive Diagnostic Starting...")
    print("This will test EVERYTHING and give you detailed insights!")
    print("=" * 60)
    
    diagnostic = TORIDiagnostic()
    diagnostic.run_comprehensive_tests()
    
    print("\n" + "=" * 60)
    print("🎉 Diagnostic Complete!")
    print(f"📄 Detailed report saved to: {diagnostic.report_file}")
    print("💡 Check the file for comprehensive insights and solutions!")

if __name__ == "__main__":
    main()
