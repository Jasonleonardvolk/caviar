#!/usr/bin/env python3
"""
🐛 SOLITON 500 ERROR REAL-TIME DEBUGGER
=====================================
Continuously monitors soliton endpoints and captures 500 errors in real-time.
Writes detailed error analysis to file.

Usage: python soliton_500_debugger.py
"""

import requests
import json
import time
import traceback
from datetime import datetime
import threading

class SolitonDebugger:
    def __init__(self):
        self.base_url = "http://localhost:8002"
        self.debug_file = f"SOLITON_500_DEBUG_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self.running = True
        self.error_count = 0
        
    def log_to_file(self, message):
        """Write to both console and file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        formatted = f"[{timestamp}] {message}"
        print(formatted)
        
        try:
            with open(self.debug_file, 'a', encoding='utf-8') as f:
                f.write(formatted + "\n")
        except Exception as e:
            print(f"Failed to write to file: {e}")
    
    def test_soliton_init_variations(self):
        """Test different ways of calling soliton init to trigger 500 error"""
        variations = [
            {
                "name": "Empty JSON Body",
                "data": {},
                "headers": {"Content-Type": "application/json"}
            },
            {
                "name": "Valid JSON Body",
                "data": {"userId": "test_user"},
                "headers": {"Content-Type": "application/json"}
            },
            {
                "name": "No Body",
                "data": None,
                "headers": {"Content-Type": "application/json"}
            },
            {
                "name": "Invalid JSON",
                "data": "invalid json",
                "headers": {"Content-Type": "application/json"},
                "raw": True
            },
            {
                "name": "No Content-Type",
                "data": {"userId": "test_user"},
                "headers": {}
            }
        ]
        
        for i, variation in enumerate(variations):
            self.log_to_file(f"\n🧪 TEST #{i+1}: {variation['name']}")
            self.log_to_file("=" * 50)
            
            try:
                url = f"{self.base_url}/api/soliton/init"
                self.log_to_file(f"URL: {url}")
                self.log_to_file(f"Headers: {variation['headers']}")
                
                if variation.get('raw'):
                    self.log_to_file(f"Raw Data: {variation['data']}")
                    response = requests.post(
                        url, 
                        data=variation['data'],
                        headers=variation['headers'],
                        timeout=5
                    )
                else:
                    self.log_to_file(f"JSON Data: {json.dumps(variation['data'])}")
                    if variation['data'] is None:
                        response = requests.post(
                            url,
                            headers=variation['headers'],
                            timeout=5
                        )
                    else:
                        response = requests.post(
                            url,
                            json=variation['data'],
                            headers=variation['headers'],
                            timeout=5
                        )
                
                # Log response details
                self.log_to_file(f"Response Status: {response.status_code}")
                self.log_to_file(f"Response Headers: {dict(response.headers)}")
                
                if response.status_code == 500:
                    self.error_count += 1
                    self.log_to_file("🚨 500 ERROR DETECTED!")
                    self.log_to_file(f"Raw Response: {response.text}")
                    
                    # Try to parse error details
                    try:
                        error_json = response.json()
                        self.log_to_file(f"Error JSON: {json.dumps(error_json, indent=2)}")
                    except:
                        self.log_to_file("Response is not valid JSON")
                        
                else:
                    self.log_to_file(f"✅ Success! Status: {response.status_code}")
                    try:
                        success_json = response.json()
                        self.log_to_file(f"Success Response: {json.dumps(success_json, indent=2)}")
                    except:
                        self.log_to_file(f"Response Text: {response.text}")
                
            except Exception as e:
                self.log_to_file(f"❌ Exception during test: {str(e)}")
                self.log_to_file(f"Traceback: {traceback.format_exc()}")
            
            time.sleep(1)  # Brief pause between tests
    
    def monitor_continuous(self):
        """Continuously monitor soliton endpoints"""
        self.log_to_file("🔄 Starting continuous monitoring...")
        
        test_data = {"userId": "monitor_user"}
        
        while self.running:
            try:
                response = requests.post(
                    f"{self.base_url}/api/soliton/init",
                    json=test_data,
                    timeout=2
                )
                
                if response.status_code == 500:
                    self.error_count += 1
                    self.log_to_file(f"🚨 CONTINUOUS MONITOR - 500 ERROR #{self.error_count}")
                    self.log_to_file(f"Time: {datetime.now()}")
                    self.log_to_file(f"Response: {response.text}")
                    
                    # Break after first error for analysis
                    self.log_to_file("Stopping continuous monitoring due to 500 error")
                    break
                elif response.status_code == 200:
                    self.log_to_file("✅ Continuous monitor - endpoint working")
                else:
                    self.log_to_file(f"⚠️ Unexpected status: {response.status_code}")
                
            except Exception as e:
                self.log_to_file(f"❌ Monitor exception: {str(e)}")
                break
            
            time.sleep(5)  # Check every 5 seconds
    
    def analyze_backend_logs(self):
        """Try to capture backend logs if possible"""
        self.log_to_file("\n📋 BACKEND LOG ANALYSIS")
        self.log_to_file("=" * 50)
        
        # Try to find common log locations
        possible_log_paths = [
            "prajna_api.log",
            "logs/prajna.log", 
            "logs/api.log",
            "tori.log"
        ]
        
        for log_path in possible_log_paths:
            try:
                if os.path.exists(log_path):
                    self.log_to_file(f"📄 Found log file: {log_path}")
                    with open(log_path, 'r') as f:
                        lines = f.readlines()
                        recent_lines = lines[-50:]  # Last 50 lines
                        self.log_to_file("Recent log entries:")
                        for line in recent_lines:
                            if "500" in line or "ERROR" in line or "soliton" in line.lower():
                                self.log_to_file(f"  ⚠️ {line.strip()}")
                else:
                    self.log_to_file(f"❌ Log file not found: {log_path}")
            except Exception as e:
                self.log_to_file(f"❌ Error reading {log_path}: {e}")
    
    def run_debug_session(self):
        """Run complete debugging session"""
        self.log_to_file("🐛 SOLITON 500 ERROR DEBUGGING SESSION")
        self.log_to_file("=" * 60)
        self.log_to_file(f"Session started: {datetime.now()}")
        self.log_to_file(f"Debug file: {self.debug_file}")
        self.log_to_file("")
        
        # Test different request variations
        self.test_soliton_init_variations()
        
        # Analyze backend logs
        self.analyze_backend_logs()
        
        # Summary
        self.log_to_file("\n📊 DEBUGGING SUMMARY")
        self.log_to_file("=" * 50)
        self.log_to_file(f"Total 500 errors detected: {self.error_count}")
        
        if self.error_count > 0:
            self.log_to_file("\n💡 RECOMMENDATIONS TO FIX 500 ERRORS:")
            self.log_to_file("1. Check FastAPI endpoint parameter handling")
            self.log_to_file("2. Verify Pydantic model requirements")
            self.log_to_file("3. Add proper exception handling in soliton endpoints")
            self.log_to_file("4. Consider making request parameters optional")
            self.log_to_file("5. Check for import errors or missing dependencies")
        else:
            self.log_to_file("✅ No 500 errors detected - endpoints working properly!")
        
        self.log_to_file(f"\nSession completed: {datetime.now()}")
        self.log_to_file("Check this file for detailed error analysis.")

def main():
    """Run the soliton debugger"""
    print("🐛 Soliton 500 Error Real-Time Debugger")
    print("This will test soliton endpoints and catch 500 errors!")
    print("Press Ctrl+C to stop...")
    print("=" * 50)
    
    debugger = SolitonDebugger()
    
    try:
        debugger.run_debug_session()
        
        print(f"\n📄 Debugging complete!")
        print(f"📝 Detailed analysis saved to: {debugger.debug_file}")
        print(f"🚨 Total 500 errors found: {debugger.error_count}")
        
    except KeyboardInterrupt:
        print("\n⏹️ Debugging stopped by user")
        debugger.running = False
        print(f"📝 Partial results saved to: {debugger.debug_file}")

if __name__ == "__main__":
    import os
    main()
