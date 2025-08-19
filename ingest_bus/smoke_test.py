#!/usr/bin/env python3
"""
Smoke Test for TORI Ingest Bus

This script tests the basic functionality of the Ingest Bus service.
It performs the following tests:
1. Check if the service is running
2. Test the queue API
3. Test the status API
4. Test the metrics API
5. Test the KB search API if ScholarSphere is available

Usage:
    python smoke_test.py [--host HOST] [--port PORT]

Options:
    --host HOST    Host where the service is running (default: localhost)
    --port PORT    Port where the service is running (default: 8080)
"""

import argparse
import json
import sys
import time
import requests
from datetime import datetime
from pathlib import Path

# Parse arguments
parser = argparse.ArgumentParser(description="Smoke test for TORI Ingest Bus")
parser.add_argument("--host", default="localhost", help="Host where the service is running")
parser.add_argument("--port", default="8080", help="Port where the service is running")
args = parser.parse_args()

BASE_URL = f"http://{args.host}:{args.port}"

def log(message, level="INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

def check_service():
    log("Checking if service is running...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            log(f"Service is running: {data}")
            return True
        else:
            log(f"Service returned unexpected status code: {response.status_code}", "ERROR")
            return False
    except requests.exceptions.ConnectionError:
        log(f"Could not connect to service at {BASE_URL}", "ERROR")
        return False

def test_queue_api():
    log("Testing queue API...")
    # Create a sample text file for testing
    test_file_path = Path("test_sample.txt")
    test_file_path.write_text("This is a test document for the Ingest Bus smoke test.")
    
    try:
        # Create a request to queue the test file
        files = {
            'file': ('test_sample.txt', open(test_file_path, 'rb'), 'text/plain')
        }
        
        data = {
            'document_type': 'text',
            'title': 'Smoke Test Document',
            'description': 'Test document for smoke testing',
            'tags': json.dumps(['test', 'smoke']),
            'metadata': json.dumps({'test': True, 'timestamp': datetime.now().isoformat()})
        }
        
        response = requests.post(f"{BASE_URL}/queue", files=files, data=data)
        
        if response.status_code in [200, 201, 202]:
            job_data = response.json()
            log(f"Successfully queued document: {job_data.get('id')}")
            return job_data.get('id')
        else:
            log(f"Failed to queue document: {response.status_code} - {response.text}", "ERROR")
            return None
    except Exception as e:
        log(f"Error queuing document: {str(e)}", "ERROR")
        return None
    finally:
        # Clean up test file
        if test_file_path.exists():
            test_file_path.unlink()

def test_status_api(job_id):
    if not job_id:
        log("Skipping status API test because job_id is not available", "WARNING")
        return False
    
    log(f"Testing status API for job {job_id}...")
    try:
        # Wait a bit for processing to start
        time.sleep(2)
        
        # Check job status
        response = requests.get(f"{BASE_URL}/status/job/{job_id}")
        
        if response.status_code == 200:
            job_data = response.json()
            log(f"Job status: {job_data.get('status')}")
            log(f"Job progress: {job_data.get('percent_complete')}%")
            
            # Wait for job to complete or fail
            max_wait = 30  # seconds
            wait_time = 0
            while job_data.get('status') not in ['completed', 'failed'] and wait_time < max_wait:
                time.sleep(2)
                wait_time += 2
                response = requests.get(f"{BASE_URL}/status/job/{job_id}")
                if response.status_code == 200:
                    job_data = response.json()
                    log(f"Job status: {job_data.get('status')}")
                    log(f"Job progress: {job_data.get('percent_complete')}%")
                else:
                    log(f"Failed to get job status: {response.status_code} - {response.text}", "ERROR")
                    return False
            
            if job_data.get('status') == 'completed':
                log("Job completed successfully")
                return True
            elif job_data.get('status') == 'failed':
                log(f"Job failed: {job_data.get('failure_message')}", "ERROR")
                return False
            else:
                log(f"Job did not complete within {max_wait} seconds", "WARNING")
                return False
        else:
            log(f"Failed to get job status: {response.status_code} - {response.text}", "ERROR")
            return False
    except Exception as e:
        log(f"Error checking job status: {str(e)}", "ERROR")
        return False

def test_metrics_api():
    log("Testing metrics API...")
    try:
        response = requests.get(f"{BASE_URL}/metrics")
        
        if response.status_code == 200:
            metrics_data = response.json()
            log(f"Metrics available: {len(metrics_data.keys())} metrics")
            log(f"Sample metrics: ingest_files_queued_total={metrics_data.get('ingest_files_queued_total')}")
            
            # Also test Prometheus metrics endpoint
            prom_response = requests.get(f"{BASE_URL}/metrics/prometheus")
            if prom_response.status_code == 200:
                log("Prometheus metrics available")
                return True
            else:
                log(f"Failed to get Prometheus metrics: {prom_response.status_code}", "WARNING")
                return False
        else:
            log(f"Failed to get metrics: {response.status_code} - {response.text}", "ERROR")
            return False
    except Exception as e:
        log(f"Error checking metrics: {str(e)}", "ERROR")
        return False

def test_kb_search():
    log("Testing KB search API...")
    try:
        # Simple search query
        query = "test document"
        response = requests.post(
            f"{BASE_URL}/kb/search",
            json={"query": query, "limit": 5, "min_relevance": 0.5}
        )
        
        if response.status_code == 200:
            search_data = response.json()
            log(f"Search results: {len(search_data.get('results', []))} results")
            return True
        elif response.status_code == 404:
            log("KB search API not available yet (no content)", "WARNING")
            return False
        else:
            log(f"Failed to search KB: {response.status_code} - {response.text}", "WARNING")
            return False
    except Exception as e:
        log(f"Error searching KB: {str(e)}", "WARNING")
        return False

def run_all_tests():
    log("Starting smoke tests for TORI Ingest Bus...")
    
    # Track test results
    results = {
        "service_check": False,
        "queue_api": False,
        "status_api": False,
        "metrics_api": False,
        "kb_search": False
    }
    
    # Check if service is running
    results["service_check"] = check_service()
    if not results["service_check"]:
        log("Service check failed, aborting remaining tests", "ERROR")
        return results
    
    # Test queue API
    job_id = test_queue_api()
    results["queue_api"] = job_id is not None
    
    # Test status API
    results["status_api"] = test_status_api(job_id)
    
    # Test metrics API
    results["metrics_api"] = test_metrics_api()
    
    # Test KB search API
    results["kb_search"] = test_kb_search()
    
    return results

if __name__ == "__main__":
    results = run_all_tests()
    
    # Print summary
    log("\nTest Results Summary:", "SUMMARY")
    log(f"Service Check: {'✅ PASS' if results['service_check'] else '❌ FAIL'}", "SUMMARY")
    log(f"Queue API: {'✅ PASS' if results['queue_api'] else '❌ FAIL'}", "SUMMARY")
    log(f"Status API: {'✅ PASS' if results['status_api'] else '❌ FAIL'}", "SUMMARY")
    log(f"Metrics API: {'✅ PASS' if results['metrics_api'] else '❌ FAIL'}", "SUMMARY")
    log(f"KB Search: {'✅ PASS' if results['kb_search'] else '⚠️ WARN - May not be available yet'}", "SUMMARY")
    
    # Determine overall result
    critical_tests = ['service_check', 'queue_api', 'status_api', 'metrics_api']
    if all(results[test] for test in critical_tests):
        log("\nSmoke test PASSED! The Ingest Bus service is functioning correctly.", "SUCCESS")
        sys.exit(0)
    else:
        log("\nSmoke test FAILED! The Ingest Bus service has issues that need to be addressed.", "ERROR")
        sys.exit(1)
