"""
Test the Prajna-TONKA Coordinator
Shows how Prajna can understand complex requests and delegate to TONKA
"""

import requests
import json

# API base URL
BASE_URL = "http://localhost:8002"

def test_coordination_examples():
    """Test various coordination patterns"""
    
    examples = [
        # Direct TONKA commands
        {
            "query": "Tonka, create a web scraper for news articles",
            "expected": "direct command to TONKA"
        },
        {
            "query": "@tonka implement a binary search tree",
            "expected": "@ mention pattern"
        },
        {
            "query": "Hey tonka, build me a REST API with authentication",
            "expected": "friendly command"
        },
        
        # Implicit code requests
        {
            "query": "I need a function to validate email addresses",
            "expected": "implicit code need"
        },
        {
            "query": "Show me how to connect to a PostgreSQL file_storage",
            "expected": "how-to pattern"
        },
        
        # Complex multi-part requests
        {
            "query": "Create an API project called TaskManager and implement user authentication",
            "expected": "multiple orders (project + feature)"
        },
        {
            "query": "Write a sorting algorithm and then create tests for it",
            "expected": "code + tests"
        },
        
        # Project creation
        {
            "query": "Bootstrap a new FastAPI project called awesome_api",
            "expected": "project scaffolding"
        },
        
        # Algorithm requests
        {
            "query": "Show me the quicksort algorithm in Python",
            "expected": "specific algorithm"
        },
        
        # Mixed requests
        {
            "query": "What is consciousness? Also, create a function to calculate fibonacci numbers",
            "expected": "philosophy + code (should handle both)"
        }
    ]
    
    print("🧪 Testing Prajna-TONKA Coordination")
    print("=" * 70)
    
    for example in examples:
        print(f"\n📝 Query: '{example['query']}'")
        print(f"   Expected: {example['expected']}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/api/chat",
                json={"user_query": example["query"]},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Check if code was generated
                if data.get("code_generated"):
                    print("   ✅ Code generation triggered!")
                    
                    # Show coordinator analysis if available
                    if "coordinator_analysis" in data:
                        analysis = data["coordinator_analysis"]
                        print(f"   📋 Orders found: {len(analysis.get('orders', []))}")
                        for order in analysis.get("orders", []):
                            print(f"      - {order['task_type']}: {order['description'][:50]}...")
                    
                    # Show preview of response
                    answer = data.get("answer", "")
                    if "```" in answer:
                        print("   💻 Code block detected in response")
                    else:
                        print(f"   📄 Response preview: {answer[:100]}...")
                        
                else:
                    print("   💬 Handled by Prajna (non-code request)")
                    print(f"   📄 Response preview: {data.get('answer', '')[:100]}...")
                    
            else:
                print(f"   ❌ Request failed: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

def test_coordinator_patterns():
    """Test specific coordinator patterns"""
    print("\n\n🔍 Testing Coordinator Pattern Recognition")
    print("=" * 70)
    
    # Import the coordinator directly to test pattern matching
    try:
        from api.prajna_tonka_coordinator import coordinator, TonkaTaskType
        
        test_patterns = [
            "tonka: create a calculator app",
            "implement bubble sort algorithm",
            "new api project my_service",
            "explain this code: def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
            "refactor this messy function to be cleaner",
            "debug why my code is not working",
            "how do I parse JSON in Python",
        ]
        
        for pattern in test_patterns:
            order = coordinator.analyze_request(pattern)
            if order:
                print(f"\n✅ Pattern: '{pattern}'")
                print(f"   Task Type: {order.task_type.value}")
                print(f"   Description: {order.description}")
                print(f"   Language: {order.language}")
                print(f"   Style: {order.style}")
            else:
                print(f"\n❌ No match: '{pattern}'")
                
    except ImportError:
        print("⚠️ Could not import coordinator for direct testing")

def test_multi_order_extraction():
    """Test extraction of multiple orders from complex requests"""
    print("\n\n🎯 Testing Multi-Order Extraction")
    print("=" * 70)
    
    complex_queries = [
        "Create an API project and also implement user authentication and then add file_storage models",
        "Write a function to parse CSV files, then create tests for it, and also document the code",
        "Tonka, build a web scraper and create a CLI tool to run it",
    ]
    
    for query in complex_queries:
        print(f"\n📝 Complex Query: '{query}'")
        
        response = requests.post(
            f"{BASE_URL}/api/chat",
            json={"user_query": query},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            
            if "coordinator_analysis" in data:
                orders = data["coordinator_analysis"].get("orders", [])
                print(f"   📋 Found {len(orders)} orders:")
                for i, order in enumerate(orders):
                    print(f"      {i+1}. {order['task_type']}: {order['description']}")
            else:
                print("   ⚠️ No coordinator analysis in response")

def main():
    """Run all tests"""
    print("🚀 Prajna-TONKA Coordinator Test Suite")
    print("Make sure the API is running: python enhanced_launcher.py")
    print("\n")
    
    try:
        # Check if API is running
        response = requests.get(f"{BASE_URL}/api/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print("✅ API is running")
            print(f"   TONKA: {health.get('tonka_ready', False)}")
            print(f"   Prajna: {health.get('prajna_loaded', False)}")
        else:
            print("❌ API health check failed")
            return
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API at", BASE_URL)
        print("Please start the server with: python enhanced_launcher.py")
        return
    
    # Run tests
    test_coordination_examples()
    test_coordinator_patterns()
    test_multi_order_extraction()
    
    print("\n\n✅ All tests completed!")
    print("\n💡 TIP: Try these in the chat interface:")
    print("   - 'Tonka, create a todo list API'")
    print("   - '@tonka implement merge sort'")
    print("   - 'I need a function to validate credit card numbers'")
    print("   - 'Create an API project and add user authentication'")

if __name__ == "__main__":
    main()
