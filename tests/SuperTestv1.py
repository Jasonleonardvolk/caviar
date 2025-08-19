import os
import json
import logging
import asyncio
from pathlib import Path

import requests
import websockets

# === TORI SuperTest v1 (ASCII Log Edition) ===
# All logging is plain text (no emojis or Unicode) for Windows/CP1252 compatibility.

logger = logging.getLogger("SuperTest")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

base_dir = Path(__file__).resolve().parent.parent
logs_dir = base_dir / "logs"
logs_dir.mkdir(exist_ok=True)
file_handler = logging.FileHandler(logs_dir / "SuperTestv1.log", mode="w", encoding="utf-8")
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

logger.info("=== Starting TORI SuperTest v1 ===")

# 1. API Port Detection
mode = None
port = None
try:
    api_port_path = base_dir / "api_port.json"
    if not api_port_path.exists():
        logger.error("FAIL: api_port.json not found at %s", api_port_path)
    else:
        with open(api_port_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        port = data.get("api_port") or data.get("port")
        mode = data.get("api_mode") or data.get("mode")
        if port:
            logger.info("INFO: Found api_port.json: API running on port %s, mode='%s'", port, mode)
            try:
                health_url = f"http://localhost:{port}/api/health"
                r = requests.get(health_url, timeout=3)
                if r.status_code == 200:
                    logger.info("PASS: API health check passed (HTTP 200)")
                else:
                    logger.error("FAIL: API health endpoint returned %d (expected 200)", r.status_code)
            except Exception as e:
                logger.error("FAIL: API health check request failed: %s", e)
        else:
            logger.error("FAIL: api_port.json is missing the API port number")
except Exception as e:
    logger.exception("FAIL: Error during API Port Detection: %s", e)

# 2. API Endpoints
endpoints_to_test = [
    ("GET", "/api/v1/concepts"),
    ("GET", "/api/v1/concept-mesh/status"),
    ("GET", "/api/v1/soliton/status"),
    ("GET", "/api/v1/soliton/query"),
]
if port:
    for method, path in endpoints_to_test:
        url = f"http://localhost:{port}{path}"
        try:
            if method == "GET":
                resp = requests.get(url, timeout=5)
            else:
                resp = requests.post(url, timeout=5)
            try:
                resp_json = resp.json()
            except ValueError:
                resp_json = resp.text.strip()
            logger.info("%s %s -> HTTP %d, Response: %s", method, path, resp.status_code, resp_json)
        except Exception as e:
            logger.error("FAIL: %s %s request failed: %s", method, path, e)
else:
    logger.error("FAIL: Skipping API endpoint tests (API port unknown)")

# Special case: /api/answer (Prajna QA endpoint)
if port:
    try:
        answer_url = f"http://localhost:{port}/api/answer"
        dummy_query = {"user_query": "What is 2+2?", "persona": {"name": "TestPersona", "psi": 0.0}}
        resp = requests.post(answer_url, json=dummy_query, timeout=8)
        try:
            resp_json = resp.json()
        except ValueError:
            resp_json = resp.text.strip()
        if resp.status_code == 200:
            logger.info("PASS: POST /api/answer -> HTTP 200, Response: %s", resp_json)
        else:
            logger.error("FAIL: POST /api/answer returned HTTP %d, Response: %s", resp.status_code, resp_json)
    except Exception as e:
        logger.error("FAIL: POST /api/answer request failed: %s", e)
else:
    logger.error("FAIL: Skipping /api/answer test (API port unknown)")

# 3. Concept Mesh Content Verification
if port:
    try:
        concepts_url = f"http://localhost:{port}/api/v1/concepts"
        r = requests.get(concepts_url, timeout=5)
        if r.status_code != 200:
            logger.error("FAIL: /api/v1/concepts returned HTTP %d (expected 200)", r.status_code)
        else:
            data = r.json()
            if isinstance(data, list):
                count = len(data)
                valid_fields = all(isinstance(item, dict) and "id" in item and "label" in item for item in data[:5])
                if count >= 5 and valid_fields:
                    logger.info("PASS: Concept mesh contains %d concepts with id and label", count)
                else:
                    if count < 5:
                        logger.error("FAIL: Concept list is shorter than expected (only %d concepts)", count)
                    if not valid_fields:
                        logger.error("FAIL: Some concept entries missing 'id' or 'label' fields")
            else:
                logger.error("FAIL: /api/v1/concepts did not return a list (got %s)", type(data).__name__)
    except Exception as e:
        logger.error("FAIL: Error checking concept mesh content: %s", e)
else:
    logger.error("FAIL: Skipping concept mesh content verification (API port unknown)")

# 4. Soliton Memory Store & Query Roundtrip
if port:
    try:
        store_url = f"http://localhost:{port}/api/v1/soliton/store"
        query_url = f"http://localhost:{port}/api/v1/soliton/query"
        test_user = "supertest_user"
        test_content = "Hello Soliton"
        store_payload = {"user_id": test_user, "content": test_content}
        r_store = requests.post(store_url, json=store_payload, timeout=5)
        if r_store.status_code != 200:
            logger.error("FAIL: Soliton store failed (HTTP %d): %s", r_store.status_code, r_store.text.strip())
        else:
            logger.info("PASS: Soliton store succeeded for user '%s'", test_user)
            query_payload = {"user_id": test_user, "query": test_content}
            r_query = requests.post(query_url, json=query_payload, timeout=5)
            if r_query.status_code != 200:
                logger.error("FAIL: Soliton query failed (HTTP %d): %s", r_query.status_code, r_query.text.strip())
            else:
                result = None
                try:
                    result = r_query.json()
                except ValueError:
                    logger.error("FAIL: Soliton query returned non-JSON response: %s", r_query.text.strip())
                if result is not None:
                    found = False
                    if isinstance(result, list):
                        for item in result:
                            if isinstance(item, dict):
                                if item.get("content") == test_content:
                                    found = True
                                    break
                    if found:
                        logger.info("PASS: Soliton memory roundtrip: stored content was retrieved successfully")
                    else:
                        logger.error("FAIL: Soliton memory roundtrip: stored content not found in query results")
    except Exception as e:
        logger.error("FAIL: Exception during soliton memory test: %s", e)
else:
    logger.error("FAIL: Skipping soliton memory tests (API port unknown)")

# 5. WebSocket Bridges (Audio & Concept)
bridge_ports = [("Audio Bridge", 8765), ("Concept Bridge", 8766)]
for name, br_port in bridge_ports:
    try:
        r_health = requests.get(f"http://localhost:{br_port}/health", timeout=3)
        if r_health.status_code == 200:
            logger.info("PASS: %s health check passed (HTTP 200 at /health)", name)
        else:
            logger.error("FAIL: %s /health returned %d (expected 200)", name, r_health.status_code)
    except Exception as e:
        logger.error("FAIL: %s /health request failed: %s", name, e)
    try:
        async def ws_handshake_test(uri):
            async with websockets.connect(uri):
                return True
        asyncio.run(ws_handshake_test(f"ws://localhost:{br_port}"))
        logger.info("PASS: %s WebSocket handshake successful (port %d)", name, br_port)
    except Exception as e:
        logger.error("FAIL: %s WebSocket handshake failed: %s", name, e)

# 6. Launcher Mode Verification
if mode:
    if str(mode).lower() != "full":
        logger.warning("WARN: Launcher mode is '%s' (expected 'full') - some features may be disabled", mode)
    else:
        logger.info("PASS: Launcher is running in full mode")
else:
    logger.error("FAIL: Could not determine launcher mode from api_port.json")

logger.info("=== SuperTestv1 complete ===")
