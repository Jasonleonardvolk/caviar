"""
Test script for Phase 4: Holographic Memory Ingestion
Tests image, audio, and video processing with cross-modal connections
"""

import asyncio
import aiohttp
import json
import time
from pathlib import Path
from typing import Dict, Any
import base64
import io
from PIL import Image
import numpy as np

# Test configuration
API_BASE_URL = "http://localhost:8000"
TEST_USER = "hologram_test_user"

# Create test media files
def create_test_image() -> bytes:
    """Create a simple test image"""
    # Create a 100x100 red square
    img = Image.new('RGB', (100, 100), color='red')
    
    # Add some text
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.text((10, 40), "TEST", fill='white')
    
    # Convert to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes.read()

def create_test_audio() -> bytes:
    """Create a simple test audio (mock WAV header)"""
    # This is a mock - in real tests, use actual audio
    # WAV header for 1 second of silence
    wav_header = b'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00'
    silence = b'\x00' * 44100 * 2  # 1 second at 44.1kHz, 16-bit
    return wav_header + silence

async def test_image_upload():
    """Test image upload and processing"""
    print("\n🧪 Testing Image Upload...")
    
    async with aiohttp.ClientSession() as session:
        # Create test image
        image_data = create_test_image()
        
        # Prepare multipart upload
        data = aiohttp.FormData()
        data.add_field('file',
                      image_data,
                      filename='test_image.png',
                      content_type='image/png')
        data.add_field('title', 'Test Image')
        data.add_field('tags', 'test,image,red')
        
        # Upload
        async with session.post(
            f"{API_BASE_URL}/api/hologram/upload?userId={TEST_USER}",
            data=data
        ) as response:
            result = await response.json()
            print(f"✅ Upload response: {json.dumps(result, indent=2)}")
            assert response.status == 200
            assert result['status'] == 'queued'
            return result['job_id']

async def test_job_status(job_id: str):
    """Test job status checking"""
    print(f"\n🧪 Testing Job Status for {job_id}...")
    
    async with aiohttp.ClientSession() as session:
        # Poll job status
        max_attempts = 10
        for i in range(max_attempts):
            async with session.get(
                f"{API_BASE_URL}/api/hologram/job/{job_id}?userId={TEST_USER}"
            ) as response:
                result = await response.json()
                print(f"Attempt {i+1}: Status = {result['status']}")
                
                if result['status'] in ['completed', 'failed']:
                    print(f"✅ Final status: {json.dumps(result, indent=2)}")
                    assert result['status'] == 'completed'
                    return result
                
                await asyncio.sleep(1)
        
        raise AssertionError("Job did not complete in time")

async def test_list_memories():
    """Test listing holographic memories"""
    print("\n🧪 Testing List Memories...")
    
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{API_BASE_URL}/api/hologram/memories?userId={TEST_USER}"
        ) as response:
            result = await response.json()
            print(f"✅ Memories: {json.dumps(result, indent=2)}")
            assert response.status == 200
            assert 'memories' in result
            return result['memories']

async def test_get_memory(memory_id: str):
    """Test getting specific memory details"""
    print(f"\n🧪 Testing Get Memory {memory_id}...")
    
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{API_BASE_URL}/api/hologram/memory/{memory_id}?userId={TEST_USER}"
        ) as response:
            result = await response.json()
            print(f"✅ Memory details:")
            print(f"  - Morphons: {len(result['morphons'])}")
            print(f"  - Strands: {len(result['strands'])}")
            
            # Check morphon types
            modalities = {}
            for morphon in result['morphons']:
                modality = morphon['modality']
                modalities[modality] = modalities.get(modality, 0) + 1
            print(f"  - Modalities: {modalities}")
            
            assert len(result['morphons']) > 0
            return result

async def test_get_morphons():
    """Test getting morphons for visualization"""
    print("\n🧪 Testing Get Morphons...")
    
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{API_BASE_URL}/api/hologram/morphons?userId={TEST_USER}&limit=10"
        ) as response:
            result = await response.json()
            print(f"✅ Retrieved {len(result['morphons'])} morphons")
            
            # Display sample morphon
            if result['morphons']:
                morphon = result['morphons'][0]
                print(f"Sample morphon:")
                print(f"  - ID: {morphon['id']}")
                print(f"  - Modality: {morphon['modality']}")
                print(f"  - Salience: {morphon['salience']}")
                print(f"  - Connections: {len(morphon['connections'])}")
            
            assert response.status == 200
            return result['morphons']

async def test_visualization_graph():
    """Test getting visualization graph data"""
    print("\n🧪 Testing Visualization Graph...")
    
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{API_BASE_URL}/api/hologram/graph?userId={TEST_USER}"
        ) as response:
            result = await response.json()
            print(f"✅ Graph data:")
            print(f"  - Nodes: {result['stats']['total_nodes']}")
            print(f"  - Edges: {result['stats']['total_edges']}")
            print(f"  - Node types: {result['stats']['node_types']}")
            
            assert response.status == 200
            assert len(result['nodes']) > 0
            return result

async def test_cross_modal_connections(memory_id: str):
    """Test cross-modal connections in a memory"""
    print(f"\n🧪 Testing Cross-Modal Connections for {memory_id}...")
    
    async with aiohttp.ClientSession() as session:
        # Get memory details
        async with session.get(
            f"{API_BASE_URL}/api/hologram/memory/{memory_id}?userId={TEST_USER}"
        ) as response:
            memory = await response.json()
        
        # Analyze strands
        strand_types = {}
        cross_modal = 0
        
        for strand in memory['strands']:
            strand_type = strand['strand_type']
            strand_types[strand_type] = strand_types.get(strand_type, 0) + 1
            
            # Check if cross-modal
            source_morphon = next((m for m in memory['morphons'] 
                                 if m['id'] == strand['source_morphon_id']), None)
            target_morphon = next((m for m in memory['morphons'] 
                                 if m['id'] == strand['target_morphon_id']), None)
            
            if (source_morphon and target_morphon and 
                source_morphon['modality'] != target_morphon['modality']):
                cross_modal += 1
        
        print(f"✅ Strand analysis:")
        print(f"  - Strand types: {strand_types}")
        print(f"  - Cross-modal connections: {cross_modal}")
        
        # Should have at least some connections
        assert len(memory['strands']) > 0

async def test_concept_linking():
    """Test linking morphons to concepts in mesh"""
    print("\n🧪 Testing Concept Linking...")
    
    async with aiohttp.ClientSession() as session:
        # First add a concept to the mesh
        concept_data = {
            "name": "red_square",
            "description": "A red colored square shape",
            "category": "visual"
        }
        
        async with session.post(
            f"{API_BASE_URL}/api/mesh/concepts?userId={TEST_USER}",
            json=concept_data
        ) as response:
            concept_result = await response.json()
            print(f"✅ Created concept: {concept_result['concept_id']}")
        
        # Upload an image that should link to this concept
        image_data = create_test_image()
        data = aiohttp.FormData()
        data.add_field('file',
                      image_data,
                      filename='red_square.png',
                      content_type='image/png')
        
        async with session.post(
            f"{API_BASE_URL}/api/hologram/upload?userId={TEST_USER}",
            data=data
        ) as response:
            upload_result = await response.json()
            job_id = upload_result['job_id']
        
        # Wait for processing
        await asyncio.sleep(3)
        
        # Check if concepts were linked
        async with session.get(
            f"{API_BASE_URL}/api/hologram/graph?userId={TEST_USER}"
        ) as response:
            graph = await response.json()
            
            # Look for concept nodes
            concept_nodes = [n for n in graph['nodes'] if n['type'] == 'concept']
            print(f"✅ Found {len(concept_nodes)} concept nodes in graph")
            
            # Should have at least our created concept
            assert len(concept_nodes) > 0

async def run_all_tests():
    """Run all Phase 4 tests"""
    print("🚀 Starting Phase 4 Holographic Memory Tests...")
    print("=" * 60)
    
    try:
        # Basic upload and processing
        job_id = await test_image_upload()
        await asyncio.sleep(2)  # Let it process
        
        job_result = await test_job_status(job_id)
        memory_id = job_result['result']['memory_id']
        
        # Memory operations
        memories = await test_list_memories()
        assert len(memories) > 0
        
        memory_details = await test_get_memory(memory_id)
        
        # Visualization
        morphons = await test_get_morphons()
        graph = await test_visualization_graph()
        
        # Analysis
        await test_cross_modal_connections(memory_id)
        await test_concept_linking()
        
        print("\n✅ All Phase 4 tests passed!")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ Test assertion failed: {e}")
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run the tests
    asyncio.run(run_all_tests())
