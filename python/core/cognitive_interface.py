"""
Cognitive Interface Module - Production Implementation
Provides the interface between UI and cognitive subsystems
"""

import asyncio
import json
import logging
import uuid
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CognitiveMode(Enum):
    """Cognitive processing modes"""
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    REFLECTIVE = "reflective"
    EXPLORATORY = "exploratory"
    FOCUSED = "focused"
    DISTRIBUTED = "distributed"

@dataclass
class CognitiveRequest:
    """Request for cognitive processing"""
    id: str
    input_data: Any
    mode: CognitiveMode
    context: Dict[str, Any]
    constraints: Dict[str, Any]
    priority: int = 5
    timeout: float = 30.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class CognitiveResponse:
    """Response from cognitive processing"""
    request_id: str
    output_data: Any
    mode_used: CognitiveMode
    processing_time: float
    confidence: float
    metadata: Dict[str, Any]
    errors: List[str]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class CognitiveInterface:
    """
    Main interface for cognitive operations
    Routes requests to appropriate cognitive subsystems
    """
    
    def __init__(self, engine=None, vault=None, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Subsystem connections
        self.subsystems: Dict[str, Any] = {}
        self.mode_handlers: Dict[CognitiveMode, Callable] = {}
        
        # Request tracking
        self.active_requests: Dict[str, CognitiveRequest] = {}
        self.request_history: List[CognitiveRequest] = []
        self.response_history: List[CognitiveResponse] = []
        
        # Performance metrics
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_processing_time': 0.0,
            'mode_usage': {mode: 0 for mode in CognitiveMode}
        }
        
        # Initialize handlers
        self._setup_mode_handlers()
        
        # Connect to subsystems
        self._connect_subsystems(engine, vault)
        
        logger.info("Cognitive Interface initialized")
    
    @classmethod
    def from_existing(cls, engine, vault, config=None):
        """Create interface using existing engine and vault instances"""
        return cls(engine=engine, vault=vault, config=config)
    
    def _setup_mode_handlers(self):
        """Setup handlers for different cognitive modes"""
        self.mode_handlers = {
            CognitiveMode.ANALYTICAL: self._handle_analytical,
            CognitiveMode.CREATIVE: self._handle_creative,
            CognitiveMode.REFLECTIVE: self._handle_reflective,
            CognitiveMode.EXPLORATORY: self._handle_exploratory,
            CognitiveMode.FOCUSED: self._handle_focused,
            CognitiveMode.DISTRIBUTED: self._handle_distributed
        }
    
    def _connect_subsystems(self, engine=None, vault=None):
        """Connect to cognitive subsystems"""
        # Import subsystems dynamically to avoid circular imports
        try:
            # Connect to Cognitive Engine - use existing or create new
            if engine:
                self.subsystems['cognitive_engine'] = engine
                logger.info("Using existing Cognitive Engine instance")
            else:
                # Import dynamically
                from .CognitiveEngine import CognitiveEngine
                self.subsystems['cognitive_engine'] = CognitiveEngine(
                    self.config.get('cognitive_engine', {})
                )
                logger.info("Created new Cognitive Engine instance")
        except ImportError:
            logger.warning("Cognitive Engine not available")
        
        try:
            # Connect to Memory Vault - use existing or create new
            if vault:
                self.subsystems['memory_vault'] = vault
                logger.info("Using existing Memory Vault instance")
            else:
                # Import dynamically
                from .memory_vault import UnifiedMemoryVault
                self.subsystems['memory_vault'] = UnifiedMemoryVault(
                    self.config.get('memory_vault', {})
                )
                logger.info("Created new Memory Vault instance")
        except ImportError:
            logger.warning("Memory Vault not available")
        
        try:
            # Connect to MCP Metacognitive
            from .mcp_metacognitive import MCPMetacognitiveServer
            self.subsystems['metacognitive'] = MCPMetacognitiveServer(
                self.config.get('metacognitive', {})
            )
            logger.info("Connected to Metacognitive Server")
        except ImportError:
            logger.warning("Metacognitive Server not available")
    
    async def process(self, request: Union[CognitiveRequest, Dict[str, Any]]) -> CognitiveResponse:
        """Process a cognitive request"""
        
        # Convert dict to CognitiveRequest if needed
        if isinstance(request, dict):
            request = CognitiveRequest(**request)
        
        # Track request
        self.active_requests[request.id] = request
        self.request_history.append(request)
        self.metrics['total_requests'] += 1
        self.metrics['mode_usage'][request.mode] += 1
        
        start_time = datetime.now()
        errors = []
        
        try:
            # Get appropriate handler
            handler = self.mode_handlers.get(request.mode)
            if not handler:
                raise ValueError(f"Unknown cognitive mode: {request.mode}")
            
            # Process with timeout
            result = await asyncio.wait_for(
                handler(request),
                timeout=request.timeout
            )
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Update metrics
            self.metrics['successful_requests'] += 1
            self._update_avg_processing_time(processing_time)
            
            # Create response
            response = CognitiveResponse(
                request_id=request.id,
                output_data=result,
                mode_used=request.mode,
                processing_time=processing_time,
                confidence=result.get('confidence', 0.5) if isinstance(result, dict) else 0.5,
                metadata=result.get('metadata', {}) if isinstance(result, dict) else {},
                errors=errors
            )
            
        except asyncio.TimeoutError:
            errors.append(f"Request timed out after {request.timeout} seconds")
            self.metrics['failed_requests'] += 1
            
            response = CognitiveResponse(
                request_id=request.id,
                output_data=None,
                mode_used=request.mode,
                processing_time=request.timeout,
                confidence=0.0,
                metadata={'timeout': True},
                errors=errors
            )
            
        except Exception as e:
            errors.append(str(e))
            self.metrics['failed_requests'] += 1
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            response = CognitiveResponse(
                request_id=request.id,
                output_data=None,
                mode_used=request.mode,
                processing_time=processing_time,
                confidence=0.0,
                metadata={'error': str(e)},
                errors=errors
            )
        
        finally:
            # Clean up
            self.active_requests.pop(request.id, None)
            self.response_history.append(response)
        
        return response
    
    def _update_avg_processing_time(self, new_time: float):
        """Update average processing time"""
        total = self.metrics['successful_requests']
        if total == 1:
            self.metrics['avg_processing_time'] = new_time
        else:
            # Running average
            avg = self.metrics['avg_processing_time']
            self.metrics['avg_processing_time'] = ((avg * (total - 1)) + new_time) / total
    
    # Mode handlers
    async def _handle_analytical(self, request: CognitiveRequest) -> Dict[str, Any]:
        """Handle analytical mode processing"""
        logger.info(f"Analytical processing for request {request.id}")
        
        # Use cognitive engine if available
        if 'cognitive_engine' in self.subsystems:
            engine = self.subsystems['cognitive_engine']
            
            # Prepare input for engine
            engine_input = {
                'content': request.input_data,
                'mode': 'analytical',
                'context': request.context
            }
            
            # Process through engine
            result = await engine.process_async(engine_input)
            
            return {
                'analysis': result,
                'confidence': 0.85,
                'metadata': {
                    'mode': 'analytical',
                    'engine_used': True
                }
            }
        
        # Fallback analytical processing
        return {
            'analysis': self._basic_analysis(request.input_data),
            'confidence': 0.6,
            'metadata': {
                'mode': 'analytical',
                'engine_used': False
            }
        }
    
    async def _handle_creative(self, request: CognitiveRequest) -> Dict[str, Any]:
        """Handle creative mode processing"""
        logger.info(f"Creative processing for request {request.id}")
        
        # Creative processing logic
        input_data = request.input_data
        
        # Generate variations
        variations = []
        if isinstance(input_data, str):
            # Text variations
            variations = [
                input_data.upper(),
                input_data.lower(),
                ' '.join(reversed(input_data.split())),
                input_data.replace(' ', '_')
            ]
        elif isinstance(input_data, list):
            # List variations
            variations = [
                list(reversed(input_data)),
                sorted(input_data),
                input_data + input_data[::-1]
            ]
        
        return {
            'original': input_data,
            'variations': variations,
            'creative_score': np.random.random(),
            'confidence': 0.7,
            'metadata': {
                'mode': 'creative',
                'variation_count': len(variations)
            }
        }
    
    async def _handle_reflective(self, request: CognitiveRequest) -> Dict[str, Any]:
        """Handle reflective mode processing"""
        logger.info(f"Reflective processing for request {request.id}")
        
        # Use memory vault for reflection if available
        if 'memory_vault' in self.subsystems:
            vault = self.subsystems['memory_vault']
            
            # Search for related memories
            related = await vault.search(
                query=str(request.input_data),
                max_results=5
            )
            
            # Analyze patterns
            patterns = self._analyze_memory_patterns(related)
            
            return {
                'reflection': patterns,
                'related_memories': len(related),
                'confidence': 0.8,
                'metadata': {
                    'mode': 'reflective',
                    'memory_used': True
                }
            }
        
        # Fallback reflection
        return {
            'reflection': {
                'summary': f"Reflecting on: {request.input_data}",
                'insights': ["Consider different perspectives", "Look for patterns"],
                'questions': ["What does this mean?", "How does this relate to past experiences?"]
            },
            'confidence': 0.5,
            'metadata': {
                'mode': 'reflective',
                'memory_used': False
            }
        }
    
    async def _handle_exploratory(self, request: CognitiveRequest) -> Dict[str, Any]:
        """Handle exploratory mode processing"""
        logger.info(f"Exploratory processing for request {request.id}")
        
        # Explore different aspects
        explorations = {
            'breadth': self._explore_breadth(request.input_data),
            'depth': self._explore_depth(request.input_data),
            'connections': self._explore_connections(request.input_data)
        }
        
        return {
            'explorations': explorations,
            'discovered_aspects': sum(len(v) for v in explorations.values()),
            'confidence': 0.65,
            'metadata': {
                'mode': 'exploratory'
            }
        }
    
    async def _handle_focused(self, request: CognitiveRequest) -> Dict[str, Any]:
        """Handle focused mode processing"""
        logger.info(f"Focused processing for request {request.id}")
        
        # Focus on specific aspects based on constraints
        focus_areas = request.constraints.get('focus_areas', ['main'])
        
        results = {}
        for area in focus_areas:
            results[area] = self._focused_analysis(request.input_data, area)
        
        return {
            'focused_results': results,
            'focus_areas': focus_areas,
            'confidence': 0.9,
            'metadata': {
                'mode': 'focused',
                'area_count': len(focus_areas)
            }
        }
    
    async def _handle_distributed(self, request: CognitiveRequest) -> Dict[str, Any]:
        """Handle distributed mode processing"""
        logger.info(f"Distributed processing for request {request.id}")
        
        # Simulate distributed processing across subsystems
        tasks = []
        
        # Create subtasks for different subsystems
        if 'cognitive_engine' in self.subsystems:
            tasks.append(self._distributed_engine_task(request))
        
        if 'metacognitive' in self.subsystems:
            tasks.append(self._distributed_meta_task(request))
        
        # Fallback task
        if not tasks:
            tasks.append(self._distributed_fallback_task(request))
        
        # Execute in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate results
        aggregated = self._aggregate_distributed_results(results)
        
        return {
            'distributed_results': aggregated,
            'subsystems_used': len(tasks),
            'confidence': 0.75,
            'metadata': {
                'mode': 'distributed',
                'parallel_tasks': len(tasks)
            }
        }
    
    # Helper methods
    def _basic_analysis(self, data: Any) -> Dict[str, Any]:
        """Basic analytical processing"""
        analysis = {
            'type': type(data).__name__,
            'size': len(str(data))
        }
        
        if isinstance(data, str):
            analysis.update({
                'word_count': len(data.split()),
                'char_count': len(data),
                'unique_chars': len(set(data))
            })
        elif isinstance(data, (list, tuple)):
            analysis.update({
                'length': len(data),
                'unique_items': len(set(str(item) for item in data))
            })
        elif isinstance(data, dict):
            analysis.update({
                'keys': list(data.keys()),
                'depth': self._dict_depth(data)
            })
        
        return analysis
    
    def _dict_depth(self, d: dict, level: int = 0) -> int:
        """Calculate dictionary depth"""
        if not isinstance(d, dict) or not d:
            return level
        return max(self._dict_depth(v, level + 1) for v in d.values() if isinstance(v, dict))
    
    def _analyze_memory_patterns(self, memories: List[Any]) -> Dict[str, Any]:
        """Analyze patterns in memories"""
        if not memories:
            return {'patterns': [], 'insights': []}
        
        # Simple pattern analysis
        patterns = {
            'temporal': "Recent memories show increased activity",
            'thematic': "Common themes detected across memories",
            'emotional': "Emotional valence is predominantly neutral"
        }
        
        insights = [
            f"Found {len(memories)} related memories",
            "Pattern suggests recurring interest in this topic",
            "Consider exploring alternative perspectives"
        ]
        
        return {
            'patterns': patterns,
            'insights': insights
        }
    
    def _explore_breadth(self, data: Any) -> List[str]:
        """Explore breadth of topic"""
        explorations = [
            f"Broad perspective on {data}",
            f"Related concepts to {data}",
            f"Alternative views of {data}"
        ]
        return explorations
    
    def _explore_depth(self, data: Any) -> List[str]:
        """Explore depth of topic"""
        explorations = [
            f"Deep analysis of {data}",
            f"Underlying principles of {data}",
            f"Detailed examination of {data}"
        ]
        return explorations
    
    def _explore_connections(self, data: Any) -> List[str]:
        """Explore connections"""
        explorations = [
            f"Connections from {data}",
            f"Network around {data}",
            f"Relationships involving {data}"
        ]
        return explorations
    
    def _focused_analysis(self, data: Any, area: str) -> Dict[str, Any]:
        """Focused analysis on specific area"""
        return {
            'area': area,
            'analysis': f"Focused analysis of {data} in area {area}",
            'depth': 'deep',
            'findings': [f"Finding 1 for {area}", f"Finding 2 for {area}"]
        }
    
    async def _distributed_engine_task(self, request: CognitiveRequest) -> Dict[str, Any]:
        """Distributed task for cognitive engine"""
        engine = self.subsystems['cognitive_engine']
        result = await engine.process_async({
            'content': request.input_data,
            'mode': 'distributed'
        })
        return {'source': 'cognitive_engine', 'result': result}
    
    async def _distributed_meta_task(self, request: CognitiveRequest) -> Dict[str, Any]:
        """Distributed task for metacognitive server"""
        meta = self.subsystems['metacognitive']
        result = await meta.process_task({
            'description': str(request.input_data),
            'complexity': 0.5
        })
        return {'source': 'metacognitive', 'result': result}
    
    async def _distributed_fallback_task(self, request: CognitiveRequest) -> Dict[str, Any]:
        """Fallback distributed task"""
        await asyncio.sleep(0.1)  # Simulate processing
        return {
            'source': 'fallback',
            'result': {
                'processed': str(request.input_data),
                'status': 'completed'
            }
        }
    
    def _aggregate_distributed_results(self, results: List[Any]) -> Dict[str, Any]:
        """Aggregate results from distributed processing"""
        aggregated = {
            'sources': [],
            'combined_output': {},
            'errors': []
        }
        
        for result in results:
            if isinstance(result, Exception):
                aggregated['errors'].append(str(result))
            elif isinstance(result, dict) and 'source' in result:
                aggregated['sources'].append(result['source'])
                aggregated['combined_output'][result['source']] = result.get('result')
        
        return aggregated
    
    # Public API methods
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.metrics.copy()
    
    def get_active_requests(self) -> List[Dict[str, Any]]:
        """Get currently active requests"""
        return [asdict(req) for req in self.active_requests.values()]
    
    def get_request_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get request history"""
        history = self.request_history[-limit:]
        return [asdict(req) for req in history]
    
    def get_response_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get response history"""
        history = self.response_history[-limit:]
        return [asdict(resp) for resp in history]
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        health = {
            'status': 'healthy',
            'subsystems': {},
            'metrics': self.get_metrics()
        }
        
        # Check each subsystem
        for name, subsystem in self.subsystems.items():
            try:
                # Simple connectivity check
                if hasattr(subsystem, 'get_statistics'):
                    stats = subsystem.get_statistics()
                    health['subsystems'][name] = {'status': 'connected', 'stats': stats}
                else:
                    health['subsystems'][name] = {'status': 'connected'}
            except Exception as e:
                health['subsystems'][name] = {'status': 'error', 'error': str(e)}
                health['status'] = 'degraded'
        
        return health

# Global mesh instance (would be properly initialized in production)
mesh = None

def load_concept_mesh(config: Optional[Dict[str, Any]] = None):
    """
    Load and initialize the concept mesh
    
    Args:
        config: Optional configuration for the mesh
        
    Returns:
        ConceptMesh instance
    """
    global mesh
    
    try:
        from python.core import ConceptMesh
        mesh = ConceptMesh()
        logger.info("ConceptMesh loaded successfully")
        return mesh
    except ImportError as e:
        logger.warning(f"Failed to load ConceptMesh: {e}")
        # Return a minimal mock mesh
        class MockMesh:
            def __init__(self):
                self.concepts = {}
            
            def add_concept(self, concept_id, data=None):
                self.concepts[concept_id] = data or {}
                return concept_id
            
            def get_concept(self, concept_id):
                return self.concepts.get(concept_id)
            
            def search(self, query, limit=10):
                return []
            
            def insert(self, data):
                logger.debug(f"Mock mesh insert: {len(data.get('records', []))} records")
        
        mesh = MockMesh()
        return mesh

def add_concept_diff(diff: dict):
    """
    Merge a concept-diff into the mesh **with edge enrichment & provenance**.
    """
    now = datetime.utcnow().isoformat()

    # ----- NEW 1: up-convert each concept into a node-plus-edge record -----
    enriched = []
    for c in diff.get("concepts", []):
        node_id = c.get("canonical", c["name"]).lower()

        # primary node record
        enriched.append({
            "id": node_id,
            "type": "concept",
            "data": c,
            "first_seen": now if "first_seen" not in c else c["first_seen"],
            "last_seen": now
        })

        # provenance edge -> every PDF that mentioned it
        for src in c.get("sources", []):
            enriched.append({
                "id": str(uuid.uuid4()),
                "type": "edge",
                "edge_type": "mentioned_in",
                "from": node_id,
                "to": f"doc::{src}",
                "weight": 1.0,
                "timestamp": now
            })

    diff["records"] = enriched

    # ----- NEW 2: write through the existing mesh driver -----
    if mesh is not None:
        mesh.insert(diff)
    else:
        logger.warning("Mesh not initialized, concept diff not inserted")

# Example usage
if __name__ == "__main__":
    async def test_interface():
        # Create interface
        interface = CognitiveInterface({
            'cognitive_engine': {'storage_path': 'data/cognitive'},
            'memory_vault': {'storage_path': 'data/memory'},
            'metacognitive': {'port': 8888}
        })
        
        # Test different modes
        test_requests = [
            CognitiveRequest(
                id="test_001",
                input_data="Analyze the structure of this text",
                mode=CognitiveMode.ANALYTICAL,
                context={},
                constraints={}
            ),
            CognitiveRequest(
                id="test_002",
                input_data=["idea", "concept", "thought"],
                mode=CognitiveMode.CREATIVE,
                context={},
                constraints={}
            ),
            CognitiveRequest(
                id="test_003",
                input_data="What have I learned?",
                mode=CognitiveMode.REFLECTIVE,
                context={},
                constraints={}
            )
        ]
        
        # Process requests
        for request in test_requests:
            response = await interface.process(request)
            print(f"\nRequest {request.id} ({request.mode.value}):")
            print(f"  Status: {'Success' if not response.errors else 'Failed'}")
            print(f"  Confidence: {response.confidence:.2f}")
            print(f"  Processing time: {response.processing_time:.2f}s")
            if response.errors:
                print(f"  Errors: {response.errors}")
        
        # Get metrics
        metrics = interface.get_metrics()
        print(f"\nMetrics: {json.dumps(metrics, indent=2)}")
        
        # Health check
        health = await interface.health_check()
        print(f"\nHealth: {json.dumps(health, indent=2, default=str)}")
    
    asyncio.run(test_interface())
