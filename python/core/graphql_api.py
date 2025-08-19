#!/usr/bin/env python3
"""
GraphQL API Layer for TORI/KHA
Provides GraphQL interface for complex queries without containers or databases
File-based storage and MCP server compatible
"""

import asyncio
import json
import time
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass
import uuid

# GraphQL imports
try:
    import strawberry
    from strawberry.asgi import GraphQL
    from strawberry.types import Info
    from strawberry.schema.config import StrawberryConfig
    from strawberry.extensions import Extension
    import uvicorn
    GRAPHQL_AVAILABLE = True
except ImportError:
    GRAPHQL_AVAILABLE = False
    logging.warning("Strawberry GraphQL not available - GraphQL API disabled")

# Import TORI components
from python.core.tori_production import TORIProductionSystem, TORIProductionConfig
from python.core.metacognitive_adapters import AdapterMode
from python.core.opentelemetry_tracing import get_tracer, TracingDecorators

logger = logging.getLogger(__name__)
tracer = get_tracer()
trace_decorator = TracingDecorators(tracer).trace

# ========== GraphQL Types ==========

@strawberry.type
class CognitiveState:
    """Cognitive state information"""
    stability_score: float
    coherence: float
    contradiction_level: float
    phase: str
    confidence: float

@strawberry.type
class EigenvalueMetrics:
    """Eigenvalue analysis metrics"""
    max_eigenvalue: float
    spectral_radius: float
    is_stable: bool
    stability_margin: float
    condition_number: float

@strawberry.type
class ChaosMetrics:
    """Chaos computation metrics"""
    events_total: int
    energy_consumed: int
    energy_generated: int
    efficiency_ratio: float
    active_tasks: int

@strawberry.type
class SafetyMetrics:
    """Safety system metrics"""
    current_level: str
    monitoring_active: bool
    interventions_total: int
    last_checkpoint: str

@strawberry.type
class SystemStatus:
    """Complete system status"""
    operational: bool
    chaos_enabled: bool
    adapter_mode: str
    uptime_seconds: float
    cognitive_state: CognitiveState
    eigenvalue_metrics: EigenvalueMetrics
    chaos_metrics: ChaosMetrics
    safety_metrics: SafetyMetrics

@strawberry.type
class QueryResult:
    """Query processing result"""
    query_id: str
    response: str
    processing_time: float
    chaos_enabled: bool
    safety_level: str
    reasoning_paths: List[str]
    confidence: float

@strawberry.type
class Memory:
    """Memory entry"""
    id: str
    type: str
    content: str
    importance: float
    timestamp: float
    access_count: int
    tags: List[str]

@strawberry.type
class MemorySearchResult:
    """Memory search results"""
    memories: List[Memory]
    total_count: int
    search_time: float

@strawberry.type
class TraceSpan:
    """Trace span information"""
    span_id: str
    name: str
    duration_ms: Optional[float]
    status: str
    attributes: strawberry.scalars.JSON

@strawberry.type
class Trace:
    """Complete trace information"""
    trace_id: str
    spans: List[TraceSpan]
    total_duration_ms: float
    error_count: int

@strawberry.type
class EfficiencyReport:
    """Chaos efficiency analysis"""
    average_gain: float
    max_gain: float
    min_gain: float
    samples: int
    trend: str  # "improving", "stable", "declining"

@strawberry.type
class Checkpoint:
    """Safety checkpoint"""
    checkpoint_id: str
    label: str
    timestamp: str
    fidelity: float
    size_mb: float

# ========== GraphQL Inputs ==========

@strawberry.input
class QueryInput:
    """Input for query processing"""
    query: str
    enable_chaos: bool = False
    context: Optional[strawberry.scalars.JSON] = None
    max_reasoning_paths: int = 3

@strawberry.input
class MemoryStoreInput:
    """Input for storing memory"""
    content: str
    memory_type: str = "semantic"
    importance: float = 1.0
    tags: List[str] = strawberry.field(default_factory=list)
    metadata: Optional[strawberry.scalars.JSON] = None

@strawberry.input
class MemorySearchInput:
    """Input for searching memories"""
    query: Optional[str] = None
    memory_type: Optional[str] = None
    tags: Optional[List[str]] = None
    min_importance: float = 0.0
    max_results: int = 100

@strawberry.input
class ChaosTaskInput:
    """Input for chaos task"""
    mode: str  # "dark_soliton", "attractor_hop", "phase_explosion", "hybrid"
    input_data: List[float]
    parameters: Optional[strawberry.scalars.JSON] = None
    energy_budget: int = 100

# ========== Subscriptions Types ==========

@strawberry.type
class StatusUpdate:
    """Real-time status update"""
    timestamp: str
    chaos_events: int
    safety_level: str
    eigenvalue_max: float
    active_queries: int

@strawberry.type
class MetricUpdate:
    """Real-time metric update"""
    timestamp: str
    metric_name: str
    value: float
    labels: strawberry.scalars.JSON

# ========== Context and Extensions ==========

class TORIContext:
    """Context for GraphQL resolvers"""
    def __init__(self):
        self.tori_system: Optional[TORIProductionSystem] = None
        self.request_id: str = str(uuid.uuid4())
        self.start_time: float = time.time()

class TracingExtension(Extension):
    """OpenTelemetry tracing for GraphQL"""
    
    async def on_request_start(self):
        self.span = tracer.span("graphql.request", {
            "request.id": self.execution_context.context.request_id
        }).__enter__()
    
    async def on_request_end(self):
        if hasattr(self, 'span'):
            self.span.__exit__(None, None, None)
    
    def on_validation_start(self):
        tracer.add_event("validation.start")
    
    def on_validation_end(self):
        tracer.add_event("validation.end")
    
    def on_parse_start(self):
        tracer.add_event("parse.start")
    
    def on_parse_end(self):
        tracer.add_event("parse.end")

# ========== Query Root ==========

@strawberry.type
class Query:
    """GraphQL Query root"""
    
    @strawberry.field
    @trace_decorator("graphql.status")
    async def status(self, info: Info[TORIContext, None]) -> SystemStatus:
        """Get current system status"""
        tori = info.context.tori_system
        status = tori.get_status()
        
        # Get current cognitive state
        cognitive_state = CognitiveState(
            stability_score=tori.state_manager.get_state_stability(),
            coherence=tori.state_manager.get_coherence(),
            contradiction_level=0.0,  # Would need to implement
            phase=tori.state_manager.current_phase,
            confidence=tori.state_manager.get_confidence()
        )
        
        # Get eigenvalue metrics
        eigen_metrics = tori.eigen_sentry.get_status()
        eigenvalue_metrics = EigenvalueMetrics(
            max_eigenvalue=eigen_metrics.get('max_eigenvalue', 0.0),
            spectral_radius=eigen_metrics.get('spectral_radius', 0.0),
            is_stable=eigen_metrics.get('is_stable', True),
            stability_margin=eigen_metrics.get('stability_margin', 1.0),
            condition_number=eigen_metrics.get('condition_number', 1.0)
        )
        
        # Get chaos metrics
        ccl_status = status.get('ccl', {})
        chaos_metrics = ChaosMetrics(
            events_total=status['statistics'].get('chaos_events', 0),
            energy_consumed=ccl_status.get('energy_consumed', 0),
            energy_generated=ccl_status.get('energy_generated', 0),
            efficiency_ratio=ccl_status.get('efficiency_ratio', 1.0),
            active_tasks=ccl_status.get('active_tasks', 0)
        )
        
        # Get safety metrics
        safety_status = status.get('safety', {})
        safety_metrics = SafetyMetrics(
            current_level=safety_status.get('current_safety_level', 'unknown'),
            monitoring_active=safety_status.get('monitoring_active', False),
            interventions_total=status['statistics'].get('safety_interventions', 0),
            last_checkpoint=safety_status.get('last_checkpoint', 'none')
        )
        
        # Calculate uptime
        from python.core.prometheus_metrics import get_metrics_collector
        metrics = get_metrics_collector()
        uptime = time.time() - metrics.start_time
        
        return SystemStatus(
            operational=status['operational'],
            chaos_enabled=status['chaos_enabled'],
            adapter_mode=status['adapter_mode'],
            uptime_seconds=uptime,
            cognitive_state=cognitive_state,
            eigenvalue_metrics=eigenvalue_metrics,
            chaos_metrics=chaos_metrics,
            safety_metrics=safety_metrics
        )
    
    @strawberry.field
    @trace_decorator("graphql.process_query")
    async def process_query(self, info: Info[TORIContext, None], input: QueryInput) -> QueryResult:
        """Process a cognitive query"""
        tori = info.context.tori_system
        
        # Generate query ID
        query_id = str(uuid.uuid4())
        tracer.set_attribute("query.id", query_id)
        
        # Build context
        context = input.context or {}
        context['enable_chaos'] = input.enable_chaos
        context['query_id'] = query_id
        context['max_reasoning_paths'] = input.max_reasoning_paths
        
        start_time = time.time()
        
        # Process query
        result = await tori.process_query(input.query, context)
        
        processing_time = time.time() - start_time
        
        # Extract reasoning paths
        reasoning_paths = [
            f"{' -> '.join(n['name'] for n in path['nodes'])} (score: {path['score']:.2f})"
            for path in result['reasoning_paths'][:input.max_reasoning_paths]
        ]
        
        return QueryResult(
            query_id=query_id,
            response=result['response'],
            processing_time=processing_time,
            chaos_enabled=result['metadata'].get('chaos_enabled', False),
            safety_level=result['metadata'].get('safety_level', 'unknown'),
            reasoning_paths=reasoning_paths,
            confidence=result['metadata'].get('confidence', 0.0)
        )
    
    @strawberry.field
    async def search_memories(self, info: Info[TORIContext, None], input: MemorySearchInput) -> MemorySearchResult:
        """Search through memories"""
        tori = info.context.tori_system
        vault = tori.metacognitive_system.memory_vault
        
        from python.core.memory_vault import MemoryType
        
        start_time = time.time()
        
        memories = await vault.search(
            query=input.query,
            memory_type=MemoryType(input.memory_type) if input.memory_type else None,
            tags=input.tags,
            min_importance=input.min_importance,
            max_results=input.max_results
        )
        
        search_time = time.time() - start_time
        
        # Convert to GraphQL type
        memory_results = [
            Memory(
                id=m.id,
                type=m.type.value,
                content=str(m.content)[:1000],  # Truncate for GraphQL
                importance=m.importance,
                timestamp=m.timestamp,
                access_count=m.access_count,
                tags=m.metadata.get('tags', [])
            )
            for m in memories
        ]
        
        return MemorySearchResult(
            memories=memory_results,
            total_count=len(memory_results),
            search_time=search_time
        )
    
    @strawberry.field
    async def get_memory(self, info: Info[TORIContext, None], memory_id: str) -> Optional[Memory]:
        """Get specific memory by ID"""
        tori = info.context.tori_system
        vault = tori.metacognitive_system.memory_vault
        
        memory = await vault.retrieve(memory_id)
        
        if memory is None:
            return None
        
        return Memory(
            id=memory.id,
            type=memory.type.value,
            content=str(memory.content),
            importance=memory.importance,
            timestamp=memory.timestamp,
            access_count=memory.access_count,
            tags=memory.metadata.get('tags', [])
        )
    
    @strawberry.field
    async def efficiency_report(self, info: Info[TORIContext, None]) -> EfficiencyReport:
        """Get chaos efficiency report"""
        tori = info.context.tori_system
        report = tori.get_efficiency_report()
        
        # Determine trend
        if report['samples'] < 10:
            trend = "stable"
        else:
            # Would need to implement trend analysis
            trend = "stable"
        
        return EfficiencyReport(
            average_gain=report.get('average_gain', 1.0),
            max_gain=report.get('max_gain', 1.0),
            min_gain=report.get('min_gain', 1.0),
            samples=report.get('samples', 0),
            trend=trend
        )
    
    @strawberry.field
    async def list_checkpoints(self, info: Info[TORIContext, None]) -> List[Checkpoint]:
        """List available checkpoints"""
        tori = info.context.tori_system
        checkpoints = tori.safety_system.list_checkpoints()
        
        return [
            Checkpoint(
                checkpoint_id=cp.checkpoint_id,
                label=cp.label,
                timestamp=cp.timestamp.isoformat(),
                fidelity=cp.metrics.fidelity,
                size_mb=cp.metrics.state_size / (1024 * 1024)
            )
            for cp in checkpoints
        ]
    
    @strawberry.field
    async def recent_traces(self, info: Info[TORIContext, None], limit: int = 10) -> List[Trace]:
        """Get recent traces"""
        from python.core.opentelemetry_tracing import TraceAnalyzer
        
        analyzer = TraceAnalyzer()
        all_traces = analyzer.load_traces()
        
        # Group by trace ID
        traces_by_id = {}
        for span in all_traces:
            if span.trace_id not in traces_by_id:
                traces_by_id[span.trace_id] = []
            traces_by_id[span.trace_id].append(span)
        
        # Convert to GraphQL type
        traces = []
        for trace_id, spans in list(traces_by_id.items())[:limit]:
            trace_spans = [
                TraceSpan(
                    span_id=s.span_id,
                    name=s.name,
                    duration_ms=s.duration_ms(),
                    status=s.status.value if hasattr(s.status, 'value') else str(s.status),
                    attributes=s.attributes
                )
                for s in spans
            ]
            
            total_duration = max(s.duration_ms() or 0 for s in spans)
            error_count = sum(1 for s in spans if s.status.name == "ERROR")
            
            traces.append(Trace(
                trace_id=trace_id,
                spans=trace_spans,
                total_duration_ms=total_duration,
                error_count=error_count
            ))
        
        return traces

# ========== Mutation Root ==========

@strawberry.type
class Mutation:
    """GraphQL Mutation root"""
    
    @strawberry.mutation
    async def store_memory(self, info: Info[TORIContext, None], input: MemoryStoreInput) -> Memory:
        """Store a new memory"""
        tori = info.context.tori_system
        vault = tori.metacognitive_system.memory_vault
        
        from python.core.memory_vault import MemoryType
        
        memory_id = await vault.store(
            content=input.content,
            memory_type=MemoryType(input.memory_type),
            metadata=input.metadata or {},
            importance=input.importance,
            tags=input.tags
        )
        
        # Retrieve to return full object
        memory = await vault.retrieve(memory_id)
        
        return Memory(
            id=memory.id,
            type=memory.type.value,
            content=str(memory.content),
            importance=memory.importance,
            timestamp=memory.timestamp,
            access_count=memory.access_count,
            tags=input.tags
        )
    
    @strawberry.mutation
    async def set_chaos_mode(self, info: Info[TORIContext, None], mode: str) -> SystemStatus:
        """Set chaos adapter mode"""
        tori = info.context.tori_system
        
        try:
            adapter_mode = AdapterMode[mode.upper()]
            tori.set_chaos_mode(adapter_mode)
        except KeyError:
            raise ValueError(f"Invalid mode. Valid modes: {[m.value for m in AdapterMode]}")
        
        # Return updated status
        return await Query().status(info)
    
    @strawberry.mutation
    async def submit_chaos_task(self, info: Info[TORIContext, None], input: ChaosTaskInput) -> str:
        """Submit chaos computation task"""
        tori = info.context.tori_system
        
        from python.core.chaos_control_layer import ChaosTask, ChaosMode
        import numpy as np
        
        # Create chaos task
        task = ChaosTask(
            task_id=str(uuid.uuid4()),
            mode=ChaosMode[input.mode.upper()],
            input_data=np.array(input.input_data),
            parameters=input.parameters or {},
            energy_budget=input.energy_budget
        )
        
        # Submit to CCL
        task_id = await tori.ccl.submit_task(task)
        
        return task_id
    
    @strawberry.mutation
    async def create_checkpoint(self, info: Info[TORIContext, None], label: str) -> Checkpoint:
        """Create safety checkpoint"""
        tori = info.context.tori_system
        
        checkpoint_id = await tori.create_checkpoint(label)
        
        # Get checkpoint info
        checkpoint = tori.safety_system.get_checkpoint(checkpoint_id)
        
        return Checkpoint(
            checkpoint_id=checkpoint.checkpoint_id,
            label=checkpoint.label,
            timestamp=checkpoint.timestamp.isoformat(),
            fidelity=checkpoint.metrics.fidelity,
            size_mb=checkpoint.metrics.state_size / (1024 * 1024)
        )
    
    @strawberry.mutation
    async def rollback_to_checkpoint(self, info: Info[TORIContext, None], checkpoint_id: str) -> bool:
        """Rollback to checkpoint"""
        tori = info.context.tori_system
        return await tori.rollback(checkpoint_id)

# ========== Subscription Root ==========

@strawberry.type
class Subscription:
    """GraphQL Subscription root"""
    
    @strawberry.subscription
    async def status_updates(self, info: Info[TORIContext, None]) -> AsyncIterator[StatusUpdate]:
        """Subscribe to real-time status updates"""
        tori = info.context.tori_system
        
        while True:
            # Get current status
            status = tori.get_status()
            eigen_status = tori.eigen_sentry.get_status()
            
            update = StatusUpdate(
                timestamp=datetime.now(timezone.utc).isoformat(),
                chaos_events=status['statistics']['chaos_events'],
                safety_level=status['safety']['current_safety_level'],
                eigenvalue_max=eigen_status.get('max_eigenvalue', 0.0),
                active_queries=status['statistics'].get('queries_processed', 0)
            )
            
            yield update
            
            # Wait before next update
            await asyncio.sleep(1.0)
    
    @strawberry.subscription
    async def metric_updates(self, info: Info[TORIContext, None], 
                           metric_names: List[str]) -> AsyncIterator[MetricUpdate]:
        """Subscribe to specific metric updates"""
        from python.core.prometheus_metrics import get_metrics_collector
        
        collector = get_metrics_collector()
        
        while True:
            # Get current metrics
            summary = collector.get_metrics_summary()
            
            for metric_name in metric_names:
                if metric_name in summary.get('gauges', {}):
                    yield MetricUpdate(
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        metric_name=metric_name,
                        value=summary['gauges'][metric_name],
                        labels={}
                    )
                elif metric_name in summary.get('counters', {}):
                    yield MetricUpdate(
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        metric_name=metric_name,
                        value=summary['counters'][metric_name],
                        labels={}
                    )
            
            await asyncio.sleep(0.5)

# ========== GraphQL Schema ==========

def create_schema(tori_system: TORIProductionSystem) -> strawberry.Schema:
    """Create GraphQL schema with TORI system"""
    
    # Create context getter
    async def get_context() -> TORIContext:
        context = TORIContext()
        context.tori_system = tori_system
        return context
    
    # Create schema
    schema = strawberry.Schema(
        query=Query,
        mutation=Mutation,
        subscription=Subscription,
        config=StrawberryConfig(auto_camel_case=True),
        extensions=[TracingExtension] if GRAPHQL_AVAILABLE else []
    )
    
    return schema

# ========== Server Runner ==========

async def run_graphql_server(port: int = 8080, tori_system: Optional[TORIProductionSystem] = None):
    """Run GraphQL server"""
    if not GRAPHQL_AVAILABLE:
        logger.error("GraphQL dependencies not available")
        return
    
    # Initialize TORI if not provided
    if tori_system is None:
        config = TORIProductionConfig()
        tori_system = TORIProductionSystem(config)
        await tori_system.start()
    
    # Create schema
    schema = create_schema(tori_system)
    
    # Create GraphQL app
    graphql_app = GraphQL(schema)
    
    # Run with uvicorn
    config = uvicorn.Config(
        graphql_app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
    
    server = uvicorn.Server(config)
    await server.serve()

# Example usage
if __name__ == "__main__":
    import asyncio
    
    # Example GraphQL queries
    example_queries = """
    # Get system status
    query SystemStatus {
        status {
            operational
            chaosEnabled
            adapterMode
            uptimeSeconds
            cognitiveState {
                stabilityScore
                coherence
                phase
            }
            chaosMetrics {
                eventsTotal
                efficiencyRatio
            }
        }
    }
    
    # Process a query
    mutation ProcessQuery {
        processQuery(input: {
            query: "What is consciousness?"
            enableChaos: true
            maxReasoningPaths: 5
        }) {
            queryId
            response
            processingTime
            chaosEnabled
            reasoningPaths
        }
    }
    
    # Search memories
    query SearchMemories {
        searchMemories(input: {
            query: "consciousness"
            memoryType: "semantic"
            maxResults: 10
        }) {
            memories {
                id
                content
                importance
                tags
            }
            totalCount
            searchTime
        }
    }
    
    # Subscribe to updates
    subscription StatusUpdates {
        statusUpdates {
            timestamp
            chaosEvents
            safetyLevel
            eigenvalueMax
        }
    }
    """
    
    print("Starting TORI GraphQL Server...")
    print(f"\nExample queries:\n{example_queries}")
    print("\nGraphQL endpoint will be available at http://localhost:8080/graphql")
    
    asyncio.run(run_graphql_server())
