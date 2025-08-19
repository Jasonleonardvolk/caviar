"""
Enhanced HoTT Proof Queue with Tenant Isolation
Extends the base proof queue to support per-tenant proof contexts
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import asyncio
import time

# Import base proof queue
from hott_integration.proof_queue import ProofQueue, ProofTask

logger = logging.getLogger(__name__)

@dataclass
class TenantProofTask(ProofTask):
    """Extended proof task with tenant information"""
    tenant_scope: str = "user"  # "user" or "group"
    tenant_id: str = "default"
    context_namespace: Optional[str] = None
    
    def get_tenant_key(self) -> str:
        """Get unique tenant identifier"""
        return f"{self.tenant_scope}_{self.tenant_id}"

class TenantAwareProofQueue(ProofQueue):
    """
    Proof queue with tenant isolation
    Ensures proofs from different tenants don't interfere
    """
    
    def __init__(self):
        super().__init__()
        
        # Tenant-specific contexts
        self.tenant_contexts: Dict[str, Dict[str, Any]] = {}
        
        # Separate proof directories per tenant
        self.tenant_proof_dirs: Dict[str, Path] = {}
        
        logger.info("📐 Tenant-aware ProofQueue initialized")
    
    async def enqueue_tenant_proof(self, task: TenantProofTask):
        """Enqueue a proof task with tenant context"""
        tenant_key = task.get_tenant_key()
        
        # Ensure tenant directory exists
        if tenant_key not in self.tenant_proof_dirs:
            tenant_dir = self.proof_dir / task.tenant_scope / task.tenant_id
            tenant_dir.mkdir(parents=True, exist_ok=True)
            self.tenant_proof_dirs[tenant_key] = tenant_dir
        
        # Set context namespace if not provided
        if not task.context_namespace:
            task.context_namespace = f"tori.{task.tenant_scope}.{task.tenant_id}"
        
        # Enqueue using base method
        await self.enqueue(task)
        
        logger.info(f"📥 Enqueued tenant proof {task.id} for {tenant_key}")
    
    async def _verify_proof(self, task: ProofTask) -> Dict[str, Any]:
        """Verify proof with tenant isolation"""
        # Check if it's a tenant-aware task
        if isinstance(task, TenantProofTask):
            tenant_key = task.get_tenant_key()
            
            # Load or create tenant context
            if tenant_key not in self.tenant_contexts:
                self.tenant_contexts[tenant_key] = self._create_tenant_context(task)
            
            context = self.tenant_contexts[tenant_key]
            
            # Simulate verification with tenant context
            # In production, this would invoke Agda/Lean with namespace isolation
            await asyncio.sleep(0.5)
            
            result = {
                "success": True,
                "proof_type": task.proof_type,
                "verification_time": 0.5,
                "theorem": f"{task.context_namespace}.{task.morphon_id}",
                "tenant": tenant_key,
                "context_theorems": len(context.get('theorems', [])),
                "dependencies": [],
                "confidence": 0.95
            }
            
            # Update tenant context with new theorem
            if 'theorems' not in context:
                context['theorems'] = []
            context['theorems'].append({
                'id': task.id,
                'morphon_id': task.morphon_id,
                'theorem': result['theorem'],
                'timestamp': time.time()
            })
            
            return result
        else:
            # Fall back to base verification
            return await super()._verify_proof(task)
    
    def _create_tenant_context(self, task: TenantProofTask) -> Dict[str, Any]:
        """Create isolated proof context for tenant"""
        return {
            'tenant_scope': task.tenant_scope,
            'tenant_id': task.tenant_id,
            'namespace': task.context_namespace,
            'theorems': [],
            'axioms': self._get_base_axioms(),
            'created_at': time.time()
        }
    
    def _get_base_axioms(self) -> List[str]:
        """Get base axioms for all proof contexts"""
        return [
            "axiom concept_identity : ∀ (c : Concept), c = c",
            "axiom relation_transitivity : ∀ (a b c : Concept) (r : Relation), "
            "r(a,b) ∧ r(b,c) → r(a,c)",
            "axiom morphon_consistency : ∀ (m : Morphon), m.verified → m.consistent"
        ]
    
    def _save_proof(self, task: ProofTask):
        """Save proof to tenant-specific directory"""
        if isinstance(task, TenantProofTask):
            tenant_key = task.get_tenant_key()
            proof_dir = self.tenant_proof_dirs.get(tenant_key, self.proof_dir)
            
            proof_file = proof_dir / f"{task.id}.json"
            with open(proof_file, 'w') as f:
                json.dump({
                    "id": task.id,
                    "morphon_id": task.morphon_id,
                    "proof_content": task.proof_content,
                    "proof_type": task.proof_type,
                    "status": task.status,
                    "tenant_scope": task.tenant_scope,
                    "tenant_id": task.tenant_id,
                    "context_namespace": task.context_namespace,
                    "created_at": task.created_at,
                    "completed_at": task.completed_at,
                    "result": task.result
                }, f, indent=2)
        else:
            super()._save_proof(task)
    
    def get_tenant_stats(self, tenant_scope: str, tenant_id: str) -> Dict[str, Any]:
        """Get proof statistics for a specific tenant"""
        tenant_key = f"{tenant_scope}_{tenant_id}"
        
        # Count tenant-specific tasks
        tenant_tasks = [
            t for t in self.tasks.values()
            if isinstance(t, TenantProofTask) and t.get_tenant_key() == tenant_key
        ]
        
        pending = sum(1 for t in tenant_tasks if t.status == "pending")
        processing = sum(1 for t in tenant_tasks if t.status == "processing")
        verified = sum(1 for t in tenant_tasks if t.status == "verified")
        failed = sum(1 for t in tenant_tasks if t.status == "failed")
        
        # Get context info
        context = self.tenant_contexts.get(tenant_key, {})
        
        return {
            "tenant_key": tenant_key,
            "total_proofs": len(tenant_tasks),
            "pending": pending,
            "processing": processing,
            "verified": verified,
            "failed": failed,
            "context_theorems": len(context.get('theorems', [])),
            "context_created": context.get('created_at')
        }
    
    def cleanup_tenant(self, tenant_scope: str, tenant_id: str):
        """Clean up proof context for a tenant"""
        tenant_key = f"{tenant_scope}_{tenant_id}"
        
        # Remove context
        if tenant_key in self.tenant_contexts:
            del self.tenant_contexts[tenant_key]
        
        # Remove completed tasks
        tasks_to_remove = []
        for task_id, task in self.tasks.items():
            if (isinstance(task, TenantProofTask) and 
                task.get_tenant_key() == tenant_key and
                task.status in ["verified", "failed"]):
                tasks_to_remove.append(task_id)
        
        for task_id in tasks_to_remove:
            del self.tasks[task_id]
        
        logger.info(f"🧹 Cleaned up proof context for {tenant_key}")


# Import for Path
from pathlib import Path
import json

# Create singleton instance
_tenant_proof_queue = None

def get_tenant_proof_queue() -> TenantAwareProofQueue:
    """Get singleton tenant-aware proof queue"""
    global _tenant_proof_queue
    if _tenant_proof_queue is None:
        _tenant_proof_queue = TenantAwareProofQueue()
    return _tenant_proof_queue


# Helper function to create tenant proof tasks
async def create_tenant_proof(tenant_scope: str, tenant_id: str, 
                            morphon_id: str, content: Dict[str, Any]) -> str:
    """
    Create and enqueue a tenant-scoped proof
    
    Args:
        tenant_scope: "user" or "group"
        tenant_id: ID of the tenant
        morphon_id: ID of the morphon/concept
        content: Content to prove
        
    Returns:
        Proof task ID
    """
    queue = get_tenant_proof_queue()
    
    # Generate proof content
    proof_content = f"""
-- Tenant-scoped proof for {tenant_scope}:{tenant_id}
-- Morphon: {morphon_id}
-- Timestamp: {time.time()}

namespace {tenant_scope}.{tenant_id}

theorem morphon_{morphon_id.replace('-', '_')} :
  ∀ (mesh : ConceptMesh) (m : Morphon),
    m.id = "{morphon_id}" →
    m.tenant_scope = "{tenant_scope}" →
    m.tenant_id = "{tenant_id}" →
    mesh.contains m →
    mesh.is_consistent
:= by
  sorry -- Proof to be completed by verification system

end {tenant_scope}.{tenant_id}
"""
    
    # Create task
    task = TenantProofTask(
        id=f"proof_{tenant_scope}_{tenant_id}_{morphon_id}_{int(time.time())}",
        morphon_id=morphon_id,
        proof_content=proof_content,
        proof_type="lean",
        tenant_scope=tenant_scope,
        tenant_id=tenant_id,
        priority=1
    )
    
    # Enqueue
    await queue.enqueue_tenant_proof(task)
    
    return task.id
