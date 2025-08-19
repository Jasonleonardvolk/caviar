
class GhostForum:
    def __init__(self, agents: List[GhostAgent] = None, psi_archive: PsiArchive = None):
        if psi_archive is None:
            raise RuntimeError("PsiArchive must be provided to GhostForum in production.")
        self.agents = agents or self._create_default_agents()
        self.psi_archive = psi_archive
        self.past_debates: Dict[str, DebateResult] = {}    # store past debates for replay
        # ... initialization continues ...
        
    async def run_debate(self, prompt: str, context: str = "", max_rounds: int = 3) -> DebateResult:
        # ... debate logic ... 
        result = DebateResult(debate_id=debate_id, prompt=prompt, context=context, agents=self.agents, ...)
        # (after debate loop completes)  
        self.past_debates[result.debate_id] = result      # save the full debate result
        await self._archive_debate(result)                # always archive (no conditional)
        # update internal stats, then return result

    async def _archive_debate(self, result: DebateResult):
        """Archive debate for learning and transparency."""
        archive_data = {
            "timestamp": datetime.now().isoformat(), 
            "debate_id": result.debate_id,
            "prompt": result.prompt,
            "context": result.context,
            "outcome": {
                "consensus": result.consensus,
                "majority_position": result.majority_position,  
                "conflict_score": result.conflict_score,
                "confidence_score": result.confidence_score
            },
            "metrics": {
                "total_statements": result.total_statements,
                "rounds_completed": result.rounds_completed,
                "debate_time": result.debate_time,
                "success": result.success  
            }
        }
        await self.psi_archive.log_ghost_debate(archive_data)
    
    def replay_debate(self, debate_id: str) -> Optional[DebateResult]:
        """Retrieve an archived debate session by ID for debugging."""
        return self.past_debates.get(debate_id)
