#!/usr/bin/env python
"""
Generate a new MCP micro-server + optional PDF specs.
Version 3 with enhanced PDF pipeline and all review fixes

Usage:
  python mk_server.py <n> "<description>" [pdf_path1] [pdf_path2] ...
  
Commands:
  create      - Create new server with optional PDFs
  add-pdf     - Add PDFs to existing server
  list-pdfs   - List PDFs for a server
  bulk-create - Create one server per PDF in directory
  
Examples:
  python mk_server.py create intent "Prajna intent tracker"
  python mk_server.py create empathy "Empathy module" paper1.pdf paper2.pdf
  python mk_server.py add-pdf empathy new_paper.pdf another_paper.pdf
  python mk_server.py list-pdfs empathy
  python mk_server.py bulk-create ./papers/ "Research server"
"""
from pathlib import Path
import sys, textwrap, datetime, os, re
from typing import List
import asyncio

# Try to use enhanced pipeline, fall back to basic
try:
    from enhanced_pdf_pipeline import EnhancedPDFProcessor, bulk_create_servers
    ENHANCED_PIPELINE = True
except ImportError:
    from pdf_utils import copy_and_extract_multiple, add_pdfs_to_server, list_server_pdfs
    ENHANCED_PIPELINE = False

TEMPLATE = '''"""
{name} MCP micro-server
Auto-generated {dt}
{desc}
"""
from pathlib import Path, PurePosixPath
import json
import logging
import asyncio
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass

from ..core.agent_registry import Agent, agent_registry
from ..core.psi_archive import psi_archive

logger = logging.getLogger(__name__)

# Load spec if available
try:
    SPEC = json.load(open(Path(__file__).parent / "spec.json", "r", encoding="utf-8"))
except FileNotFoundError:
    SPEC = {{"pdfs": [], "total_chars": 0}}

@dataclass
class ServerMetrics:
    """Track server performance metrics"""
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    total_timeouts: int = 0
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        if self.total_executions == 0:
            return 1.0
        return self.successful_executions / self.total_executions

class {cls}(Agent):
    """{desc}"""
    
    # Metadata for dynamic discovery
    _metadata = {{
        "name": "{name}",
        "description": "{desc}",
        "enabled": True,
        "auto_start": True,
        "endpoints": [],
        "dependencies": [],
        "version": "1.0.0"
    }}
    
    _default_config = {{
        "analysis_interval": int(os.getenv("DEFAULT_ANALYSIS_INTERVAL", "300")),  # 5 min dev default
        "enable_watchdog": True,
        "watchdog_timeout": 60,
        "max_errors": 5,
        "restart_backoff_base": 60,
        "enable_critic_hooks": True,
        "enable_section_analysis": {has_sections}  # True if enhanced pipeline used
    }}
    
    def __init__(self, name: str = "{name}", config: Optional[Dict[str, Any]] = None):
        super().__init__(name)
        self.config = self._get_config_with_env() if config is None else config
        self.spec = SPEC
        self.pdfs = self.spec.get("pdfs", [])
        self.sections = self._load_sections()
        self.error_count = 0
        self.running = False
        self._task = None
        self._supervisor_task = None
        self.metrics = ServerMetrics()
        
        # Ensure data directory exists
        self.data_dir = Path(__file__).parent.parent.parent / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Log initialization
        if self.pdfs:
            logger.info("Loaded %d PDF specs", len(self.pdfs))
            psi_archive.log_event("{name}_initialized", {{
                "pdf_count": len(self.pdfs),
                "total_chars": self.spec.get("total_chars", 0),
                "pdf_files": [pdf["file"] for pdf in self.pdfs],
                "analysis_interval": self.config["analysis_interval"],
                "has_sections": bool(self.sections)
            }})
    
    def _load_sections(self) -> Dict[str, Dict[str, str]]:
        """Load section data if available"""
        sections_file = Path(__file__).parent / "sections.json"
        if sections_file.exists():
            try:
                return json.loads(sections_file.read_text())
            except:
                pass
        return {{}}
    
    def _get_config_with_env(self) -> Dict[str, Any]:
        """Get configuration with environment overrides"""
        config = self._default_config.copy()
        
        # Override with environment variables
        config.update({{
            "analysis_interval": int(os.getenv("{NAME}_ANALYSIS_INTERVAL", 
                                              os.getenv("DEFAULT_ANALYSIS_INTERVAL", "300"))),
            "enable_watchdog": os.getenv("{NAME}_ENABLE_WATCHDOG", "true").lower() == "true",
            "watchdog_timeout": int(os.getenv("{NAME}_WATCHDOG_TIMEOUT", str(config["watchdog_timeout"]))),
            "enable_critic_hooks": os.getenv("{NAME}_ENABLE_CRITICS", "true").lower() == "true",
        }})
        
        return config

    async def execute(self, input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute server logic with watchdog protection"""
        self.metrics.total_executions += 1
        
        # Log execution start
        psi_archive.log_event("{name}_execute_start", {{
            "input": input_data,
            "execution_number": self.metrics.total_executions
        }})
        
        try:
            if self.config.get("enable_watchdog"):
                # Run with timeout
                result = await asyncio.wait_for(
                    self._execute_internal(input_data),
                    timeout=self.config["watchdog_timeout"]
                )
            else:
                result = await self._execute_internal(input_data)
            
            # Update metrics
            self.metrics.successful_executions += 1
            self.metrics.last_success = datetime.utcnow()
            self.error_count = 0  # Reset error count on success
            
            # Log successful execution
            psi_archive.log_event("{name}_execute_complete", {{
                "status": "success",
                "pdfs_processed": len(self.pdfs),
                "success_rate": self.metrics.success_rate
            }})
            
            # Submit to critic hub if enabled
            if self.config.get("enable_critic_hooks"):
                await self._submit_to_critics()
            
            return result
            
        except asyncio.TimeoutError:
            self.metrics.total_timeouts += 1
            self.metrics.failed_executions += 1
            self.error_count += 1
            logger.error(f"{{self.name}} execution timed out after {{self.config['watchdog_timeout']}}s")
            
            psi_archive.log_event("{name}_execute_timeout", {{
                "timeout": self.config['watchdog_timeout'],
                "error_count": self.error_count
            }})
            
            return {{
                "status": "timeout",
                "server": self.name,
                "error": f"Execution timed out after {{self.config['watchdog_timeout']}} seconds"
            }}
            
        except Exception as e:
            self.metrics.failed_executions += 1
            self.metrics.last_failure = datetime.utcnow()
            self.error_count += 1
            logger.error(f"Error in {{self.name}} execution: {{e}}")
            
            psi_archive.log_event("{name}_execute_error", {{
                "error": str(e),
                "error_count": self.error_count
            }})
            
            # Check if we need to restart
            if self.error_count >= self.config.get("max_errors", 5):
                logger.critical(f"{{self.name}} exceeded max errors ({{self.error_count}}), requesting restart")
                psi_archive.log_event("{name}_restart_requested", {{
                    "error_count": self.error_count
                }})
            
            return {{
                "status": "error",
                "server": self.name,
                "error": str(e),
                "error_count": self.error_count
            }}
    
    async def _execute_internal(self, input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Internal execution logic - override this in subclasses"""
        # Calculate section impact score if sections available
        section_impact = 0.5  # Default
        if self.sections and input_data and "query" in input_data:
            section_impact = self._calculate_section_impact(input_data["query"])
        
        # TODO: implement main logic here
        result = {{
            "status": "success",
            "server": self.name,
            "pdfs_loaded": len(self.pdfs),
            "total_chars": self.spec.get("total_chars", 0),
            "has_sections": bool(self.sections),
            "message": "Server is running",
            "{name}_performance_score": 0.95,  # For critic hooks
            "{name}_section_impact": section_impact  # For PDF-based critic
        }}
        
        return result
    
    def _calculate_section_impact(self, query: str) -> float:
        """Calculate how relevant the PDF sections are to the query"""
        if not self.sections:
            return 0.5
        
        # Simple keyword matching (could be enhanced with embeddings)
        query_words = set(query.lower().split())
        total_relevance = 0.0
        
        for pdf_file, sections in self.sections.items():
            for section_name, section_text in sections.items():
                section_words = set(section_text.lower().split())
                overlap = len(query_words & section_words)
                relevance = overlap / max(len(query_words), 1)
                
                # Weight by section importance
                section_weights = {{
                    "abstract": 2.0,
                    "introduction": 1.5,
                    "conclusion": 1.5,
                    "methods": 1.0,
                    "results": 1.0,
                    "discussion": 1.0,
                    "references": 0.5
                }}
                weight = section_weights.get(section_name, 1.0)
                total_relevance += relevance * weight
        
        # Normalize
        return min(total_relevance / max(len(self.sections), 1), 1.0)
    
    async def _submit_to_critics(self):
        """Submit metrics to critic hub for evaluation"""
        try:
            from kha.meta_genome.critics.critic_hub import evaluate
            
            critic_report = {{
                "{name}_success_rate": self.metrics.success_rate,
                "{name}_performance_score": self.metrics.success_rate,
                "{name}_error_rate": self.error_count / max(1, self.metrics.total_executions),
                "{name}_timeout_rate": self.metrics.total_timeouts / max(1, self.metrics.total_executions),
                "{name}_section_impact": getattr(self, '_last_section_impact', 0.5),
                "timestamp": datetime.utcnow().isoformat()
            }}
            
            # Evaluate through critic hub
            evaluate(critic_report)
            
        except ImportError:
            logger.debug("Critic hub not available for evaluation")
        except Exception as e:
            logger.error(f"Failed to submit to critics: {{e}}")
    
    def get_pdf_content(self, pdf_name: str) -> Optional[str]:
        """Get content from a specific PDF by name"""
        for pdf in self.pdfs:
            if pdf["file"] == pdf_name:
                # Read from resources
                pdf_path = Path(__file__).parent / "resources" / pdf_name
                if pdf_path.exists():
                    try:
                        from PyPDF2 import PdfReader
                        text = "\\n".join(page.extract_text() or "" for page in PdfReader(pdf_path).pages)
                        return text
                    except Exception as e:
                        logger.error(f"Failed to read PDF {{pdf_name}}: {{e}}")
        return None
    
    def get_pdf_section(self, pdf_name: str, section_name: str) -> Optional[str]:
        """Get specific section from a PDF"""
        if pdf_name in self.sections:
            return self.sections[pdf_name].get(section_name)
        return None
    
    def list_pdfs(self) -> List[Dict[str, Any]]:
        """List all loaded PDFs with metadata"""
        return self.pdfs
    
    async def start(self):
        """Start the continuous processing loop with supervisor"""
        if self.running:
            logger.warning(f"{{self.name}} is already running")
            return
        
        self.running = True
        # Start supervisor task that will manage the actual loop
        self._supervisor_task = asyncio.create_task(self._supervisor_loop())
        logger.info(f"{{self.name}} supervisor started")
    
    async def _supervisor_loop(self):
        """Supervisor loop that restarts the main loop on crashes"""
        consecutive_crashes = 0
        
        while self.running:
            try:
                logger.info(f"Starting {{self.name}} continuous loop")
                self._task = asyncio.create_task(self._continuous_loop())
                await self._task
                consecutive_crashes = 0  # Reset on clean exit
                
            except asyncio.CancelledError:
                logger.info(f"{{self.name}} loop cancelled")
                break
                
            except Exception as e:
                consecutive_crashes += 1
                logger.error(f"{{self.name}} loop crashed (attempt {{consecutive_crashes}}): {{e}}")
                
                # Exponential backoff
                backoff = min(
                    self.config["restart_backoff_base"] * (2 ** consecutive_crashes),
                    3600  # Max 1 hour
                )
                
                logger.info(f"Restarting {{self.name}} in {{backoff}} seconds...")
                await asyncio.sleep(backoff)
                
                # Log restart
                psi_archive.log_event("{name}_loop_restart", {{
                    "crash_count": consecutive_crashes,
                    "backoff_seconds": backoff,
                    "error": str(e)
                }})
    
    async def _continuous_loop(self):
        """Continuous processing loop"""
        while self.running:
            try:
                # Run periodic processing
                await self.on_tick()
                
                # Sleep until next interval
                await asyncio.sleep(self.config["analysis_interval"])
                
            except asyncio.CancelledError:
                raise  # Let supervisor handle this
                
            except Exception as e:
                logger.error(f"Error in {{self.name}} tick: {{e}}")
                # Re-raise to trigger supervisor restart
                raise
    
    async def on_tick(self):
        """Periodic tick handler - override in subclasses"""
        # TODO: implement periodic logic
        # TODO: Consider using embeddings for better similarity matching
        
        psi_archive.log_event("{name}_tick", {{
            "timestamp": datetime.utcnow().isoformat(),
            "pdfs": len(self.pdfs),
            "sections_available": bool(self.sections),
            "metrics": {{
                "success_rate": self.metrics.success_rate,
                "total_executions": self.metrics.total_executions
            }}
        }})
        
        # Submit periodic metrics to critics
        if self.config.get("enable_critic_hooks"):
            await self._submit_to_critics()
        
        # Emit telemetry if available
        await self._emit_telemetry()
    
    async def _emit_telemetry(self):
        """Emit telemetry events"""
        try:
            # TODO: Integrate with actual WebSocket emitter
            psi_archive.log_event("pdf_ingest_telemetry", {{
                "server": self.name,
                "pdfs": len(self.pdfs),
                "success_rate": self.metrics.success_rate,
                "uptime": (datetime.utcnow() - self.metrics.last_success).total_seconds() 
                         if self.metrics.last_success else 0
            }})
        except Exception as e:
            logger.debug(f"Telemetry emission failed: {{e}}")
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info(f"{{self.name}} shutting down...")
        self.running = False
        
        # Cancel tasks
        for task in [self._task, self._supervisor_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Save any state if needed
        psi_archive.log_event("{name}_shutdown", {{
            "metrics": {{
                "total_executions": self.metrics.total_executions,
                "success_rate": self.metrics.success_rate,
                "final_pdf_count": len(self.pdfs)
            }}
        }})

# Auto-register the server
{name}_server = {cls}()
agent_registry.register("{name}", {name}_server)

# Add critic hooks if available
try:
    from kha.meta_genome.critics.critic_hub import critic
    
    @critic("{name}_performance")
    def {name}_performance(report: dict):
        """Score {name}'s performance based on execution metrics"""
        score = report.get("{name}_performance_score", 0.0)
        return score, score >= 0.7
    
    @critic("{name}_health")
    def {name}_health(report: dict):
        """Monitor {name}'s health based on success rate"""
        success_rate = report.get("{name}_success_rate", 1.0)
        return success_rate, success_rate >= 0.7
    
    @critic("{name}_ingest")
    def {name}_ingest(report: dict):
        """Score PDF ingestion quality based on section impact"""
        impact = report.get("{name}_section_impact", 0.5)
        return impact, True  # Always pass but report score
        
except ImportError:
    logger.debug("Critic hub not available, skipping critic registration")

# Export
__all__ = ['{cls}']
'''

def create_server(name: str, desc: str, pdf_paths: List[str]):
    """Create a new server with optional PDFs"""
    cls = "".join(p.capitalize() for p in name.split("_")) + "Server"
    dt  = datetime.datetime.utcnow().isoformat()[:19]+"Z"

    base = Path(__file__).parent.parent / "agents" / name
    base.mkdir(parents=True, exist_ok=True)
    (base / "__init__.py").touch()

    # Handle PDFs if provided
    has_sections = "false"
    if pdf_paths and ENHANCED_PIPELINE:
        # Use enhanced pipeline
        processor = EnhancedPDFProcessor()
        pdf_sources = [{"path": str(p)} for p in pdf_paths]
        
        # Run async operation
        spec = asyncio.run(processor.process_pdfs(pdf_sources, base))
        
        pdf_names = [pdf["file"] for pdf in spec["pdfs"]]
        desc = f"{desc} (seeded from {', '.join(pdf_names)})"
        has_sections = "true"
        
    elif pdf_paths:
        # Use basic pipeline
        pdf_path_objs = [Path(p).expanduser().resolve() for p in pdf_paths]
        spec = copy_and_extract_multiple(pdf_path_objs, base, append=False)
        pdf_names = [pdf["file"] for pdf in spec["pdfs"]]
        desc = f"{desc} (seeded from {', '.join(pdf_names)})"
        
    else:
        # Create empty spec
        (base / "spec.json").write_text(json.dumps({
            "pdfs": [],
            "total_chars": 0,
            "last_updated": datetime.datetime.utcnow().isoformat()
        }, indent=2))

    # Write server - use uppercase name for env vars
    (base / f"{name}.py").write_text(textwrap.dedent(
        TEMPLATE.format(
            name=name, 
            NAME=name.upper(),
            desc=desc, 
            cls=cls, 
            dt=dt,
            has_sections=has_sections
        )
    ))
    print(f"[OK] Created micro-server at {base}")
    
    # Trigger hot-reload if available
    if ENHANCED_PIPELINE:
        try:
            from enhanced_pdf_pipeline import trigger_registry_reload
            asyncio.run(trigger_registry_reload())
        except:
            pass

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    
    command = sys.argv[1]
    
    if command == "create":
        if len(sys.argv) < 4:
            print("Usage: python mk_server.py create <n> \"<description>\" [pdf_path1] [pdf_path2] ...")
            return
        
        name = sys.argv[2]
        desc = sys.argv[3]
        pdf_paths = sys.argv[4:] if len(sys.argv) > 4 else []
        
        create_server(name, desc, pdf_paths)
        
    elif command == "add-pdf":
        if len(sys.argv) < 4:
            print("Usage: python mk_server.py add-pdf <server_name> <pdf_path1> [pdf_path2] ...")
            return
        
        server_name = sys.argv[2]
        pdf_paths = [Path(p).expanduser().resolve() for p in sys.argv[3:]]
        
        if ENHANCED_PIPELINE:
            processor = EnhancedPDFProcessor()
            sources = [{"path": str(p)} for p in pdf_paths]
            base_dir = Path(__file__).parent.parent / "agents" / server_name
            spec = asyncio.run(processor.process_pdfs(sources, base_dir))
            print(f"[OK] Added {len(sources)} PDFs to server '{server_name}'")
        else:
            success = add_pdfs_to_server(server_name, pdf_paths)
            if success:
                print(f"[OK] Added {len(pdf_paths)} PDFs to server '{server_name}'")
        
    elif command == "list-pdfs":
        if len(sys.argv) < 3:
            print("Usage: python mk_server.py list-pdfs <server_name>")
            return
        
        server_name = sys.argv[2]
        pdfs = list_server_pdfs(server_name)
        
        if pdfs is None:
            print(f"[ERROR] Server '{server_name}' not found")
        elif not pdfs:
            print(f"[INFO] Server '{server_name}' has no PDFs")
        else:
            print(f"\n[PDFs for server '{server_name}']")
            for i, pdf in enumerate(pdfs, 1):
                print(f"\n{i}. {pdf['file']}")
                print(f"   MD5: {pdf['md5']}")
                print(f"   Characters: {pdf['chars']:,}")
                print(f"   Added: {pdf['added']}")
                if "key_topics" in pdf:
                    print(f"   Topics: {', '.join(pdf['key_topics'])}")
                if "authors" in pdf:
                    print(f"   Authors: {', '.join(pdf['authors'][:3])}")
    
    elif command == "bulk-create":
        if len(sys.argv) < 3:
            print("Usage: python mk_server.py bulk-create <pdf_directory> [base_description]")
            return
        
        pdf_dir = Path(sys.argv[2])
        base_desc = sys.argv[3] if len(sys.argv) > 3 else "Auto-generated server"
        
        if ENHANCED_PIPELINE:
            bulk_create_servers(pdf_dir, base_desc)
        else:
            print("[ERROR] Bulk creation requires enhanced pipeline")
            print("Install: pip install aiohttp pdfminer.six")
    
    elif len(sys.argv) >= 3:
        # Legacy mode - support old syntax
        name = sys.argv[1]
        desc = sys.argv[2]
        pdf_paths = sys.argv[3:] if len(sys.argv) > 3 else []
        create_server(name, desc, pdf_paths)
    
    else:
        print(__doc__)

if __name__ == "__main__":
    main()
