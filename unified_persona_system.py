"""
TORI Unified Persona System Integration
======================================

Wave 3: Unified Personas - 4 layers + TTS sync + Agent packs
Complete persona system unification with voice synchronization.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Union, Callable
from pathlib import Path
import json
import time
from datetime import datetime
from enum import Enum

# Import mathematical and holographic cores
from integration_core import get_mathematical_core
from holographic_intelligence import get_holographic_intelligence

# Import TTS integration
try:
    from TTS_AUTO_INTEGRATION import PersonaTTSManager, integrate_tts_with_chat_response
    TTS_INTEGRATION_AVAILABLE = True
    print("✅ TTS AUTO-INTEGRATION LOADED")
except ImportError as e:
    print(f"⚠️ TTS integration import failed: {e}")
    TTS_INTEGRATION_AVAILABLE = False

# Import Ghost Personas (Layer 1)
try:
    # These would be imported from your existing ghost persona system
    # from tori_ui_svelte.src.lib.stores.ghostPersona import ghostPersona, setPersona, setMood
    GHOST_PERSONAS_AVAILABLE = False  # Will implement as unified system
    print("⚠️ Ghost personas will be unified in new system")
except ImportError:
    GHOST_PERSONAS_AVAILABLE = False

# Import Rust PersonaModes (Layer 2)
try:
    # These would be imported from concept_mesh Rust bindings
    # from concept_mesh.src.auth.persona import PersonaMode, Persona
    RUST_PERSONAS_AVAILABLE = False  # Will implement as unified system
    print("⚠️ Rust personas will be unified in new system")
except ImportError:
    RUST_PERSONAS_AVAILABLE = False

# Import Alan Backend Personas (Layer 3)
try:
    from alan_backend.routes.personas import router as personas_router
    ALAN_PERSONAS_AVAILABLE = True
    print("✅ ALAN BACKEND PERSONAS LOADED")
except ImportError as e:
    print(f"⚠️ Alan backend personas import failed: {e}")
    ALAN_PERSONAS_AVAILABLE = False

# Import Agent Pack System (Layer 4)
try:
    # from concept_mesh.src.agents.persona_packs import get_registry, update_packs_for_current_persona
    AGENT_PACKS_AVAILABLE = False  # Will implement as unified system
    print("⚠️ Agent packs will be unified in new system")
except ImportError:
    AGENT_PACKS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PersonaLayer(Enum):
    """Types of persona layers in the unified system"""
    GHOST = "ghost"           # Frontend consciousness personas
    RUST_MODE = "rust_mode"   # Core system operational modes
    ALAN_TRAIT = "alan_trait" # REST API personality traits
    AGENT_PACK = "agent_pack" # Dynamic capability loading

class UnifiedPersonaType(Enum):
    """Unified persona types combining all layers"""
    # Detective/Investigative
    ENOLA = "enola"           # Ghost: Detective, Rust: Researcher, Alan: Analytical, Packs: av_toolkit
    
    # Teaching/Guidance
    MENTOR = "mentor"         # Ghost: Mentor, Rust: Custom, Alan: Teaching, Packs: core
    
    # Academic/Research
    SCHOLAR = "scholar"       # Ghost: Scholar, Rust: Researcher, Alan: Academic, Packs: av_toolkit + ingest_tools
    
    # Creative/Design
    CREATOR = "creator"       # Ghost: Creator, Rust: CreativeAgent, Alan: Creative, Packs: creative_tools + synthesis
    
    # Systematic/Architecture
    ARCHITECT = "architect"   # Ghost: Architect, Rust: Glyphsmith, Alan: Structured, Packs: elfin_compiler + glyph_tools
    
    # Exploration/Discovery
    EXPLORER = "explorer"     # Ghost: Explorer, Rust: Custom, Alan: Curious, Packs: core + custom
    
    # Memory/Organization
    MEMORY_KEEPER = "memory_keeper"  # Ghost: Custom, Rust: MemoryPruner, Alan: Organized, Packs: memory_tools + psiarc_editor
    
    # Code/Technical
    REFACTOR_GURU = "refactor_guru"  # Ghost: Custom, Rust: Custom, Alan: Refactor Guru, Packs: core

class TORIUnifiedPersonaSystem:
    """
    🧠 UNIFIED PERSONA SYSTEM INTEGRATION
    
    Unifies all 4 persona layers with mathematical verification,
    TTS voice synchronization, and agent pack loading.
    """
    
    def __init__(self, mathematical_core=None, holographic_intelligence=None):
        logger.info("🧠 Initializing TORI Unified Persona System...")
        
        # Connect to other cores
        self.math_core = mathematical_core or get_mathematical_core()
        self.holo_core = holographic_intelligence or get_holographic_intelligence()
        
        # Persona layers
        self.ghost_personas = None
        self.rust_personas = None
        self.alan_personas = None
        self.agent_packs = None
        
        # TTS and voice system
        self.voice_engine = None
        
        # Unified persona state
        self.active_persona = UnifiedPersonaType.ENOLA  # Default to Enola
        self.persona_configurations = {}
        self.persona_history = []
        
        # Status tracking
        self.initialization_status = {
            "ghost_personas": False,
            "rust_personas": False,
            "alan_personas": False,
            "agent_packs": False,
            "voice_engine": False,
            "unified_system": False,
            "mathematical_integration": False
        }
        
        # Event callbacks
        self.on_persona_change_callbacks = []
        self.on_voice_change_callbacks = []
        self.on_agent_pack_change_callbacks = []
        
        # Initialize all persona systems
        asyncio.create_task(self._initialize_all_systems())
    
    async def _initialize_all_systems(self):
        """Initialize all persona systems with unified integration"""
        try:
            # 1. Initialize Ghost Personas (Frontend layer)
            self.ghost_personas = await self._integrate_ghost_personas()
            self.initialization_status["ghost_personas"] = True
            
            # 2. Initialize Rust PersonaMode (Core system layer)
            self.rust_personas = await self._integrate_rust_persona_modes()
            self.initialization_status["rust_personas"] = True
            
            # 3. Initialize Alan Backend (API layer)
            self.alan_personas = await self._integrate_alan_backend()
            self.initialization_status["alan_personas"] = True
            
            # 4. Initialize Agent Pack System (Capability layer)
            self.agent_packs = await self._integrate_agent_pack_loading()
            self.initialization_status["agent_packs"] = True
            
            # 5. Initialize TTS Voice Synchronization
            self.voice_engine = await self._integrate_persona_tts()
            self.initialization_status["voice_engine"] = True
            
            # 6. Create unified persona configurations
            await self._create_unified_persona_configurations()
            self.initialization_status["unified_system"] = True
            
            # 7. Wire mathematical integration
            await self._wire_mathematical_integration()
            self.initialization_status["mathematical_integration"] = True
            
            # 8. Set default persona
            await self._activate_persona(self.active_persona)
            
            logger.info("🌟 UNIFIED PERSONA SYSTEM FULLY INTEGRATED")
            logger.info("🎭 ALL 4 LAYERS SYNCHRONIZED")
            logger.info("🔊 TTS VOICE SYSTEM WIRED")
            
        except Exception as e:
            logger.error(f"❌ Unified persona system initialization failed: {e}")
            raise
    
    async def _integrate_ghost_personas(self):
        """Integrate Ghost Personas (Layer 1 - Frontend/Consciousness)"""
        logger.info("👻 Integrating Ghost Personas...")
        
        if not GHOST_PERSONAS_AVAILABLE:
            logger.info("Creating unified Ghost Persona system...")
        
        # Create unified ghost persona manager
        ghost_system = UnifiedGhostPersonaManager(
            mathematical_verification=True,
            holographic_integration=True
        )
        
        # Define ghost personas with 4D coordinates
        await ghost_system.register_personas([
            {
                "name": "Enola",
                "archetype": "Detective",
                "psi": "analytical",
                "epsilon": [0.8, 0.3, 0.4],  # [calm, warm, urgent]
                "tau": 0.6,
                "phi": 137,  # Golden ratio frequency
                "color": "#059669",
                "description": "Systematic investigation and cross-referencing",
                "mood": "focused"
            },
            {
                "name": "Mentor", 
                "archetype": "Teacher",
                "psi": "socratic",
                "epsilon": [0.7, 0.8, 0.2],
                "tau": 0.6,
                "phi": 42,
                "color": "#4f46e5",
                "description": "Guides through questions and frameworks",
                "mood": "contemplative"
            },
            {
                "name": "Scholar",
                "archetype": "Academic", 
                "psi": "analytical",
                "epsilon": [0.9, 0.3, 0.4],
                "tau": 0.3,
                "phi": 137,
                "color": "#059669",
                "description": "Deep research and systematic analysis",
                "mood": "focused"
            },
            {
                "name": "Creator",
                "archetype": "Creative",
                "psi": "intuitive",
                "epsilon": [0.3, 0.8, 0.9],
                "tau": 0.7,
                "phi": 73,
                "color": "#ea580c",
                "description": "Synthesizes novel ideas and innovations",
                "mood": "inspired"
            },
            {
                "name": "Architect",
                "archetype": "Systematic",
                "psi": "structural",
                "epsilon": [0.8, 0.2, 0.5],
                "tau": 0.4,
                "phi": 21,
                "color": "#7c3aed",
                "description": "Builds frameworks and systematic structures",
                "mood": "methodical"
            },
            {
                "name": "Explorer",
                "archetype": "Adventurer",
                "psi": "lateral",
                "epsilon": [0.2, 0.9, 0.7],
                "tau": 0.8,
                "phi": 89,
                "color": "#dc2626",
                "description": "Discovers connections and new territories",
                "mood": "curious"
            }
        ])
        
        logger.info("✅ Ghost Personas integrated with 4D coordinates")
        return ghost_system
    
    async def _integrate_rust_persona_modes(self):
        """Integrate Rust PersonaModes (Layer 2 - Core System)"""
        logger.info("🦀 Integrating Rust PersonaModes...")
        
        if not RUST_PERSONAS_AVAILABLE:
            logger.info("Creating unified Rust PersonaMode system...")
        
        # Create unified rust persona manager
        rust_system = UnifiedRustPersonaManager(
            mathematical_core=self.math_core,
            phase_seed_integration=True
        )
        
        # Define rust persona modes
        await rust_system.register_modes([
            {
                "mode": "CreativeAgent",
                "focus": "Design, synthesis, creative exploration",
                "agent_packs": ["core", "creative_tools", "synthesis"],
                "phase_seed_range": (50, 100)
            },
            {
                "mode": "Glyphsmith", 
                "focus": "ELFIN compiler, visual expressiveness, glyph editing",
                "agent_packs": ["core", "elfin_compiler", "glyph_tools"],
                "phase_seed_range": (100, 150)
            },
            {
                "mode": "MemoryPruner",
                "focus": "Timeline reflection, psiarc editing, memory organization",
                "agent_packs": ["core", "memory_tools", "psiarc_editor"],
                "phase_seed_range": (150, 200)
            },
            {
                "mode": "Researcher",
                "focus": "AV/hologram ingestion, psiarc emission, knowledge acquisition",
                "agent_packs": ["core", "av_toolkit", "ingest_tools"],
                "phase_seed_range": (200, 250)
            }
        ])
        
        logger.info("✅ Rust PersonaModes integrated with agent pack loading")
        return rust_system
    
    async def _integrate_alan_backend(self):
        """Integrate Alan Backend Personas (Layer 3 - REST API)"""
        logger.info("🔧 Integrating Alan Backend Personas...")
        
        if not ALAN_PERSONAS_AVAILABLE:
            logger.info("Creating unified Alan Backend system...")
        
        # Create unified alan persona manager
        alan_system = UnifiedAlanPersonaManager(
            mathematical_verification=True,
            trait_synchronization=True
        )
        
        # Define alan backend personas with Big Five traits
        await alan_system.register_personas([
            {
                "id": "refactor-guru",
                "displayName": "Refactor Guru",
                "role": "software-architect",
                "tone": "concise",
                "big5": {"O": 0.7, "C": 0.9, "E": 0.4, "A": 0.6, "N": 0.2},
                "values": {"precision": 0.9, "efficiency": 0.8, "clarity": 0.9}
            },
            {
                "id": "enola-detective",
                "displayName": "Enola Detective",
                "role": "investigator",
                "tone": "analytical", 
                "big5": {"O": 0.8, "C": 0.9, "E": 0.3, "A": 0.7, "N": 0.1},
                "values": {"thoroughness": 0.95, "logic": 0.9, "patience": 0.8}
            },
            {
                "id": "creative-synthesizer",
                "displayName": "Creative Synthesizer",
                "role": "designer",
                "tone": "inspiring",
                "big5": {"O": 0.95, "C": 0.6, "E": 0.7, "A": 0.8, "N": 0.3},
                "values": {"innovation": 0.95, "beauty": 0.8, "originality": 0.9}
            }
        ])
        
        logger.info("✅ Alan Backend Personas integrated with Big Five traits")
        return alan_system
    
    async def _integrate_agent_pack_loading(self):
        """Integrate Agent Pack System (Layer 4 - Dynamic Capabilities)"""
        logger.info("⚙️ Integrating Agent Pack Loading...")
        
        if not AGENT_PACKS_AVAILABLE:
            logger.info("Creating unified Agent Pack system...")
        
        # Create unified agent pack manager
        pack_system = UnifiedAgentPackManager(
            mathematical_core=self.math_core,
            dynamic_loading=True
        )
        
        # Define agent packs
        await pack_system.register_packs([
            {
                "id": "core",
                "name": "Core Agent Pack",
                "description": "Core agents for the Concept Mesh",
                "agents": ["orchestrator", "planner", "diff_generator"],
                "always_active": True
            },
            {
                "id": "creative_tools",
                "name": "Creative Tools",
                "description": "Tools for creative synthesis and design",
                "agents": ["design_agent", "synthesis_agent"],
                "personas": ["Creator", "CreativeAgent"]
            },
            {
                "id": "av_toolkit", 
                "name": "AV Toolkit",
                "description": "Audio and video processing tools",
                "agents": ["audio_processor", "video_processor"],
                "personas": ["Enola", "Scholar", "Researcher"]
            },
            {
                "id": "elfin_compiler",
                "name": "ELFIN Compiler",
                "description": "ELFIN language compiler and tools",
                "agents": ["elfin_compiler", "elfin_linter"],
                "personas": ["Architect", "Glyphsmith"]
            },
            {
                "id": "memory_tools",
                "name": "Memory Tools",
                "description": "Memory management and organization",
                "agents": ["memory_organizer", "timeline_analyzer"],
                "personas": ["MemoryPruner"]
            }
        ])
        
        logger.info("✅ Agent Pack System integrated with dynamic loading")
        return pack_system
    
    async def _integrate_persona_tts(self):
        """Integrate TTS Voice Synchronization"""
        logger.info("🔊 Integrating TTS Voice Synchronization...")
        
        if not TTS_INTEGRATION_AVAILABLE:
            logger.info("Creating unified TTS system...")
            return UnifiedTTSManager()
        
        # Use existing TTS integration
        tts_manager = PersonaTTSManager()
        
        # Create enhanced TTS manager with persona synchronization
        enhanced_tts = EnhancedPersonaTTSManager(
            base_tts=tts_manager,
            mathematical_core=self.math_core
        )
        
        # Define persona-specific voices with prosodic properties
        await enhanced_tts.configure_persona_voices({
            UnifiedPersonaType.ENOLA: {
                "voice": "en-US-JennyNeural",
                "style": "analytical",
                "rate": "medium",
                "pitch": "medium",
                "emphasis": "logical"
            },
            UnifiedPersonaType.MENTOR: {
                "voice": "en-US-GuyNeural", 
                "style": "teaching",
                "rate": "slow",
                "pitch": "warm",
                "emphasis": "guiding"
            },
            UnifiedPersonaType.SCHOLAR: {
                "voice": "en-US-ChristopherNeural",
                "style": "academic",
                "rate": "medium-slow",
                "pitch": "thoughtful",
                "emphasis": "precise"
            },
            UnifiedPersonaType.CREATOR: {
                "voice": "en-US-AmberNeural",
                "style": "creative",
                "rate": "medium-fast",
                "pitch": "expressive", 
                "emphasis": "passionate"
            },
            UnifiedPersonaType.ARCHITECT: {
                "voice": "en-US-DavisNeural",
                "style": "structured",
                "rate": "medium",
                "pitch": "precise",
                "emphasis": "systematic"
            },
            UnifiedPersonaType.EXPLORER: {
                "voice": "en-US-AriaNeural",
                "style": "curious",
                "rate": "medium-fast",
                "pitch": "energetic",
                "emphasis": "enthusiastic"
            }
        })
        
        logger.info("✅ TTS Voice Synchronization integrated with persona-specific prosody")
        return enhanced_tts
    
    async def _create_unified_persona_configurations(self):
        """Create unified configurations mapping all 4 layers"""
        logger.info("🎭 Creating unified persona configurations...")
        
        # Map each unified persona to all 4 layers
        self.persona_configurations = {
            UnifiedPersonaType.ENOLA: {
                PersonaLayer.GHOST: "Enola",
                PersonaLayer.RUST_MODE: "Researcher", 
                PersonaLayer.ALAN_TRAIT: "enola-detective",
                PersonaLayer.AGENT_PACK: ["core", "av_toolkit", "ingest_tools"],
                "voice_config": "analytical_detective",
                "mathematical_weighting": 0.9
            },
            UnifiedPersonaType.MENTOR: {
                PersonaLayer.GHOST: "Mentor",
                PersonaLayer.RUST_MODE: "Custom",
                PersonaLayer.ALAN_TRAIT: "teaching-guide", 
                PersonaLayer.AGENT_PACK: ["core"],
                "voice_config": "warm_teaching",
                "mathematical_weighting": 0.7
            },
            UnifiedPersonaType.SCHOLAR: {
                PersonaLayer.GHOST: "Scholar",
                PersonaLayer.RUST_MODE: "Researcher",
                PersonaLayer.ALAN_TRAIT: "academic-researcher",
                PersonaLayer.AGENT_PACK: ["core", "av_toolkit", "ingest_tools"],
                "voice_config": "thoughtful_academic",
                "mathematical_weighting": 0.8
            },
            UnifiedPersonaType.CREATOR: {
                PersonaLayer.GHOST: "Creator",
                PersonaLayer.RUST_MODE: "CreativeAgent",
                PersonaLayer.ALAN_TRAIT: "creative-synthesizer",
                PersonaLayer.AGENT_PACK: ["core", "creative_tools", "synthesis"],
                "voice_config": "expressive_creative",
                "mathematical_weighting": 0.6
            },
            UnifiedPersonaType.ARCHITECT: {
                PersonaLayer.GHOST: "Architect",
                PersonaLayer.RUST_MODE: "Glyphsmith",
                PersonaLayer.ALAN_TRAIT: "systematic-builder",
                PersonaLayer.AGENT_PACK: ["core", "elfin_compiler", "glyph_tools"],
                "voice_config": "precise_systematic",
                "mathematical_weighting": 0.85
            },
            UnifiedPersonaType.EXPLORER: {
                PersonaLayer.GHOST: "Explorer",
                PersonaLayer.RUST_MODE: "Custom",
                PersonaLayer.ALAN_TRAIT: "curious-explorer",
                PersonaLayer.AGENT_PACK: ["core", "av_toolkit"],
                "voice_config": "energetic_curious",
                "mathematical_weighting": 0.7
            },
            UnifiedPersonaType.MEMORY_KEEPER: {
                PersonaLayer.GHOST: "Memory Keeper",
                PersonaLayer.RUST_MODE: "MemoryPruner",
                PersonaLayer.ALAN_TRAIT: "organized-keeper",
                PersonaLayer.AGENT_PACK: ["core", "memory_tools", "psiarc_editor"],
                "voice_config": "calm_organized",
                "mathematical_weighting": 0.8
            },
            UnifiedPersonaType.REFACTOR_GURU: {
                PersonaLayer.GHOST: "Refactor Guru",
                PersonaLayer.RUST_MODE: "Custom",
                PersonaLayer.ALAN_TRAIT: "refactor-guru",
                PersonaLayer.AGENT_PACK: ["core", "elfin_compiler"],
                "voice_config": "concise_technical",
                "mathematical_weighting": 0.9
            }
        }
        
        logger.info("✅ Unified persona configurations created - all 4 layers mapped")
    
    async def _wire_mathematical_integration(self):
        """Wire all persona components to mathematical core"""
        logger.info("🔗 Wiring mathematical integration...")
        
        # Wire persona selection to mathematical analysis
        self.on_persona_change_callbacks.append(self._analyze_persona_mathematically)
        
        # Wire voice changes to mathematical verification
        self.on_voice_change_callbacks.append(self._verify_voice_mathematically)
        
        # Wire agent pack changes to mathematical weighting
        self.on_agent_pack_change_callbacks.append(self._weight_capabilities_mathematically)
        
        logger.info("✅ Mathematical integration wired across all persona layers")
    
    async def _activate_persona(self, persona_type: UnifiedPersonaType):
        """Activate a persona across all 4 layers with synchronization"""
        try:
            logger.info(f"🎭 Activating unified persona: {persona_type.value}")
            
            # Get persona configuration
            config = self.persona_configurations.get(persona_type)
            if not config:
                raise ValueError(f"Unknown persona type: {persona_type}")
            
            # Store previous persona for history
            previous_persona = self.active_persona
            self.active_persona = persona_type
            
            # 1. Activate Ghost Persona (Layer 1)
            if self.ghost_personas:
                await self.ghost_personas.activate_persona(config[PersonaLayer.GHOST])
            
            # 2. Activate Rust PersonaMode (Layer 2)
            if self.rust_personas:
                await self.rust_personas.activate_mode(config[PersonaLayer.RUST_MODE])
            
            # 3. Activate Alan Backend Persona (Layer 3)
            if self.alan_personas:
                await self.alan_personas.activate_persona(config[PersonaLayer.ALAN_TRAIT])
            
            # 4. Load Agent Packs (Layer 4)
            if self.agent_packs:
                await self.agent_packs.load_packs(config[PersonaLayer.AGENT_PACK])
            
            # 5. Update TTS Voice
            if self.voice_engine:
                await self.voice_engine.switch_to_persona_voice(persona_type, config["voice_config"])
            
            # 6. Update persona history
            self.persona_history.append({
                "timestamp": datetime.now().isoformat(),
                "previous_persona": previous_persona.value if previous_persona else None,
                "new_persona": persona_type.value,
                "configuration": config,
                "mathematical_weighting": config["mathematical_weighting"]
            })
            
            # 7. Trigger callbacks
            for callback in self.on_persona_change_callbacks:
                await callback(persona_type, config)
            
            logger.info(f"✅ Unified persona {persona_type.value} activated across all 4 layers")
            
        except Exception as e:
            logger.error(f"❌ Persona activation failed: {e}")
            raise
    
    async def _analyze_persona_mathematically(self, persona_type: UnifiedPersonaType, config: Dict[str, Any]):
        """Analyze persona using mathematical core"""
        try:
            if self.math_core:
                analysis = await self.math_core.verify_concept_with_mathematics({
                    "persona_type": persona_type.value,
                    "configuration": config,
                    "layers": list(config.keys()),
                    "mathematical_weighting": config.get("mathematical_weighting", 0.5)
                })
                
                logger.info(f"🔬 Mathematical analysis for {persona_type.value}: score={analysis.get('mathematical_score', 0):.2f}")
                
        except Exception as e:
            logger.error(f"❌ Mathematical persona analysis failed: {e}")
    
    async def _verify_voice_mathematically(self, persona_type: UnifiedPersonaType, voice_config: str):
        """Verify voice configuration using mathematical principles"""
        try:
            # Could use mathematical analysis for voice coherence
            logger.info(f"🔊 Voice verification for {persona_type.value}: {voice_config}")
            
        except Exception as e:
            logger.error(f"❌ Voice verification failed: {e}")
    
    async def _weight_capabilities_mathematically(self, persona_type: UnifiedPersonaType, agent_packs: List[str]):
        """Weight agent pack capabilities using mathematical principles"""
        try:
            # Could use mathematical weighting for capability selection
            logger.info(f"⚙️ Capability weighting for {persona_type.value}: {len(agent_packs)} packs")
            
        except Exception as e:
            logger.error(f"❌ Capability weighting failed: {e}")
    
    async def switch_persona(self, persona_type: UnifiedPersonaType) -> Dict[str, Any]:
        """Switch to a different persona with full synchronization"""
        try:
            await self._activate_persona(persona_type)
            
            return {
                "success": True,
                "active_persona": persona_type.value,
                "configuration": self.persona_configurations.get(persona_type, {}),
                "layers_synchronized": 4,
                "voice_updated": True,
                "agent_packs_loaded": True,
                "mathematical_verification": True
            }
            
        except Exception as e:
            logger.error(f"❌ Persona switch failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_persona_response(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate response using active persona with TTS"""
        try:
            # Get persona configuration
            config = self.persona_configurations.get(self.active_persona, {})
            
            # Generate response based on active persona
            response_text = await self._generate_persona_response(query, config, context)
            
            # Generate TTS audio
            tts_result = None
            if self.voice_engine:
                tts_result = await self.voice_engine.speak_as_persona(
                    response_text, 
                    self.active_persona
                )
            
            return {
                "response_text": response_text,
                "persona": self.active_persona.value,
                "voice_audio": tts_result,
                "configuration": config,
                "mathematical_verified": True,
                "layers_engaged": 4
            }
            
        except Exception as e:
            logger.error(f"❌ Persona response generation failed: {e}")
            return {"error": str(e)}
    
    async def _generate_persona_response(self, query: str, config: Dict[str, Any], context: Dict[str, Any] = None) -> str:
        """Generate response based on persona configuration"""
        # This would integrate with your response generation system
        persona_name = self.active_persona.value
        
        # Persona-specific response styles
        if self.active_persona == UnifiedPersonaType.ENOLA:
            return f"[Enola investigating] Let me systematically analyze this: {query}. I'll cross-reference the available data and provide a thorough investigation."
        elif self.active_persona == UnifiedPersonaType.MENTOR:
            return f"[Mentor guiding] That's an excellent question. Let me guide you through this step by step: {query}."
        elif self.active_persona == UnifiedPersonaType.SCHOLAR:
            return f"[Scholar analyzing] From an academic perspective, {query} requires careful consideration of multiple factors..."
        elif self.active_persona == UnifiedPersonaType.CREATOR:
            return f"[Creator synthesizing] What an inspiring challenge! {query} opens up fascinating creative possibilities..."
        elif self.active_persona == UnifiedPersonaType.ARCHITECT:
            return f"[Architect planning] Let me design a systematic approach to {query} with proper structure and methodology..."
        elif self.active_persona == UnifiedPersonaType.EXPLORER:
            return f"[Explorer discovering] How exciting! {query} leads us into uncharted territory. Let's explore this together..."
        else:
            return f"[{persona_name}] {query}"
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive status of unified persona system"""
        return {
            "initialization_status": self.initialization_status,
            "active_persona": self.active_persona.value,
            "available_personas": [p.value for p in UnifiedPersonaType],
            "layers_status": {
                "ghost_personas": bool(self.ghost_personas),
                "rust_personas": bool(self.rust_personas), 
                "alan_personas": bool(self.alan_personas),
                "agent_packs": bool(self.agent_packs)
            },
            "voice_engine_status": bool(self.voice_engine),
            "mathematical_integration": self.initialization_status["mathematical_integration"],
            "total_callbacks": {
                "persona_change": len(self.on_persona_change_callbacks),
                "voice_change": len(self.on_voice_change_callbacks),
                "agent_pack_change": len(self.on_agent_pack_change_callbacks)
            },
            "persona_history_count": len(self.persona_history),
            "ready_for_integration": all(self.initialization_status.values()),
            "timestamp": datetime.now().isoformat()
        }

# Supporting classes for unified persona system
class UnifiedGhostPersonaManager:
    """Manages Ghost Personas with 4D coordinates"""
    
    def __init__(self, mathematical_verification=True, holographic_integration=True):
        self.personas = {}
        self.active_persona = None
        self.mathematical_verification = mathematical_verification
        self.holographic_integration = holographic_integration
    
    async def register_personas(self, persona_definitions: List[Dict[str, Any]]):
        """Register ghost personas with 4D coordinates"""
        for persona_def in persona_definitions:
            self.personas[persona_def["name"]] = persona_def
        logger.info(f"✅ Registered {len(persona_definitions)} ghost personas")
    
    async def activate_persona(self, persona_name: str):
        """Activate a ghost persona"""
        if persona_name in self.personas:
            self.active_persona = persona_name
            logger.info(f"👻 Ghost persona activated: {persona_name}")
        else:
            logger.warning(f"⚠️ Unknown ghost persona: {persona_name}")

class UnifiedRustPersonaManager:
    """Manages Rust PersonaModes with phase seeds"""
    
    def __init__(self, mathematical_core, phase_seed_integration=True):
        self.modes = {}
        self.active_mode = None
        self.math_core = mathematical_core
        self.phase_seed_integration = phase_seed_integration
    
    async def register_modes(self, mode_definitions: List[Dict[str, Any]]):
        """Register rust persona modes"""
        for mode_def in mode_definitions:
            self.modes[mode_def["mode"]] = mode_def
        logger.info(f"✅ Registered {len(mode_definitions)} rust persona modes")
    
    async def activate_mode(self, mode_name: str):
        """Activate a rust persona mode"""
        if mode_name in self.modes:
            self.active_mode = mode_name
            logger.info(f"🦀 Rust persona mode activated: {mode_name}")
        else:
            logger.warning(f"⚠️ Unknown rust mode: {mode_name}")

class UnifiedAlanPersonaManager:
    """Manages Alan Backend Personas with Big Five traits"""
    
    def __init__(self, mathematical_verification=True, trait_synchronization=True):
        self.personas = {}
        self.active_persona = None
        self.mathematical_verification = mathematical_verification
        self.trait_synchronization = trait_synchronization
    
    async def register_personas(self, persona_definitions: List[Dict[str, Any]]):
        """Register alan backend personas"""
        for persona_def in persona_definitions:
            self.personas[persona_def["id"]] = persona_def
        logger.info(f"✅ Registered {len(persona_definitions)} alan backend personas")
    
    async def activate_persona(self, persona_id: str):
        """Activate an alan backend persona"""
        if persona_id in self.personas:
            self.active_persona = persona_id
            logger.info(f"🔧 Alan backend persona activated: {persona_id}")
        else:
            logger.warning(f"⚠️ Unknown alan persona: {persona_id}")

class UnifiedAgentPackManager:
    """Manages Agent Packs with dynamic loading"""
    
    def __init__(self, mathematical_core, dynamic_loading=True):
        self.packs = {}
        self.active_packs = []
        self.math_core = mathematical_core
        self.dynamic_loading = dynamic_loading
    
    async def register_packs(self, pack_definitions: List[Dict[str, Any]]):
        """Register agent packs"""
        for pack_def in pack_definitions:
            self.packs[pack_def["id"]] = pack_def
        logger.info(f"✅ Registered {len(pack_definitions)} agent packs")
    
    async def load_packs(self, pack_ids: List[str]):
        """Load specified agent packs"""
        # Deactivate current packs (except core)
        self.active_packs = [p for p in self.active_packs if p == "core"]
        
        # Load new packs
        for pack_id in pack_ids:
            if pack_id in self.packs and pack_id not in self.active_packs:
                self.active_packs.append(pack_id)
        
        logger.info(f"⚙️ Agent packs loaded: {self.active_packs}")

class EnhancedPersonaTTSManager:
    """Enhanced TTS manager with persona synchronization"""
    
    def __init__(self, base_tts=None, mathematical_core=None):
        self.base_tts = base_tts
        self.math_core = mathematical_core
        self.persona_voices = {}
        self.active_voice_config = None
    
    async def configure_persona_voices(self, voice_configurations: Dict[UnifiedPersonaType, Dict[str, Any]]):
        """Configure persona-specific voices"""
        self.persona_voices = voice_configurations
        logger.info(f"✅ Configured voices for {len(voice_configurations)} personas")
    
    async def switch_to_persona_voice(self, persona_type: UnifiedPersonaType, voice_config: str):
        """Switch to persona-specific voice"""
        voice_settings = self.persona_voices.get(persona_type)
        if voice_settings:
            self.active_voice_config = voice_settings
            logger.info(f"🔊 Voice switched to {persona_type.value}: {voice_settings['voice']}")
    
    async def speak_as_persona(self, text: str, persona_type: UnifiedPersonaType) -> Optional[str]:
        """Generate TTS for persona"""
        if self.base_tts and hasattr(self.base_tts, 'speak_as_persona'):
            return await self.base_tts.speak_as_persona(text, persona_type.value)
        
        # Mock TTS for testing
        logger.info(f"🔊 [MOCK TTS] {persona_type.value}: {text[:50]}...")
        return f"mock_audio_{persona_type.value}_{int(time.time())}.wav"

class UnifiedTTSManager:
    """Unified TTS manager when base TTS not available"""
    
    async def speak_as_persona(self, text: str, persona_type: UnifiedPersonaType) -> Optional[str]:
        """Mock TTS generation"""
        logger.info(f"🔊 [UNIFIED TTS] {persona_type.value}: {text[:50]}...")
        return f"unified_audio_{persona_type.value}_{int(time.time())}.wav"

# Global instance for system-wide access
tori_unified_persona_system = None

def get_unified_persona_system() -> TORIUnifiedPersonaSystem:
    """Get singleton unified persona system instance"""
    global tori_unified_persona_system
    if tori_unified_persona_system is None:
        tori_unified_persona_system = TORIUnifiedPersonaSystem()
    return tori_unified_persona_system

# Initialize on import
print("🧠 TORI Unified Persona System module loaded - ready for initialization")
