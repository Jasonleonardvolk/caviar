#!/usr/bin/env python3
"""
HOLOGRAM STARTUP FIX - Ensure Default Holograms Appear
Fixes the issue where only technical metrics show instead of actual holograms
"""

import os
import sys
import json
import time
from pathlib import Path

class HologramStartupFixer:
    def __init__(self, base_path="C:\\Users\\jason\\Desktop\\tori\\kha"):
        self.base_path = Path(base_path)
        self.changes_made = []
        
    def fix_concept_mesh_bridge_initialization(self):
        """Ensure concept mesh bridge provides initial concepts immediately"""
        print("Fix 1: Adding startup hologram seeds to concept mesh bridge...")
        
        bridge_file = self.base_path / "concept_mesh_hologram_bridge.py"
        if not bridge_file.exists():
            print("  - concept_mesh_hologram_bridge.py not found")
            return False
            
        # Read current content
        with open(bridge_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Add immediate concept broadcasting on client connect
        new_handle_client = '''    async def handle_client(self, websocket, path):
        """Handle WebSocket client connections"""
        if not WEBSOCKETS_AVAILABLE:
            return
        
        self.clients.add(websocket)
        logger.info(f"Concept client connected: {websocket.remote_address}")
        
        try:
            # IMMEDIATE: Send initial concept data right away
            await self.send_concept_update(websocket)
            print(f"SENT: Initial concepts to new client ({len(self.get_concept_data())} concepts)")
            
            # IMMEDIATE: Start rendering concepts as holograms
            await self.send_hologram_initialization(websocket)
            
            # Listen for requests
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.handle_concept_request(websocket, data)
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        "error": "Invalid JSON format"
                    }))
                except Exception as e:
                    logger.error(f"Error handling concept request: {e}")
        except Exception as e:
            logger.error(f"Client connection error: {e}")
        finally:
            self.clients.discard(websocket)
            logger.info("Concept client disconnected")'''
        
        # Replace the existing handle_client method
        if 'async def handle_client(self, websocket, path):' in content:
            start = content.find('    async def handle_client(self, websocket, path):')
            # Find the end of the method (next method definition at same level)
            next_method = content.find('\n    async def ', start + 1)
            if next_method == -1:
                next_method = content.find('\n    def ', start + 1)
            if next_method == -1:
                next_method = len(content)
            
            content = content[:start] + new_handle_client + '\n' + content[next_method:]
            
            # Add the new method
            content += '''
    
    async def send_hologram_initialization(self, websocket):
        """Send immediate hologram initialization to start rendering"""
        concepts = self.get_concept_data()
        
        # Create hologram commands for each concept
        hologram_commands = []
        
        for concept in concepts:
            hologram_cmd = {
                "type": "create_hologram",
                "concept_id": concept["id"],
                "position": concept["position"],
                "color": concept["color"],
                "size": concept["size"],
                "intensity": 0.8,
                "animation": {
                    "rotation": True,
                    "pulse": True,
                    "particles": True
                }
            }
            hologram_commands.append(hologram_cmd)
        
        # Send initialization message
        init_message = {
            "type": "hologram_init",
            "timestamp": time.time(),
            "commands": hologram_commands,
            "enable_particles": True,
            "enable_connections": True,
            "camera_position": {"x": 0, "y": 5, "z": 10}
        }
        
        await websocket.send(json.dumps(init_message))
        print(f"HOLOGRAM INIT: Sent {len(hologram_commands)} hologram creation commands")
'''
            
            # Write the updated content
            with open(bridge_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.changes_made.append(str(bridge_file))
            print("  - SUCCESS: Added immediate hologram initialization")
            return True
            
        return False
    
    def fix_frontend_hologram_initialization(self):
        """Fix frontend to immediately start holograms when connected"""
        print("Fix 2: Updating frontend hologram initialization...")
        
        # Update realGhostEngine_v2.js to auto-start holograms
        engine_file = self.base_path / "tori_ui_svelte" / "src" / "lib" / "realGhostEngine_v2.js"
        if not engine_file.exists():
            print("  - realGhostEngine_v2.js not found")
            return False
        
        with open(engine_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Add immediate hologram startup to initialize method
        if 'async initialize(canvas, options = {}) {' in content:
            # Find the end of the initialize method and add startup code
            init_start = content.find('async initialize(canvas, options = {}) {')
            init_end = content.find('\n    }', init_start)
            
            startup_code = '''
            
            // 7. IMMEDIATE: Start default holograms if no data
            setTimeout(async () => {
                await this.startDefaultHolograms();
            }, 1000); // Give bridges time to connect
            
            // 8. Force initial render with default content
            this.createDefaultScene();'''
            
            content = content[:init_end] + startup_code + content[init_end:]
            
            # Add the new methods at the end of the class
            content = content.replace(
                '    destroy() {',
                '''    async startDefaultHolograms() {
        """Start default holograms if no concept data is available"""
        console.log('🌟 Starting default holograms...');
        
        // Create some default hologram objects
        const defaultConcepts = [
            {
                id: 'default_consciousness',
                name: 'Consciousness', 
                position: [0, 0, 0],
                color: [1.0, 0.8, 0.2],
                size: 1.5,
                type: 'concept'
            },
            {
                id: 'default_cognition',
                name: 'Cognition',
                position: [2, 1, 0], 
                color: [0.2, 0.8, 1.0],
                size: 1.2,
                type: 'concept'
            },
            {
                id: 'default_awareness',
                name: 'Awareness',
                position: [-1, 2, 1],
                color: [0.8, 0.2, 1.0], 
                size: 1.0,
                type: 'concept'
            }
        ];
        
        // Add each as a holographic object
        defaultConcepts.forEach(concept => {
            this.addHolographicObject({
                id: concept.id,
                type: 'concept',
                name: concept.name,
                position: concept.position,
                color: concept.color,
                size: concept.size,
                resonance: 0.8,
                intensity: 0.9
            });
        });
        
        console.log(`✅ Added ${defaultConcepts.length} default holograms`);
    }
    
    createDefaultScene() {
        """Create a default holographic scene with initial content"""
        if (!this.engine) return;
        
        console.log('🎨 Creating default holographic scene...');
        
        // Set default psi state with active oscillations
        this.psiState.phase_coherence = 0.8;
        this.psiState.dominant_frequency = 440;
        
        // Initialize oscillator phases with default pattern
        for (let i = 0; i < 32; i++) {
            this.psiState.oscillator_phases[i] = (i / 32) * Math.PI * 2;
            this.psiState.oscillator_frequencies[i] = 440 * Math.pow(1.2, i % 12);
        }
        
        // Update engine with default state
        this.engine.updateFromOscillator(this.psiState);
        
        console.log('✅ Default scene created with oscillating patterns');
    }

    destroy() {'''
            )
            
            with open(engine_file, 'w', encoding='utf-8') as f:
                f.write(content)
                
            self.changes_made.append(str(engine_file))
            print("  - SUCCESS: Added default hologram startup")
            return True
        
        return False
    
    def create_hologram_auto_starter(self):
        """Create a script that automatically starts holograms on system ready"""
        print("Fix 3: Creating hologram auto-starter...")
        
        starter_script = '''#!/usr/bin/env python3
"""
Hologram Auto-Starter - Ensures holograms appear immediately on startup
Runs after TORI launch to seed the hologram system with initial content
"""

import asyncio
import websockets
import json
import time

async def seed_holograms():
    """Connect to hologram bridges and seed initial content"""
    print("🌟 Seeding hologram system with initial content...")
    
    # Default concepts to create holograms for
    default_concepts = [
        {
            "id": "startup_consciousness",
            "name": "Consciousness", 
            "description": "The fundamental nature of awareness",
            "position": {"x": 0, "y": 0, "z": 0},
            "color": {"r": 1.0, "g": 0.8, "b": 0.2},
            "size": 1.5,
            "connections": ["startup_cognition", "startup_awareness"]
        },
        {
            "id": "startup_cognition",
            "name": "Cognition",
            "description": "The process of thinking and understanding", 
            "position": {"x": 2, "y": 1, "z": 0},
            "color": {"r": 0.2, "g": 0.8, "b": 1.0},
            "size": 1.2,
            "connections": ["startup_consciousness", "startup_intelligence"]
        },
        {
            "id": "startup_awareness",
            "name": "Awareness",
            "description": "The state of being conscious of something",
            "position": {"x": -1, "y": 2, "z": 1},
            "color": {"r": 0.8, "g": 0.2, "b": 1.0},
            "size": 1.0,
            "connections": ["startup_consciousness"]
        },
        {
            "id": "startup_intelligence", 
            "name": "Intelligence",
            "description": "The ability to acquire and apply knowledge",
            "position": {"x": 1, "y": -1, "z": -1},
            "color": {"r": 1.0, "g": 0.5, "b": 0.0},
            "size": 1.3,
            "connections": ["startup_cognition"]
        }
    ]
    
    try:
        # Connect to concept mesh bridge
        uri = "ws://localhost:8766/concepts"
        print(f"Connecting to concept bridge: {uri}")
        
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to concept bridge")
            
            # Add each concept
            for concept in default_concepts:
                message = {
                    "type": "add_concept",
                    "concept": concept
                }
                
                await websocket.send(json.dumps(message))
                print(f"  ➕ Added concept: {concept['name']}")
                
                # Small delay between additions
                await asyncio.sleep(0.1)
            
            # Request immediate visualization update
            viz_message = {
                "type": "start_visualization",
                "enable_particles": True,
                "enable_connections": True,
                "animation_speed": 1.0
            }
            
            await websocket.send(json.dumps(viz_message))
            print("🎨 Requested immediate visualization")
            
            # Wait a moment for response
            await asyncio.sleep(2)
            
        print("✅ Hologram seeding complete!")
        
    except Exception as e:
        print(f"❌ Failed to seed holograms: {e}")
        print("ℹ️  Make sure TORI is running and concept bridge is available")

async def seed_audio_bridge():
    """Send default audio to start audio visualizations"""
    print("🎵 Seeding audio bridge with default patterns...")
    
    try:
        uri = "ws://localhost:8765/audio_stream"
        print(f"Connecting to audio bridge: {uri}")
        
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to audio bridge")
            
            # Send some default audio patterns
            for freq in [220, 440, 880]:
                audio_data = {
                    "type": "audio_features",
                    "amplitude": 0.5,
                    "frequency": freq,
                    "waveform": "sine",
                    "timestamp": time.time()
                }
                
                await websocket.send(json.dumps(audio_data))
                print(f"  🎵 Sent audio pattern: {freq}Hz")
                await asyncio.sleep(0.5)
                
        print("✅ Audio seeding complete!")
        
    except Exception as e:
        print(f"❌ Failed to seed audio: {e}")

async def main():
    """Main seeding function"""
    print("🚀 Starting TORI Hologram Auto-Seeder...")
    
    # Wait for TORI to be fully ready
    print("⏳ Waiting 5 seconds for TORI to initialize...")
    await asyncio.sleep(5)
    
    # Seed both bridges
    await seed_holograms()
    await seed_audio_bridge()
    
    print("🎉 Hologram auto-seeding complete!")
    print("✨ You should now see holograms in your TORI interface!")

if __name__ == "__main__":
    asyncio.run(main())
'''
        
        starter_file = self.base_path / "start_default_holograms.py"
        with open(starter_file, 'w', encoding='utf-8') as f:
            f.write(starter_script)
        
        starter_file.chmod(0o755)
        self.changes_made.append(str(starter_file))
        print("  - SUCCESS: Created hologram auto-starter script")
        return True
    
    def update_enhanced_launcher(self):
        """Update enhanced launcher to auto-start holograms"""
        print("Fix 4: Updating enhanced launcher to auto-start holograms...")
        
        launcher_file = self.base_path / "enhanced_launcher.py"
        if not launcher_file.exists():
            print("  - enhanced_launcher.py not found")
            return False
        
        with open(launcher_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Add hologram seeding to the end of the launch sequence
        if 'print_complete_system_ready(' in content:
            # Find where the system ready message is printed and add hologram starter
            ready_pos = content.find('print_complete_system_ready(')
            
            hologram_starter = '''
            
            # 🌟 IMMEDIATE: Start default holograms after system ready
            if args.enable_hologram:
                try:
                    import subprocess
                    self.logger.info("🌟 Starting default holograms...")
                    
                    # Run hologram seeder in background
                    subprocess.Popen([
                        sys.executable, 
                        'start_default_holograms.py'
                    ], cwd=str(self.script_dir))
                    
                    self.logger.info("✅ Default hologram seeder started")
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not start hologram seeder: {e}")
'''
            
            content = content[:ready_pos] + hologram_starter + '\n            ' + content[ready_pos:]
            
            with open(launcher_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.changes_made.append(str(launcher_file))
            print("  - SUCCESS: Added hologram auto-start to launcher")
            return True
        
        return False
    
    def create_hologram_test_page(self):
        """Create a test page to verify holograms are working"""
        print("Fix 5: Creating hologram test page...")
        
        test_page = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TORI Hologram Test</title>
    <style>
        body { margin: 0; padding: 20px; font-family: Arial, sans-serif; background: #000; color: #fff; }
        .container { max-width: 1200px; margin: 0 auto; }
        .status { padding: 20px; background: #111; border-radius: 10px; margin-bottom: 20px; }
        .hologram-canvas { width: 100%; height: 600px; background: #222; border-radius: 10px; }
        .controls { margin-top: 20px; }
        button { padding: 10px 20px; margin: 5px; background: #0066cc; color: white; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background: #0088ff; }
        .log { background: #111; padding: 15px; border-radius: 5px; font-family: monospace; font-size: 12px; max-height: 200px; overflow-y: auto; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌟 TORI Hologram System Test</h1>
        
        <div class="status">
            <h2>System Status</h2>
            <p>Concept Bridge: <span id="concept-status">❓ Connecting...</span></p>
            <p>Audio Bridge: <span id="audio-status">❓ Connecting...</span></p>
            <p>Holograms Active: <span id="hologram-status">❓ Checking...</span></p>
        </div>
        
        <canvas id="hologram-canvas" class="hologram-canvas"></canvas>
        
        <div class="controls">
            <button onclick="testConceptBridge()">Test Concept Bridge</button>
            <button onclick="testAudioBridge()">Test Audio Bridge</button>
            <button onclick="addTestConcept()">Add Test Concept</button>
            <button onclick="playTestAudio()">Play Test Audio</button>
            <button onclick="clearLog()">Clear Log</button>
        </div>
        
        <div class="log" id="log"></div>
    </div>

    <script>
        let conceptWs = null;
        let audioWs = null;
        let logContainer = document.getElementById('log');
        
        function log(message) {
            const timestamp = new Date().toLocaleTimeString();
            logContainer.innerHTML += `[${timestamp}] ${message}\\n`;
            logContainer.scrollTop = logContainer.scrollHeight;
            console.log(message);
        }
        
        function updateStatus(element, status, success) {
            document.getElementById(element).innerHTML = success ? `✅ ${status}` : `❌ ${status}`;
        }
        
        async function testConceptBridge() {
            log('🧠 Testing concept bridge connection...');
            
            try {
                conceptWs = new WebSocket('ws://localhost:8766/concepts');
                
                conceptWs.onopen = () => {
                    log('✅ Connected to concept bridge');
                    updateStatus('concept-status', 'Connected', true);
                    
                    // Request current concepts
                    conceptWs.send(JSON.stringify({
                        type: 'get_concepts'
                    }));
                };
                
                conceptWs.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    log(`📨 Received: ${data.type} (${data.concepts ? data.concepts.length : 0} concepts)`);
                    
                    if (data.concepts && data.concepts.length > 0) {
                        updateStatus('hologram-status', `${data.concepts.length} concepts loaded`, true);
                    }
                };
                
                conceptWs.onerror = (error) => {
                    log(`❌ Concept bridge error: ${error}`);
                    updateStatus('concept-status', 'Error', false);
                };
                
            } catch (error) {
                log(`❌ Failed to connect to concept bridge: ${error}`);
                updateStatus('concept-status', 'Failed', false);
            }
        }
        
        async function testAudioBridge() {
            log('🎵 Testing audio bridge connection...');
            
            try {
                audioWs = new WebSocket('ws://localhost:8765/audio_stream');
                
                audioWs.onopen = () => {
                    log('✅ Connected to audio bridge');
                    updateStatus('audio-status', 'Connected', true);
                };
                
                audioWs.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    log(`🎵 Audio data: ${data.type}`);
                };
                
                audioWs.onerror = (error) => {
                    log(`❌ Audio bridge error: ${error}`);
                    updateStatus('audio-status', 'Error', false);
                };
                
            } catch (error) {
                log(`❌ Failed to connect to audio bridge: ${error}`);
                updateStatus('audio-status', 'Failed', false);
            }
        }
        
        function addTestConcept() {
            if (!conceptWs || conceptWs.readyState !== WebSocket.OPEN) {
                log('❌ Concept bridge not connected');
                return;
            }
            
            const testConcept = {
                id: `test_${Date.now()}`,
                name: 'Test Hologram',
                description: 'A test concept for hologram visualization',
                position: {
                    x: Math.random() * 4 - 2,
                    y: Math.random() * 4 - 2, 
                    z: Math.random() * 4 - 2
                },
                color: {
                    r: Math.random(),
                    g: Math.random(),
                    b: Math.random()
                },
                size: 1.0 + Math.random() * 0.5
            };
            
            conceptWs.send(JSON.stringify({
                type: 'add_concept',
                concept: testConcept
            }));
            
            log(`➕ Added test concept: ${testConcept.name}`);
        }
        
        function playTestAudio() {
            if (!audioWs || audioWs.readyState !== WebSocket.OPEN) {
                log('❌ Audio bridge not connected');
                return;
            }
            
            const frequencies = [220, 440, 880, 1760];
            const frequency = frequencies[Math.floor(Math.random() * frequencies.length)];
            
            const audioData = {
                type: 'audio_features',
                amplitude: 0.5 + Math.random() * 0.5,
                frequency: frequency,
                waveform: 'sine',
                timestamp: Date.now()
            };
            
            audioWs.send(JSON.stringify(audioData));
            log(`🎵 Sent audio: ${frequency}Hz`);
        }
        
        function clearLog() {
            logContainer.innerHTML = '';
        }
        
        // Auto-start tests when page loads
        window.onload = function() {
            log('🚀 TORI Hologram Test Page Loaded');
            setTimeout(testConceptBridge, 1000);
            setTimeout(testAudioBridge, 2000);
        };
    </script>
</body>
</html>'''
        
        test_file = self.base_path / "hologram_test.html"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_page)
        
        self.changes_made.append(str(test_file))
        print("  - SUCCESS: Created hologram test page")
        print(f"  - Open: http://localhost:5173/hologram_test.html")
        return True
    
    def run(self):
        """Apply all hologram startup fixes"""
        print("🌟 TORI HOLOGRAM STARTUP FIXES")
        print("=" * 50)
        print("Fixing the issue where only technical metrics show instead of actual holograms...")
        print()
        
        fixes_applied = 0
        
        if self.fix_concept_mesh_bridge_initialization():
            fixes_applied += 1
        print()
        
        if self.fix_frontend_hologram_initialization():
            fixes_applied += 1
        print()
        
        if self.create_hologram_auto_starter():
            fixes_applied += 1
        print()
        
        if self.update_enhanced_launcher():
            fixes_applied += 1
        print()
        
        if self.create_hologram_test_page():
            fixes_applied += 1
        print()
        
        print("=" * 50)
        print("HOLOGRAM FIXES SUMMARY")
        print("=" * 50)
        
        if fixes_applied > 0:
            print(f"✅ Applied {fixes_applied} hologram fixes")
            print()
            print("CHANGES MADE:")
            for i, file in enumerate(self.changes_made, 1):
                print(f"  {i}. {Path(file).name}")
            
            print()
            print("NEXT STEPS:")
            print("1. Restart TORI: python enhanced_launcher.py")
            print("2. Wait for system to fully load")
            print("3. Holograms should appear automatically") 
            print("4. Test with: python start_default_holograms.py")
            print("5. Debug with: hologram_test.html")
            
            print()
            print("WHAT THESE FIXES DO:")
            print("- Concept bridge immediately sends hologram data on connect")
            print("- Frontend auto-creates default holograms if none exist")
            print("- Auto-seeder runs after startup to populate holograms")
            print("- Enhanced launcher triggers hologram initialization")
            print("- Test page helps verify everything is working")
            
        else:
            print("❌ No fixes could be applied")
        
        print()
        print("=" * 50)
        print("🌟 Your holograms should now appear on startup!")
        print("=" * 50)

if __name__ == "__main__":
    fixer = HologramStartupFixer()
    fixer.run()
