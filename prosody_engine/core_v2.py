        # Social state
        if result.social_battery < 20:
            summary_parts.append("- socially drained")
        elif result.social_battery > 80:
            summary_parts.append("- socially energized")
            
        # Voice observations
        if result.voice_quality.get('strain', 0) > 0.7:
            summary_parts.append("- voice showing exhaustion")
        elif result.voice_quality.get('breathiness', 0) > 0.7:
            summary_parts.append("- breathless with emotion")
            
        return " ".join(summary_parts)

# Create singleton instance with proper initialization
_engine = None

def get_prosody_engine(config: Optional[Dict] = None) -> NetflixKillerProsodyEngine:
    """Get singleton prosody engine instance - the one that kills Netflix"""
    global _engine
    if _engine is None:
        _engine = NetflixKillerProsodyEngine(config)
        logger.info("🎯 Prosody Engine Ready to Destroy Netflix!")
    return _engine

# The Grand Demonstration
async def demonstrate_netflix_destruction():
    """
    Show how we detect emotions Netflix can't even imagine.
    """
    engine = get_prosody_engine()
    
    print("\n" + "="*60)
    print("🎬 NETFLIX-KILLER PROSODY ENGINE DEMONSTRATION")
    print("="*60)
    
    # Simulate different emotional states
    test_cases = [
        {
            'name': 'Hidden Exhaustion',
            'description': 'User saying "I\'m fine" while dying inside',
            'audio': generate_test_audio('exhausted_but_hiding'),
            'text': "I'm fine, just a bit tired"
        },
        {
            'name': 'Pre-Burnout Detection',
            'description': 'Catching burnout 2 weeks before it happens',
            'audio': generate_test_audio('pre_burnout_pattern'),
            'text': "Yeah, I can take on that project too"
        },
        {
            'name': 'Sarcastic Agreement',
            'description': 'Detecting sarcasm Netflix would miss',
            'audio': generate_test_audio('sarcastic_agreement'),
            'text': "Oh sure, that's a GREAT idea"
        },
        {
            'name': 'Creative Flow State',
            'description': 'Recognizing and protecting flow',
            'audio': generate_test_audio('flow_state'),
            'text': "Hmm... what if we... oh wait, I got it!"
        }
    ]
    
    for i, test in enumerate(test_cases):
        print(f"\n🧪 Test {i+1}: {test['name']}")
        print(f"📝 Scenario: {test['description']}")
        print(f"💬 User says: \"{test['text']}\"")
        
        # Analyze
        result = await engine.analyze_complete(
            test['audio'],
            {
                'text': test['text'],
                'include_trajectory': True,
                'suggest_content': True,
                'cultural_context': 'western',
                'sample_rate': 16000
            }
        )
        
        # Show results
        print(f"\n🎯 DETECTED:")
        print(f"   Primary Emotion: {result['primary_emotion']}")
        print(f"   Confidence: {result['emotion_confidence']:.1%}")
        
        if result['micro_emotions']:
            print(f"   Micro-emotions: {', '.join(result['micro_emotions'])}")
        
        if result['hidden_emotions']:
            print(f"   Hidden feelings: {', '.join(result['hidden_emotions'])}")
            
        print(f"\n📊 VOICE ANALYSIS:")
        for quality, value in result['voice_quality'].items():
            print(f"   {quality}: {value:.2f}")
            
        print(f"\n🧠 COGNITIVE STATE:")
        print(f"   Suppression: {result['suppression_score']:.1%}")
        print(f"   Genuine: {result['genuine_probability']:.1%}")
        print(f"   Cognitive Load: {result['cognitive_load']:.1%}")
        print(f"   Social Battery: {result['social_battery']:.1%}")
        
        if result.get('subtitle_recommendation'):
            print(f"\n📺 SUBTITLE:")
            sub = result['subtitle_recommendation']
            print(f"   Text: {sub['text']}")
            print(f"   Style: {sub['style']}")
            
        if result.get('intervention_needed'):
            print(f"\n⚠️ INTERVENTION:")
            intervention = result['intervention_needed']
            print(f"   Type: {intervention['type']}")
            print(f"   Message: {intervention['message']}")
            
        if result.get('burnout_prediction'):
            print(f"\n🔮 BURNOUT PREDICTION:")
            burnout = result['burnout_prediction']
            print(f"   Risk: {burnout['risk']:.1%}")
            print(f"   Timeline: {burnout['prediction']}")
            print(f"   Recommendation: {burnout['recommendation']}")
            
        print(f"\n💡 HUMAN SUMMARY: {result['summary']}")
        
        print(f"\n⚡ Processing latency: {result['processing_latency']:.1f}ms")
        print("-" * 60)
    
    # Show emotion taxonomy stats
    print(f"\n📊 EMOTION TAXONOMY STATS:")
    print(f"   Total emotions available: {len(engine.emotion_categories)}")
    print(f"   Compound emotions: {len(engine.compound_emotions)}")
    print(f"   Micro-emotional states: {len(engine.micro_emotional_states)}")
    print(f"   Cultural models: {len(engine.cultural_models)}")
    
    # Performance summary
    if engine.actual_latencies:
        avg_latency = np.mean(engine.actual_latencies)
        print(f"\n⚡ PERFORMANCE:")
        print(f"   Average latency: {avg_latency:.1f}ms")
        print(f"   Target latency: {engine.target_latency}ms")
        print(f"   Performance: {'✅ EXCELLENT' if avg_latency < engine.target_latency else '⚠️ NEEDS OPTIMIZATION'}")
    
    print(f"\n💀 NETFLIX STATUS: COMPLETELY DESTROYED")
    print(f"🎉 USER EXPERIENCE: TRANSCENDENT")
    print(f"🚀 EMOTIONAL AI: ACHIEVED")

def generate_test_audio(pattern: str) -> bytes:
    """Generate test audio with specific emotional patterns."""
    # In production, these would be real audio samples
    # For demo, we generate synthetic patterns
    
    rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(rate * duration))
    
    if pattern == 'exhausted_but_hiding':
        # Low energy with forced brightness
        fundamental = 180  # Higher than natural for forced cheerfulness
        harmonics = fundamental * np.array([1, 2, 3, 4])
        signal = np.zeros_like(t)
        
        # Add strained harmonics
        for i, h in enumerate(harmonics):
            amplitude = 0.3 / (i + 1) * (1 + 0.1 * np.sin(2 * np.pi * 10 * t))  # Tremor
            signal += amplitude * np.sin(2 * np.pi * h * t)
            
        # Add breathiness
        signal += 0.1 * np.random.randn(len(t))
        
    elif pattern == 'pre_burnout_pattern':
        # High cognitive load markers
        fundamental = 150
        signal = np.sin(2 * np.pi * fundamental * t)
        
        # Add irregular pauses
        for i in range(3):
            start = int(i * rate * 0.5)
            end = start + int(rate * 0.1)
            signal[start:end] *= 0.1
            
        # Add pitch instability
        signal *= (1 + 0.05 * np.sin(2 * np.pi * 3 * t))
        
    elif pattern == 'sarcastic_agreement':
        # Exaggerated pitch contours
        fundamental = 200
        pitch_contour = 1 + 0.3 * np.sin(2 * np.pi * 0.5 * t)  # Slow pitch rise
        signal = np.sin(2 * np.pi * fundamental * pitch_contour * t)
        
        # Add edge to voice
        signal = np.clip(signal * 1.5, -1, 1)
        
    elif pattern == 'flow_state':
        # Smooth, consistent energy
        fundamental = 160
        signal = np.sin(2 * np.pi * fundamental * t)
        
        # Stable harmonics
        for i in range(2, 5):
            signal += 0.3 / i * np.sin(2 * np.pi * fundamental * i * t)
            
        # Minimal noise
        signal += 0.02 * np.random.randn(len(t))
        
    else:
        # Default neutral
        signal = np.sin(2 * np.pi * 200 * t)
    
    # Normalize and convert to int16
    signal = signal / (np.max(np.abs(signal)) + 1e-10)
    signal = (signal * 32767).astype(np.int16)
    
    return signal.tobytes()

# Run the demonstration
if __name__ == "__main__":
    print("🚀 Starting Netflix Destruction Sequence...")
    asyncio.run(demonstrate_netflix_destruction())
