"""
EXPANDED EMOTION TAXONOMY - 2000+ REAL EMOTIONS
===============================================

The complete emotional spectrum that makes Netflix's "happy/sad" look like cave paintings.
"""

# Base emotional dimensions
BASE_EMOTIONS = [
    # Primary emotions (12)
    'joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust',
    'trust', 'anticipation', 'love', 'contempt', 'pride', 'shame',
    
    # Secondary emotions (24)
    'excitement', 'delight', 'sorrow', 'aversion', 'hesitation', 
    'depression', 'helplessness', 'confusion', 'admiration', 'anxious',
    'bitter', 'aggrieved', 'contentment', 'guilt', 'envy', 'gratitude',
    'hope', 'despair', 'curiosity', 'boredom', 'awe', 'nostalgia',
    'melancholy', 'euphoria',
    
    # Tertiary emotions (36)
    'wistful', 'yearning', 'longing', 'pensive', 'brooding', 'moody',
    'irritated', 'frustrated', 'exasperated', 'indignant', 'outraged', 'furious',
    'worried', 'nervous', 'apprehensive', 'terrified', 'panicked', 'horrified',
    'amused', 'gleeful', 'elated', 'jubilant', 'ecstatic', 'blissful',
    'disappointed', 'discouraged', 'dejected', 'despondent', 'miserable', 'grief-stricken',
    'confident', 'secure', 'empowered', 'determined', 'motivated', 'inspired',
    
    # Social emotions (24)
    'embarrassed', 'humiliated', 'mortified', 'vindicated', 'validated', 'understood',
    'isolated', 'lonely', 'abandoned', 'connected', 'belonging', 'accepted',
    'jealous', 'resentful', 'bitter', 'forgiving', 'compassionate', 'empathetic',
    'dominant', 'submissive', 'competitive', 'cooperative', 'supportive', 'protective',
    
    # Complex cognitive emotions (18)
    'ambivalent', 'conflicted', 'torn', 'decisive', 'certain', 'doubtful',
    'skeptical', 'cynical', 'optimistic', 'pessimistic', 'realistic', 'idealistic',
    'overwhelmed', 'underwhelmed', 'focused', 'scattered', 'present', 'dissociated'
]

# Intensity modifiers (10 levels)
INTENSITIES = [
    'barely', 'slightly', 'mildly', 'moderately', 'notably',
    'significantly', 'strongly', 'intensely', 'extremely', 'overwhelmingly'
]

# Authenticity contexts (8 types)
CONTEXTS = [
    'genuine',      # True feeling
    'performed',    # Acting/faking
    'suppressed',   # Hiding it
    'masked',       # Covering with another emotion
    'conflicted',   # Mixed with opposite
    'emerging',     # Just starting to feel
    'fading',       # Diminishing
    'cyclical'      # Coming in waves
]

# Temporal aspects (6 types)
TEMPORAL = [
    'momentary',    # Flash emotion
    'lingering',    # Hanging around
    'persistent',   # Won't go away
    'recurring',    # Keeps coming back
    'building',     # Growing stronger
    'releasing'     # Finally letting go
]

# Social contexts (5 types)  
SOCIAL = [
    'private',      # Felt alone
    'public',       # In social setting
    'intimate',     # With close person
    'professional', # Work context
    'performative'  # For audience
]

def generate_full_emotion_taxonomy():
    """
    Generate 2000+ unique emotional states.
    
    Formula: base × intensity × context × temporal × social = 114 × 10 × 8 × 6 × 5 = 27,360 possible states
    We'll use the most meaningful 2000+
    """
    emotions = {}
    
    for base in BASE_EMOTIONS:
        for intensity in INTENSITIES:
            for context in CONTEXTS:
                # Skip nonsensical combinations
                if (base in ['joy', 'excitement', 'love'] and context == 'suppressed' and intensity == 'overwhelmingly'):
                    continue  # Overwhelmingly suppressed joy is rare
                
                if (base in ['grief-stricken', 'terrified'] and intensity == 'barely'):
                    continue  # Barely grief-stricken doesn't make sense
                
                # Create emotion name
                emotion_name = f"{base}_{intensity}_{context}"
                
                # Create unique embedding based on components
                base_idx = BASE_EMOTIONS.index(base)
                intensity_idx = INTENSITIES.index(intensity)
                context_idx = CONTEXTS.index(context)
                
                # Generate embedding that encodes the emotion
                embedding = generate_emotion_embedding(base_idx, intensity_idx, context_idx)
                
                emotions[emotion_name] = {
                    'embedding': embedding,
                    'base': base,
                    'intensity': intensity_idx / 10.0,
                    'context': context,
                    'valence': get_valence(base),  # Positive/negative
                    'arousal': get_arousal(base, intensity),  # High/low energy
                    'dominance': get_dominance(base)  # Control level
                }
    
    print(f"Generated {len(emotions)} unique emotional states!")
    return emotions

def generate_emotion_embedding(base_idx, intensity_idx, context_idx):
    """Generate a unique 256-dimensional embedding for each emotion."""
    import numpy as np
    
    # Start with base emotion direction
    embedding = np.zeros(256)
    
    # Base emotion gets primary dimensions
    embedding[base_idx*2:(base_idx+1)*2] = 1.0
    
    # Intensity affects magnitude
    embedding *= (intensity_idx + 1) / 10.0
    
    # Context adds phase shift
    phase = context_idx * np.pi / 4
    embedding = np.roll(embedding, context_idx * 10)
    
    # Add unique signature
    embedding += np.random.randn(256) * 0.1
    
    # Normalize
    embedding = embedding / (np.linalg.norm(embedding) + 1e-10)
    
    return embedding

def get_valence(emotion):
    """Get positive/negative valence of emotion."""
    positive = ['joy', 'love', 'excitement', 'delight', 'contentment', 'gratitude',
                'hope', 'awe', 'confident', 'inspired', 'amused', 'gleeful']
    negative = ['sadness', 'anger', 'fear', 'disgust', 'shame', 'guilt', 'envy',
                'despair', 'anxious', 'bitter', 'grief-stricken', 'terrified']
    
    if emotion in positive:
        return 1.0
    elif emotion in negative:
        return -1.0
    else:
        return 0.0

def get_arousal(emotion, intensity):
    """Get arousal level (energy) of emotion."""
    high_arousal = ['excitement', 'anger', 'fear', 'panic', 'ecstatic', 'furious']
    low_arousal = ['sadness', 'contentment', 'boredom', 'melancholy', 'secure']
    
    base_arousal = 0.5
    if emotion in high_arousal:
        base_arousal = 0.8
    elif emotion in low_arousal:
        base_arousal = 0.2
    
    # Intensity modifies arousal
    return base_arousal + (intensity * 0.2)

def get_dominance(emotion):
    """Get dominance/control level of emotion."""
    high_dominance = ['anger', 'confident', 'proud', 'determined', 'dominant']
    low_dominance = ['fear', 'shame', 'helpless', 'submissive', 'anxious']
    
    if emotion in high_dominance:
        return 0.8
    elif emotion in low_dominance:
        return 0.2
    else:
        return 0.5

# Special compound emotions that Netflix can't even imagine
COMPOUND_EMOTIONS = {
    'freudenfreude': 'joy_at_others_joy',
    'schadenfreude': 'pleasure_from_others_misfortune',
    'mudita': 'sympathetic_joy',
    'saudade': 'bittersweet_longing_for_absent',
    'hygge': 'cozy_intimate_contentment',
    'ikigai': 'life_purpose_satisfaction',
    'weltschmerz': 'world_weariness',
    'torschlusspanik': 'fear_of_diminishing_opportunities',
    'kaukokaipuu': 'longing_for_far_away_places',
    'gigil': 'overwhelming_urge_to_squeeze_cute',
    'kilig': 'butterflies_from_romantic_experience',
    'tartle': 'hesitation_forgetting_name',
    'iktsuarpok': 'anticipation_checking_for_arrival',
    'komorebi': 'peaceful_from_light_through_leaves',
    'tsundoku': 'guilt_from_unread_books',
    'mono_no_aware': 'bittersweet_awareness_of_impermanence',
    'fernweh': 'ache_for_distant_places',
    'sehnsucht': 'inconsolable_longing',
    'hiraeth': 'homesickness_for_place_never_was',
    'mamihlapinatapai': 'shared_look_both_want_neither_initiates'
}

# Emotion transitions that reveal deeper states
REVEALING_TRANSITIONS = {
    'joy_to_sadness': 'bittersweet_realization',
    'anger_to_tears': 'hurt_beneath_rage',
    'laughter_to_silence': 'defense_mechanism_failing',
    'confidence_to_hesitation': 'imposter_syndrome_emerging',
    'calm_to_panic': 'triggered_trauma_response',
    'excitement_to_exhaustion': 'burnout_moment',
    'focus_to_dissociation': 'overwhelm_protection',
    'social_to_withdrawn': 'social_battery_depleted'
}

# Netflix can only dream of detecting these
MICRO_EMOTIONAL_STATES = {
    'pre_cry_throat_tightness': 'About to break down',
    'post_laugh_vulnerability': 'Opened up too much',
    'decision_paralysis_loop': 'Stuck in options',
    'creative_breakthrough_pending': '5 minutes from eureka',
    'social_mask_slipping': 'Can't pretend anymore',
    'love_realization_dawn': 'Just figured it out',
    'trust_rebuilding_tender': 'Trying again carefully',
    'grief_wave_incoming': 'Memory just hit',
    'pride_mixed_with_fear': 'Success brings pressure',
    'connection_spark': 'Found my people'
}

if __name__ == "__main__":
    # Generate the full taxonomy
    full_emotions = generate_full_emotion_taxonomy()
    print(f"\n🎭 Emotion types Netflix has: ~7")
    print(f"🧠 Emotion types we have: {len(full_emotions)}")
    print(f"💀 Netflix status: CRYING")
