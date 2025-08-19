#!/usr/bin/env python3
"""
Fixes for soliton_memory_integration.py
- Fix memory fusion to remove orphan oscillators
- Implement memory fission logic
- Improve heat and migration logic
"""

# Fix 1: Update _perform_memory_fusion to remove orphan oscillators
def _perform_memory_fusion(self) -> int:
    """Fuse similar memories and clean up oscillators"""
    fused = 0
    concept_groups = defaultdict(list)
    
    # Group by primary concept
    for mem_id, entry in self.memory_entries.items():
        if entry.concept_ids:
            concept_groups[entry.concept_ids[0]].append((mem_id, entry))
    
    # Get global lattice for oscillator management
    lattice = get_global_lattice()
    
    # Fuse duplicates
    for concept, memories in concept_groups.items():
        if len(memories) > 1:
            # Sort by amplitude
            memories.sort(key=lambda x: x[1].amplitude, reverse=True)
            main_id, main_entry = memories[0]
            
            # Merge others into main
            for other_id, other_entry in memories[1:]:
                if self._should_fuse(main_entry, other_entry):
                    # Combine amplitudes
                    main_entry.amplitude = min(2.0, 
                        np.sqrt(main_entry.amplitude**2 + other_entry.amplitude**2))
                    
                    # Merge heat
                    main_entry.heat = max(main_entry.heat, other_entry.heat)
                    
                    # Remove oscillator(s) for the fused memory
                    if "oscillator_idx" in other_entry.metadata:
                        idx = other_entry.metadata["oscillator_idx"]
                        if idx < len(lattice.oscillators):
                            lattice.oscillators[idx]["active"] = False
                    
                    # For dark solitons, also remove baseline oscillator
                    if other_entry.polarity == "dark" and "baseline_idx" in other_entry.metadata:
                        baseline_idx = other_entry.metadata["baseline_idx"]
                        if baseline_idx < len(lattice.oscillators):
                            lattice.oscillators[baseline_idx]["active"] = False
                    
                    # Remove the memory entry
                    self.memory_entries.pop(other_id, None)
                    fused += 1
    
    return fused

# Fix 2: Implement _perform_memory_fission
def _perform_memory_fission(self) -> int:
    """Split complex memories into smaller ones"""
    split = 0
    lattice = get_global_lattice()
    
    # Find memories that should be split
    memories_to_split = []
    for mem_id, entry in self.memory_entries.items():
        if self._should_split(entry):
            memories_to_split.append((mem_id, entry))
    
    for mem_id, entry in memories_to_split:
        # Split criteria: high amplitude or very long content
        if entry.amplitude > 1.5 or len(entry.content) > 1000:
            # Create two new memories from the original
            content_mid = len(entry.content) // 2
            
            # First half
            new_entry1 = SolitonMemoryEntry(
                id=self._generate_memory_id(entry.content[:content_mid], entry.concept_ids),
                content=entry.content[:content_mid],
                memory_type=entry.memory_type,
                phase=entry.phase,
                amplitude=entry.amplitude * 0.6,  # Split amplitude
                frequency=entry.frequency,
                timestamp=datetime.now(timezone.utc),
                concept_ids=entry.concept_ids,
                sources=entry.sources,
                metadata={"split_from": mem_id}
            )
            
            # Second half
            new_entry2 = SolitonMemoryEntry(
                id=self._generate_memory_id(entry.content[content_mid:], entry.concept_ids),
                content=entry.content[content_mid:],
                memory_type=entry.memory_type,
                phase=(entry.phase + np.pi/4) % (2 * np.pi),  # Slightly different phase
                amplitude=entry.amplitude * 0.6,
                frequency=entry.frequency,
                timestamp=datetime.now(timezone.utc),
                concept_ids=entry.concept_ids,
                sources=entry.sources,
                metadata={"split_from": mem_id}
            )
            
            # Add new oscillators for split memories
            osc_idx1 = lattice.add_oscillator(
                phase=new_entry1.phase,
                natural_freq=new_entry1.frequency * 0.1,
                amplitude=new_entry1.amplitude,
                stability=0.8
            )
            new_entry1.metadata["oscillator_idx"] = osc_idx1
            
            osc_idx2 = lattice.add_oscillator(
                phase=new_entry2.phase,
                natural_freq=new_entry2.frequency * 0.1,
                amplitude=new_entry2.amplitude,
                stability=0.8
            )
            new_entry2.metadata["oscillator_idx"] = osc_idx2
            
            # Store new entries
            self.memory_entries[new_entry1.id] = new_entry1
            self.memory_entries[new_entry2.id] = new_entry2
            
            # Remove original oscillator
            if "oscillator_idx" in entry.metadata:
                idx = entry.metadata["oscillator_idx"]
                if idx < len(lattice.oscillators):
                    lattice.oscillators[idx]["active"] = False
            
            # Remove original entry
            self.memory_entries.pop(mem_id, None)
            split += 1
    
    return split

def _should_split(self, memory: SolitonMemoryEntry) -> bool:
    """Determine if a memory should be split"""
    # Split if amplitude is very high or content is very long
    return (memory.amplitude > 1.5 or 
            len(memory.content) > 1000 or
            memory.heat > 0.9)  # Also split very hot memories

# Fix 3: Improve migration logic
def _migrate_to_stable_position(self, entry: SolitonMemoryEntry, lattice):
    """Migrate memory to more stable lattice position"""
    if "oscillator_idx" in entry.metadata:
        old_idx = entry.metadata["oscillator_idx"]
        
        # Get topology-specific stable positions
        topology_name = lattice.current_topology.name() if hasattr(lattice, 'current_topology') else "kagome"
        
        if topology_name == "kagome":
            # In Kagome, center positions (middle third) are more stable
            total_positions = len(lattice.oscillators)
            stable_start = total_positions // 3
            stable_end = 2 * total_positions // 3
            
            # If outside stable zone, try to migrate
            if old_idx < stable_start or old_idx > stable_end:
                # Find an inactive oscillator in stable zone
                for new_idx in range(stable_start, stable_end):
                    if new_idx < len(lattice.oscillators) and not lattice.oscillators[new_idx].get("active", True):
                        # Swap positions
                        lattice.oscillators[new_idx] = lattice.oscillators[old_idx].copy()
                        lattice.oscillators[old_idx]["active"] = False
                        entry.metadata["oscillator_idx"] = new_idx
                        logger.info(f"Migrated memory from position {old_idx} to stable position {new_idx}")
                        break
        
        # Update stability regardless of migration
        entry.stability = min(1.0, entry.stability + 0.1)
        
        # TODO: For other topologies (hexagonal, square, small-world), 
        # implement appropriate stable position identification

# Fix 4: Enhanced dark soliton configuration
def _store_dark_soliton(self, entry: SolitonMemoryEntry, lattice):
    """Store dark soliton with configurable coupling strength"""
    phase = entry.phase
    freq = entry.frequency * 0.1
    
    # Get coupling strength from config or use default
    dark_coupling_strength = getattr(self, 'dark_coupling_strength', 1.0)
    
    # Add baseline oscillator
    base_idx = lattice.add_oscillator(
        phase=phase,
        natural_freq=freq,
        amplitude=1.0,
        stability=1.0
    )
    
    # Add dip oscillator (π out of phase)
    dip_idx = lattice.add_oscillator(
        phase=(phase + np.pi) % (2 * np.pi),
        natural_freq=freq,
        amplitude=entry.amplitude,
        stability=1.0
    )
    
    # Strong coupling between baseline and dip
    lattice.set_coupling(base_idx, dip_idx, dark_coupling_strength)
    lattice.set_coupling(dip_idx, base_idx, dark_coupling_strength)
    
    entry.metadata["oscillator_idx"] = dip_idx
    entry.metadata["baseline_idx"] = base_idx
    
    # Store in memory entries
    self.memory_entries[entry.id] = entry
    
    logger.info(f"Stored dark soliton with coupling strength {dark_coupling_strength}")
