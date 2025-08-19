"""
ConceptMesh KB Integration Patch

This patch adds entity linking support to concept_mesh.py for Wikidata KB integration.
Apply these changes to the existing concept_mesh.py file.
"""

# ==== CHANGES TO ADD TO concept_mesh.py ====

# 1. Add to __init__ method after line ~227 (after self.name_index initialization):
        # KB entity index for deduplication
        self.kb_index: Dict[str, ConceptID] = {}  # kb_id -> concept_id mapping

# 2. Replace the existing add_concept method with this enhanced version:
    def add_concept(
        self,
        name: str,
        description: str = "",
        category: str = "general",
        importance: float = 1.0,
        embedding: Optional[EmbeddingVector] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConceptID:
        """Add a new concept to the mesh (with KB deduplication)"""
        with self._lock:
            metadata = metadata or {}
            
            # Check for Wikidata ID in metadata for deduplication
            if 'wikidata_id' in metadata:
                kb_id = metadata['wikidata_id']
                
                # Check if this KB entity already exists
                if kb_id in self.kb_index:
                    existing_id = self.kb_index[kb_id]
                    logger.info(f"KB entity '{kb_id}' already exists as concept {existing_id}")
                    
                    # Update access info
                    self.concepts[existing_id].last_accessed = datetime.now()
                    self.concepts[existing_id].access_count += 1
                    
                    # Update phase metadata if provided
                    if 'entity_phase' in metadata:
                        self.concepts[existing_id].metadata['entity_phase'] = metadata['entity_phase']
                    if 'phase_locked' in metadata:
                        self.concepts[existing_id].metadata['phase_locked'] = metadata['phase_locked']
                    
                    return existing_id
                
                # Mark as canonical entity
                metadata['canonical'] = True
            
            # Check if concept already exists by name
            if name in self.name_index:
                existing_id = self.name_index[name]
                existing_concept = self.concepts[existing_id]
                
                # If existing concept doesn't have KB ID but new one does, update it
                if 'wikidata_id' in metadata and 'wikidata_id' not in existing_concept.metadata:
                    existing_concept.metadata.update({
                        'wikidata_id': metadata['wikidata_id'],
                        'canonical': True,
                        'entity_phase': metadata.get('entity_phase'),
                        'phase_locked': metadata.get('phase_locked', False)
                    })
                    # Add to KB index
                    self.kb_index[metadata['wikidata_id']] = existing_id
                    logger.info(f"Updated concept '{name}' with KB ID {metadata['wikidata_id']}")
                else:
                    logger.info(f"Concept '{name}' already exists with ID {existing_id}")
                
                # Update access info
                existing_concept.last_accessed = datetime.now()
                existing_concept.access_count += 1
                
                return existing_id
            
            # Generate UUID for new concept
            concept_id = str(uuid.uuid4())
            
            # Create concept
            concept = Concept(
                id=concept_id,
                name=name,
                description=description,
                category=category,
                importance=importance,
                embedding=embedding,
                metadata=metadata
            )
            
            # Add to storage
            self.concepts[concept_id] = concept
            self.graph.add_node(concept_id, **asdict(concept))
            
            # Update indices
            self.name_index[name] = concept_id
            self.category_index[category].add(concept_id)
            
            # Update KB index if applicable
            if 'wikidata_id' in metadata:
                self.kb_index[metadata['wikidata_id']] = concept_id
            
            # Cache embedding if provided
            if embedding is not None:
                # Clear cache for this ID to update it
                self._get_embedding_cached.cache_clear()
            
            # Create diff
            diff = ConceptDiff(
                id=str(uuid.uuid4()),
                diff_type="add",
                concepts=[concept_id],
                new_value=asdict(concept)
            )
            self._record_diff(diff)
            
            logger.info(f"Added concept: {name} (ID: {concept_id}){' [KB: ' + metadata.get('wikidata_id', '') + ']' if 'wikidata_id' in metadata else ''}")
            return concept_id

# 3. Add this new method after add_concept:
    def add_concept_from_kb(
        self,
        name: str,
        kb_id: str,
        entity_type: str = "ENTITY",
        confidence: float = 1.0,
        description: str = "",
        category: str = "entity",
        entity_phase: Optional[float] = None,
        phase_locked: bool = False,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> ConceptID:
        """Add a concept from knowledge base with canonical metadata"""
        metadata = {
            'wikidata_id': kb_id,
            'entity_type': entity_type,
            'confidence': confidence,
            'canonical': True,
            'source': 'entity_linker'
        }
        
        # Add phase information if provided
        if entity_phase is not None:
            metadata['entity_phase'] = entity_phase
            metadata['phase_locked'] = phase_locked
        
        # Merge additional metadata
        if additional_metadata:
            metadata.update(additional_metadata)
        
        return self.add_concept(
            name=name,
            description=description,
            category=category,
            importance=confidence,
            metadata=metadata
        )

# 4. Add this method to find concept by KB ID:
    def find_concept_by_kb_id(self, kb_id: str) -> Optional[Concept]:
        """Find a concept by its knowledge base ID"""
        with self._lock:
            concept_id = self.kb_index.get(kb_id)
            if concept_id and concept_id in self.concepts:
                concept = self.concepts[concept_id]
                # Update access info
                concept.last_accessed = datetime.now()
                concept.access_count += 1
                return concept
            return None

# 5. Add this method to get all canonical concepts:
    def get_canonical_concepts(self) -> List[Concept]:
        """Get all concepts that are linked to knowledge base entities"""
        with self._lock:
            return [
                concept for concept in self.concepts.values()
                if concept.metadata.get('canonical', False)
            ]

# 6. Update the _save_mesh method to include kb_index (add after indices saving):
                # Save KB index
                with open(indices_temp, 'w') as f:
                    json.dump({
                        'name_index': self.name_index,
                        'category_index': {k: list(v) for k, v in self.category_index.items()},
                        'kb_index': self.kb_index  # Add this line
                    }, f)

# 7. Update the _load_mesh method to load kb_index (add after loading indices):
                self.name_index = indices_data.get('name_index', {})
                category_index = indices_data.get('category_index', {})
                self.category_index = defaultdict(set)
                for k, v in category_index.items():
                    self.category_index[k] = set(v)
                
                # Load KB index
                self.kb_index = indices_data.get('kb_index', {})

# 8. Update get_statistics to include KB entity count:
            stats = {
                'total_concepts': len(self.concepts),
                'total_relations': len(self.relations),
                'canonical_concepts': len(self.kb_index),  # Add this line
                'categories': dict(self.category_index),
                # ... rest of stats
            }

# 9. Add method to migrate existing concepts to canonical:
    def link_concept_to_kb(
        self, 
        concept_id: ConceptID, 
        kb_id: str,
        entity_phase: Optional[float] = None,
        phase_locked: bool = False
    ) -> bool:
        """Link an existing concept to a KB entity"""
        with self._lock:
            if concept_id not in self.concepts:
                logger.warning(f"Concept {concept_id} not found")
                return False
            
            # Check if KB ID already exists
            if kb_id in self.kb_index and self.kb_index[kb_id] != concept_id:
                logger.warning(f"KB ID {kb_id} already linked to different concept")
                return False
            
            concept = self.concepts[concept_id]
            
            # Update metadata
            concept.metadata['wikidata_id'] = kb_id
            concept.metadata['canonical'] = True
            
            if entity_phase is not None:
                concept.metadata['entity_phase'] = entity_phase
                concept.metadata['phase_locked'] = phase_locked
            
            # Update KB index
            self.kb_index[kb_id] = concept_id
            
            logger.info(f"Linked concept {concept_id} to KB entity {kb_id}")
            return True

# ==== END OF PATCH ====
