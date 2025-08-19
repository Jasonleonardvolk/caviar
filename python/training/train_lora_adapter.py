#!/usr/bin/env python3
"""
LoRA Adapter Training Pipeline for Saigon
Trains user-specific or global adapters from intent traces and mesh data
"""

import os
import json
import torch
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent))
from core.adapter_loader import AdapterManager, LoRALayer
from core.saigon_inference import SaigonLSTM, SaigonConfig

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class TrainingConfig:
    """Configuration for adapter training."""
    # Paths
    base_model_path: str = "models/saigon_base/lstm_model.pt"
    vocab_path: str = "models/saigon_base/vocab.json"
    output_dir: str = "models/adapters"
    data_dir: str = "data/adapter_training"
    
    # Training parameters
    batch_size: int = 8
    learning_rate: float = 1e-4
    num_epochs: int = 10
    warmup_steps: int = 100
    gradient_clip: float = 1.0
    
    # LoRA parameters
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.1
    target_modules: List[str] = None  # Will default to ["lstm", "linear"]
    
    # Data parameters
    max_sequence_length: int = 256
    validation_split: float = 0.1
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Logging
    log_every: int = 10
    save_every: int = 100
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["lstm", "linear"]

# ============================================================================
# DATASET
# ============================================================================

class AdapterTrainingDataset(Dataset):
    """
    Dataset for adapter training from intent traces and mesh data.
    """
    
    def __init__(self, 
                 data_path: str,
                 vocab: Dict[str, int],
                 max_length: int = 256):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to training data (JSONL format)
            vocab: Token to ID mapping
            max_length: Maximum sequence length
        """
        self.vocab = vocab
        self.max_length = max_length
        self.data = []
        
        # Load data
        self._load_data(data_path)
        
        logger.info(f"Loaded {len(self.data)} training examples")
    
    def _load_data(self, data_path: str) -> None:
        """Load training data from file."""
        path = Path(data_path)
        
        if not path.exists():
            logger.warning(f"No data found at {path}")
            return
        
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        # Expected format: {"input": "...", "output": "..."}
                        if "input" in item and "output" in item:
                            self.data.append(item)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON line: {line[:50]}...")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get training example."""
        item = self.data[idx]
        
        # Tokenize input and output
        input_ids = self._tokenize(item["input"])
        target_ids = self._tokenize(item["output"])
        
        # Combine for sequence-to-sequence
        # Format: <bos> input <sep> output <eos>
        bos_id = self.vocab.get("<bos>", 0)
        eos_id = self.vocab.get("<eos>", 0)
        sep_id = self.vocab.get("<sep>", self.vocab.get("|", 0))
        
        # Construct sequence
        sequence = [bos_id] + input_ids + [sep_id] + target_ids + [eos_id]
        
        # Truncate if needed
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
        
        # Pad if needed
        if len(sequence) < self.max_length:
            pad_id = self.vocab.get("<pad>", 0)
            sequence = sequence + [pad_id] * (self.max_length - len(sequence))
        
        # Create attention mask
        attention_mask = [1 if tok != self.vocab.get("<pad>", 0) else 0 for tok in sequence]
        
        return {
            "input_ids": torch.tensor(sequence[:-1], dtype=torch.long),
            "labels": torch.tensor(sequence[1:], dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask[:-1], dtype=torch.long)
        }
    
    def _tokenize(self, text: str) -> List[int]:
        """Tokenize text to IDs."""
        tokens = []
        for char in text:
            if char in self.vocab:
                tokens.append(self.vocab[char])
            else:
                tokens.append(self.vocab.get("<unk>", 1))
        return tokens

# ============================================================================
# DATA GENERATION WITH MESH CONTEXT
# ============================================================================

def generate_training_data_from_mesh(
    user_id: str,
    mesh_contexts_dir: str = "models/mesh_contexts",
    memory_vault_dir: str = "memory_vault",
    output_path: Optional[str] = None,
    mask_group_concepts: bool = False
) -> str:
    """
    Generate training data from user's mesh context, intent traces, and conversations.
    
    Args:
        user_id: User identifier
        mesh_contexts_dir: Directory with mesh summaries
        memory_vault_dir: Memory vault directory
        output_path: Where to save training data
        mask_group_concepts: Whether to exclude group/team concepts
        
    Returns:
        Path to generated training data
    """
    if not output_path:
        output_path = f"data/adapter_training/user_{user_id}_mesh_finetune.jsonl"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    training_data = []
    
    # Load mesh summary
    mesh_file = Path(mesh_contexts_dir) / f"{user_id}_mesh.json"
    mesh_summary = None
    if mesh_file.exists():
        with open(mesh_file, 'r') as f:
            mesh_summary = json.load(f)
        logger.info(f"Loaded mesh summary for {user_id}")
    
    # Generate from open intents
    if mesh_summary and "open_intents" in mesh_summary:
        for intent in mesh_summary["open_intents"]:
            # Create Q&A for unresolved intent
            example = {
                "input": f"How can I {intent['description']}?",
                "output": f"To address '{intent['description']}', let me help you with that. This is an open question that needs resolution."
            }
            training_data.append(example)
            
            # Create intent recognition example
            example = {
                "input": intent['description'],
                "output": f"Intent type: {intent.get('intent_type', 'unknown')}. Priority: {intent.get('priority', 'normal')}."
            }
            training_data.append(example)
    
    # Generate from personal concepts
    if mesh_summary and "personal_concepts" in mesh_summary:
        for concept in mesh_summary["personal_concepts"][:10]:
            # Create concept explanation
            example = {
                "input": f"Tell me about {concept['name']}",
                "output": f"{concept['name']}: {concept['summary']}"
            }
            training_data.append(example)
            
            # Create contextual reference
            if concept['score'] > 0.7:
                example = {
                    "input": f"What have we been working on related to {concept['name']}?",
                    "output": f"We've been focusing on {concept['name']}. {concept['summary']}"
                }
                training_data.append(example)
    
    # Generate from team concepts (unless masked)
    if not mask_group_concepts and mesh_summary and "team_concepts" in mesh_summary:
        for team, concepts in mesh_summary["team_concepts"].items():
            for concept in concepts[:5]:
                example = {
                    "input": f"What is {concept['name']} in the context of {team}?",
                    "output": f"In {team}, {concept['name']}: {concept['summary']}"
                }
                training_data.append(example)
    
    # Load intent traces
    traces_path = Path(memory_vault_dir) / "traces"
    if traces_path.exists():
        for trace_file in traces_path.glob(f"*{user_id}*.jsonl"):
            with open(trace_file, 'r') as f:
                for line in f:
                    if line.strip():
                        event = json.loads(line)
                        
                        # Extract training examples from intent traces
                        if event.get("event") == "intent_opened":
                            example = {
                                "input": event.get("description", ""),
                                "output": f"Intent recognized: {event.get('intent_type', 'unknown')}"
                            }
                            training_data.append(example)
    
    # Load conversation history
    sessions_path = Path(memory_vault_dir) / "sessions"
    if sessions_path.exists():
        for session_file in sessions_path.glob(f"*{user_id}*.jsonl"):
            with open(session_file, 'r') as f:
                conversation = []
                for line in f:
                    if line.strip():
                        event = json.loads(line)
                        
                        if event.get("event") == "conversation":
                            role = event.get("role")
                            text = event.get("text", "")
                            
                            if role == "user":
                                conversation.append(("user", text))
                            elif role == "assistant" and conversation and conversation[-1][0] == "user":
                                # Create Q&A pair
                                example = {
                                    "input": conversation[-1][1],
                                    "output": text
                                }
                                training_data.append(example)
    
    # Save training data
    with open(output_path, 'w') as f:
        for example in training_data:
            f.write(json.dumps(example) + '\n')
    
    logger.info(f"Generated {len(training_data)} training examples for user {user_id}")
    logger.info(f"  - {len([e for e in training_data if 'Intent' in e.get('output', '')])} intent examples")
    logger.info(f"  - {len([e for e in training_data if any(c['name'] in e.get('input', '') for c in mesh_summary.get('personal_concepts', []))])} concept examples" if mesh_summary else "")
    
    return str(output_path)

def generate_training_data_from_traces(
    user_id: str,
    memory_vault_dir: str = "memory_vault",
    output_path: Optional[str] = None
) -> str:
    """
    Legacy function - now calls generate_training_data_from_mesh.
    """
    return generate_training_data_from_mesh(
        user_id=user_id,
        memory_vault_dir=memory_vault_dir,
        output_path=output_path
    )

# ============================================================================
# TRAINING
# ============================================================================

class AdapterTrainer:
    """
    Trainer for LoRA adapters.
    """
    
    def __init__(self, config: TrainingConfig):
        """Initialize trainer."""
        self.config = config
        
        # Load vocabulary
        with open(config.vocab_path, 'r') as f:
            vocab_data = json.load(f)
            if isinstance(vocab_data, dict):
                self.vocab = vocab_data.get("vocab", [])
            else:
                self.vocab = vocab_data
        
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        
        # Initialize model
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        # Statistics
        self.train_losses = []
        self.val_losses = []
    
    def prepare_model(self) -> nn.Module:
        """Load base model and inject LoRA layers."""
        # Load base model
        logger.info(f"Loading base model from {self.config.base_model_path}")
        
        if Path(self.config.base_model_path).exists():
            model = torch.load(self.config.base_model_path, map_location=self.config.device)
        else:
            # Create default model
            model = SaigonLSTM(
                vocab_size=len(self.vocab),
                hidden_size=256,
                num_layers=2
            )
        
        # Inject LoRA layers
        logger.info(f"Injecting LoRA layers (rank={self.config.lora_rank})")
        
        lora_layers = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Check if this module should be adapted
                module_type = module.__class__.__name__.lower()
                if any(target in name.lower() for target in self.config.target_modules):
                    # Create and attach LoRA layer
                    lora_layer = LoRALayer(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        rank=self.config.lora_rank,
                        alpha=self.config.lora_alpha,
                        dropout=self.config.lora_dropout
                    )
                    
                    # Attach to module
                    module.lora_layer = lora_layer
                    lora_layers.append(lora_layer)
                    
                    # Monkey-patch forward
                    original_forward = module.forward
                    
                    def adapted_forward(self, x):
                        base_output = original_forward(x)
                        if hasattr(self, 'lora_layer'):
                            return self.lora_layer(x, base_output)
                        return base_output
                    
                    module.forward = adapted_forward.__get__(module, module.__class__)
        
        logger.info(f"Injected {len(lora_layers)} LoRA layers")
        
        # Move to device
        model = model.to(self.config.device)
        
        # Freeze base model parameters
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze LoRA parameters
        for lora_layer in lora_layers:
            for param in lora_layer.parameters():
                param.requires_grad = True
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
        
        return model
    
    def train(self, 
             data_path: str,
             output_name: str,
             user_id: Optional[str] = None) -> str:
        """
        Train adapter on data.
        
        Args:
            data_path: Path to training data
            output_name: Name for output adapter
            user_id: Optional user ID for mapping
            
        Returns:
            Path to saved adapter
        """
        # Prepare model
        self.model = self.prepare_model()
        
        # Create dataset
        dataset = AdapterTrainingDataset(
            data_path=data_path,
            vocab=self.token_to_id,
            max_length=self.config.max_sequence_length
        )
        
        if len(dataset) == 0:
            logger.error("No training data found")
            return None
        
        # Split into train/val
        val_size = int(len(dataset) * self.config.validation_split)
        train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        ) if val_size > 0 else None
        
        # Setup optimizer (only LoRA parameters)
        lora_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(lora_params, lr=self.config.learning_rate)
        
        # Setup scheduler
        total_steps = len(train_loader) * self.config.num_epochs
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps
        )
        
        # Loss function
        criterion = nn.CrossEntropyLoss(ignore_index=self.token_to_id.get("<pad>", 0))
        
        # Training loop
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        
        global_step = 0
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch_idx, batch in enumerate(train_loader):
                # Move to device
                input_ids = batch["input_ids"].to(self.config.device)
                labels = batch["labels"].to(self.config.device)
                attention_mask = batch["attention_mask"].to(self.config.device)
                
                # Forward pass
                if hasattr(self.model, 'forward'):
                    outputs = self.model(input_ids)
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                    else:
                        logits = outputs
                else:
                    logits = self.model(input_ids)
                
                # Reshape for loss
                logits = logits.view(-1, logits.size(-1))
                labels = labels.view(-1)
                
                # Calculate loss
                loss = criterion(logits, labels)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(lora_params, self.config.gradient_clip)
                
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                
                # Statistics
                train_loss += loss.item()
                global_step += 1
                
                # Logging
                if global_step % self.config.log_every == 0:
                    avg_loss = train_loss / (batch_idx + 1)
                    logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}, "
                              f"Step {global_step}, Loss: {avg_loss:.4f}")
                
                # Save checkpoint
                if global_step % self.config.save_every == 0:
                    self._save_checkpoint(output_name, epoch, global_step)
            
            # Validation
            if val_loader:
                val_loss = self._validate(val_loader, criterion)
                logger.info(f"Epoch {epoch+1} - Train Loss: {train_loss/len(train_loader):.4f}, "
                          f"Val Loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_adapter(output_name, user_id, is_best=True)
            else:
                logger.info(f"Epoch {epoch+1} - Train Loss: {train_loss/len(train_loader):.4f}")
        
        # Save final adapter
        output_path = self._save_adapter(output_name, user_id, is_best=False)
        
        logger.info(f"Training complete! Adapter saved to {output_path}")
        return output_path
    
    def _validate(self, val_loader: DataLoader, criterion: nn.Module) -> float:
        """Run validation."""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move to device
                input_ids = batch["input_ids"].to(self.config.device)
                labels = batch["labels"].to(self.config.device)
                
                # Forward pass
                outputs = self.model(input_ids)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                # Calculate loss
                logits = logits.view(-1, logits.size(-1))
                labels = labels.view(-1)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
        
        return val_loss / len(val_loader)
    
    def _save_checkpoint(self, name: str, epoch: int, step: int) -> None:
        """Save training checkpoint."""
        checkpoint_dir = Path(self.config.output_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"{name}_epoch{epoch}_step{step}.pt"
        
        # Extract LoRA weights only
        lora_state_dict = {}
        for name, module in self.model.named_modules():
            if hasattr(module, 'lora_layer'):
                lora_layer = module.lora_layer
                lora_state_dict[f"{name}.lora_A"] = lora_layer.lora_A.data
                lora_state_dict[f"{name}.lora_B"] = lora_layer.lora_B.data
        
        torch.save(lora_state_dict, checkpoint_path)
        logger.debug(f"Saved checkpoint to {checkpoint_path}")
    
    def _save_adapter(self, name: str, user_id: Optional[str], is_best: bool = False) -> str:
        """Save final adapter."""
        # Determine output path
        if user_id:
            output_name = f"user_{user_id}_lora.pt"
        else:
            output_name = f"{name}_lora.pt"
        
        if is_best:
            output_name = output_name.replace(".pt", "_best.pt")
        
        output_path = Path(self.config.output_dir) / output_name
        
        # Extract LoRA weights
        lora_state_dict = {}
        for module_name, module in self.model.named_modules():
            if hasattr(module, 'lora_layer'):
                lora_layer = module.lora_layer
                lora_state_dict[f"{module_name}.lora_A"] = lora_layer.lora_A.data
                lora_state_dict[f"{module_name}.lora_B"] = lora_layer.lora_B.data
        
        # Add metadata
        lora_state_dict["_metadata"] = {
            "rank": self.config.lora_rank,
            "alpha": self.config.lora_alpha,
            "dropout": self.config.lora_dropout,
            "target_modules": self.config.target_modules,
            "created_at": datetime.now().isoformat(),
            "training_epochs": self.config.num_epochs,
            "user_id": user_id
        }
        
        # Save
        torch.save(lora_state_dict, output_path)
        
        # Update adapter index if user_id provided
        if user_id:
            manager = AdapterManager(adapters_dir=str(self.config.output_dir))
            manager.update_user_mapping(user_id, output_name)
        
        logger.info(f"Adapter saved to {output_path}")
        return str(output_path)

# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Main CLI interface for training adapters with mesh context."""
    parser = argparse.ArgumentParser(description="Train LoRA adapter for Saigon with mesh context")
    
    # Required arguments
    parser.add_argument("--user_id", type=str, required=True,
                       help="User ID for adapter")
    
    # Optional arguments
    parser.add_argument("--data_path", type=str, default=None,
                       help="Path to training data (JSONL)")
    parser.add_argument("--memory_vault", type=str, default="memory_vault",
                       help="Memory vault directory")
    parser.add_argument("--mesh_contexts", type=str, default="models/mesh_contexts",
                       help="Mesh contexts directory")
    parser.add_argument("--output_name", type=str, default=None,
                       help="Name for output adapter")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--lora_rank", type=int, default=8,
                       help="LoRA rank")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use")
    
    # Mesh context flags
    parser.add_argument("--mask_group_concepts", action="store_true",
                       help="Exclude team/group concepts from training")
    parser.add_argument("--use_mesh", action="store_true", default=True,
                       help="Use mesh context for training data generation")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s'
    )
    
    # Generate training data if not provided
    if not args.data_path:
        logger.info(f"Generating training data for user {args.user_id}")
        if args.use_mesh:
            args.data_path = generate_training_data_from_mesh(
                user_id=args.user_id,
                mesh_contexts_dir=args.mesh_contexts,
                memory_vault_dir=args.memory_vault,
                mask_group_concepts=args.mask_group_concepts
            )
        else:
            args.data_path = generate_training_data_from_traces(
                user_id=args.user_id,
                memory_vault_dir=args.memory_vault
            )
    
    # Create training config
    config = TrainingConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        device=args.device
    )
    
    # Initialize trainer
    trainer = AdapterTrainer(config)
    
    # Train adapter
    output_name = args.output_name or f"user_{args.user_id}"
    adapter_path = trainer.train(
        data_path=args.data_path,
        output_name=output_name,
        user_id=args.user_id
    )
    
    if adapter_path:
        print(f"\n✅ Adapter training complete!")
        print(f"   Saved to: {adapter_path}")
        print(f"   User: {args.user_id}")
    else:
        print("\n❌ Training failed")

if __name__ == "__main__":
    main()
