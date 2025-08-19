from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# {PROJECT_ROOT}\models\efficientnet\saigon.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import logging
import time
from typing import Dict, List, Any, Optional, Union

# Import utilities
try:
    from .saigon_utils import mesh_to_text_enhanced, log_mesh_traversal, validate_mesh_path
except ImportError:
    from saigon_utils import mesh_to_text_enhanced, log_mesh_traversal, validate_mesh_path

logger = logging.getLogger("saigon")

# --- Config ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "saigon_lstm.pt")
VOCAB_PATH = os.path.join(os.path.dirname(__file__), "vocab.json")

# --- Character-level LSTM core ---
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size=512, num_layers=2, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size, 
            hidden_size, 
            num_layers, 
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        logits = self.fc(out)
        return logits, hidden


class SaigonGenerator:
    """
    Main Saigon mesh-to-text generator with robust error handling and graceful fallback.
    """
    
    def __init__(self, model_path: Optional[str] = None, vocab_path: Optional[str] = None, device: str = 'cpu'):
        self.model_path = model_path or MODEL_PATH
        self.vocab_path = vocab_path or VOCAB_PATH
        self.device = device
        self.model = None
        self.vocab = None
        self.model_loaded = False
        
    def load_model(self) -> bool:
        """
        Load the LSTM model and vocabulary with robust error handling.
        
        Returns:
            bool: True if model loaded successfully, False if falling back to raw mode
        """
        try:
            # Load vocabulary
            if not os.path.exists(self.vocab_path):
                logger.warning(f"Vocabulary file not found: {self.vocab_path}")
                return False
                
            with open(self.vocab_path, "r", encoding="utf-8") as f:
                self.vocab = json.load(f)
            
            # Create reverse mapping if needed
            if 'itos' not in self.vocab:
                self.vocab['itos'] = {str(v): k for k, v in self.vocab.items()}
            if 'stoi' not in self.vocab:
                self.vocab['stoi'] = self.vocab
                
            # Load model
            if not os.path.exists(self.model_path):
                logger.warning(f"Model file not found: {self.model_path}")
                return False
                
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Initialize model with correct vocab size
            vocab_size = len(self.vocab['itos']) if 'itos' in self.vocab else len(self.vocab)
            self.model = CharLSTM(vocab_size).to(self.device)
            
            # Load state dict
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
                
            self.model_loaded = True
            logger.info(f"Saigon model loaded successfully: vocab_size={vocab_size}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load Saigon model: {e}")
            self.model_loaded = False
            return False
    
    def text_to_tensor(self, text: str) -> torch.Tensor:
        """Convert text to tensor using vocabulary."""
        if not self.vocab:
            raise ValueError("Vocabulary not loaded")
            
        stoi = self.vocab.get('stoi', self.vocab)
        unk_token = stoi.get('<unk>', stoi.get(' ', 0))
        
        idxs = [stoi.get(c, unk_token) for c in text]
        return torch.tensor(idxs, dtype=torch.long).unsqueeze(0)
    
    def sample_lstm(self, seed: str, max_len: int = 200, temperature: float = 1.0) -> str:
        """
        Generate text using the LSTM model.
        
        Args:
            seed: Initial text to seed generation
            max_len: Maximum length to generate
            temperature: Sampling temperature (1.0 = "smartest-ever" mode)
            
        Returns:
            str: Generated text
        """
        if not self.model_loaded:
            raise ValueError("Model not loaded")
            
        self.model.eval()
        x = self.text_to_tensor(seed).to(self.device)
        output = []
        hidden = None
        
        itos = self.vocab.get('itos', {str(i): chr(i) for i in range(256)})
        
        with torch.no_grad():
            for i in range(max_len):
                logits, hidden = self.model(x, hidden)
                logits = logits[:, -1, :] / max(temperature, 1e-8)  # Prevent division by zero
                probs = F.softmax(logits, dim=-1)
                idx = torch.multinomial(probs, num_samples=1).item()
                output.append(idx)
                x = torch.tensor([[idx]], dtype=torch.long).to(self.device)
                
                # Stop on newline or specific tokens
                char = itos.get(str(idx), '')
                if char in ['\n', '.', '!', '?'] and len(output) > 20:
                    break
                    
        generated = ''.join([itos.get(str(i), '') for i in output])
        return seed + generated
    
    def generate(self, mesh_path: List[Dict[str, Any]], 
                 smoothing: bool = True, 
                 max_len: int = 256, 
                 temperature: float = 1.0) -> Dict[str, Any]:
        """
        Main generation function with mesh-to-text conversion and optional LSTM smoothing.
        
        Args:
            mesh_path: List of concept relationship dictionaries
            smoothing: Whether to apply LSTM smoothing
            max_len: Maximum generation length
            temperature: Sampling temperature
            
        Returns:
            dict: Generation results with audit information
        """
        start_time = time.time()
        
        # Validate mesh path
        if not validate_mesh_path(mesh_path):
            logger.error("Invalid mesh path provided")
            return {
                "text": "Invalid mesh path structure.",
                "method": "error",
                "processing_time": time.time() - start_time,
                "audit": {"error": "Invalid mesh path"}
            }
        
        # Convert mesh to base text using enhanced templates
        try:
            base_text = mesh_to_text_enhanced(mesh_path)
        except Exception as e:
            logger.error(f"Mesh-to-text conversion failed: {e}")
            base_text = "Mesh conversion error."
        
        # Apply LSTM smoothing if requested and available
        final_text = base_text
        method = "raw_mesh"
        
        if smoothing:
            try:
                if not self.model_loaded:
                    self.load_model()
                
                if self.model_loaded:
                    final_text = self.sample_lstm(base_text, max_len, temperature)
                    method = "lstm_smoothed"
                else:
                    logger.warning("LSTM model not available - falling back to raw mesh text")
                    method = "fallback_raw"
                    
            except Exception as e:
                logger.warning(f"LSTM generation failed: {e} - falling back to raw mesh text")
                final_text = base_text
                method = "fallback_raw"
        
        processing_time = time.time() - start_time
        
        # Create audit log
        audit_data = log_mesh_traversal(mesh_path, final_text, processing_time)
        audit_data.update({
            "method": method,
            "base_text_length": len(base_text),
            "final_text_length": len(final_text),
            "smoothing_requested": smoothing,
            "temperature": temperature,
            "model_loaded": self.model_loaded
        })
        
        return {
            "text": final_text,
            "base_text": base_text,
            "method": method,
            "processing_time": processing_time,
            "audit": audit_data
        }


# --- Legacy compatibility functions ---
def mesh_to_text(mesh_path):
    """Legacy mesh-to-text function for backward compatibility."""
    sentences = []
    for node in mesh_path:
        c = node.get('concept', 'Something')
        r = node.get('relation', 'relates to')
        ctx = node.get('context', '')
        if ctx:
            sentence = f"{c.capitalize()} {r} {ctx}."
        else:
            sentence = f"{c.capitalize()} {r}."
        sentences.append(sentence)
    return " ".join(sentences)

def saigon_generate(mesh_path, smoothing=True, max_len=256, device='cpu', temperature=1.0):
    """
    Legacy generation function for backward compatibility.
    """
    generator = SaigonGenerator(device=device)
    result = generator.generate(mesh_path, smoothing, max_len, temperature)
    return result["text"]

# --- Minimal CLI/test driver ---
if __name__ == "__main__":
    # Example mesh path
    mesh_path = [
        {"concept": "abduction", "relation": "supports", "context": "inference"},
        {"concept": "deduction", "relation": "derives", "context": "logic"},
        {"concept": "analogy", "relation": "bridges", "context": "reasoning"}
    ]
    print("Mesh-to-Text:", mesh_to_text(mesh_path))
    print("Saigon Output:", saigon_generate(mesh_path, smoothing=True))
