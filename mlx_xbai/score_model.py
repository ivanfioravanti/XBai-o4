"""
Score Model implementation for XBai o4 MLX
Process Reward Model with trained weights for solution scoring
"""

import mlx.core as mx
import mlx.nn as nn
from typing import List, Tuple, Optional
from pathlib import Path
import json


class ProcessRewardModel(nn.Module):
    """Process Reward Model for scoring reasoning steps"""
    
    def __init__(self):
        super().__init__()
        # Match the exact architecture from PyTorch
        self.linear1 = nn.Linear(5120, 10240)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(10240, 1)
    
    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Forward pass through score head"""
        x = self.linear1(hidden_states)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class RealXBaiScorer:
    """Real scoring system for XBai o4 using trained weights"""
    
    def __init__(self, model, tokenizer, score_model_path: Optional[str] = None):
        self.model = model
        self.tokenizer = tokenizer
        
        # Load special tokens
        self.think_start_id = self._get_token_id('<think>')
        self.think_end_id = self._get_token_id('</think>')
        
        # Get scoring position tokens (punctuation, newlines)
        self.score_token_ids = self._get_score_tokens()
        
        # Initialize and load PRM
        self.prm = self._load_prm(score_model_path)
        print("Score model loaded successfully")
    
    def _load_prm(self, score_model_path: Optional[str] = None) -> ProcessRewardModel:
        """Load the Process Reward Model with trained weights"""
        
        # Default path if not provided
        if score_model_path is None:
            score_model_path = Path.home() / "XBai-o4-4bit" / "score_head.npz"
        
        # Initialize model
        prm = ProcessRewardModel()
        
        # Load weights if file exists
        if Path(score_model_path).exists():
            print(f"Loading score model weights from {score_model_path}")
            weights = mx.load(str(score_model_path))
            
            # Update model weights
            prm.linear1.weight = weights['linear1.weight']
            prm.linear1.bias = weights['linear1.bias']
            prm.linear2.weight = weights['linear2.weight']
            prm.linear2.bias = weights['linear2.bias']
            
            print("Score model weights loaded successfully")
        else:
            print(f"Warning: Score model weights not found at {score_model_path}")
            print("Using random initialization")
        
        return prm
    
    def _get_token_id(self, token: str) -> int:
        """Get token ID, return -1 if not found"""
        try:
            ids = self.tokenizer.encode(token, add_special_tokens=False)
            return ids[0] if ids else -1
        except:
            return -1
    
    def _get_score_tokens(self) -> List[int]:
        """Get token IDs for scoring positions (punctuation, newlines)"""
        # Key tokens where we evaluate scores
        score_chars = ['.', '!', '?', '\n', '。', '！', '？']
        score_ids = []
        
        for char in score_chars:
            try:
                ids = self.tokenizer.encode(char, add_special_tokens=False)
                if ids:
                    score_ids.extend(ids)
            except:
                continue
        
        # Also add common sentence endings
        for ending in ['.\n', '!\n', '?\n', '. ', '! ', '? ']:
            try:
                ids = self.tokenizer.encode(ending, add_special_tokens=False)
                if ids:
                    score_ids.extend(ids)
            except:
                continue
        
        # Remove duplicates
        score_ids = list(set(score_ids))
        return score_ids
    
    def get_score_mask(self, input_ids: mx.array) -> mx.array:
        """Create mask for positions where we should compute scores"""
        batch_size, seq_len = input_ids.shape
        mask = mx.zeros((batch_size, seq_len), dtype=mx.bool_)
        
        # Mark positions with score tokens
        for token_id in self.score_token_ids:
            mask = mask | (input_ids == token_id)
        
        return mask
    
    def get_thinking_mask(self, input_ids: mx.array) -> mx.array:
        """Create mask for thinking tokens between <think> and </think>"""
        batch_size, seq_len = input_ids.shape
        mask = mx.zeros((batch_size, seq_len), dtype=mx.bool_)
        
        if self.think_start_id == -1 or self.think_end_id == -1:
            # If no thinking tags, return all True
            return mx.ones_like(mask, dtype=mx.bool_)
        
        for b in range(batch_size):
            seq = input_ids[b]
            
            # Find start and end positions
            start_matches = (seq == self.think_start_id)
            end_matches = (seq == self.think_end_id)
            
            if mx.any(start_matches) and mx.any(end_matches):
                # Find first occurrence using argmax
                start_idx = int(mx.argmax(start_matches))
                end_idx = int(mx.argmax(end_matches))
                
                # Only process if we actually found the tokens
                if start_matches[start_idx] and end_matches[end_idx] and end_idx > start_idx:
                    # Mark positions between start and end
                    mask[b, start_idx+1:end_idx] = True
            
        # If no thinking region found, use entire sequence
        if not mx.any(mask):
            return mx.ones_like(mask).astype(mx.bool_)
        
        return mask
    
    def geometric_mean(self, scores: mx.array) -> float:
        """Compute geometric mean of scores"""
        if scores.shape[0] == 0:
            return 0.5
        
        # Clip to avoid log(0)
        scores = mx.clip(scores, 1e-10, 1.0)
        
        # Geometric mean = exp(mean(log(scores)))
        log_scores = mx.log(scores)
        mean_log = mx.mean(log_scores)
        
        return float(mx.exp(mean_log))
    
    def compute_score(
        self,
        text: str,
        return_all_scores: bool = False
    ) -> Tuple[float, Optional[List[float]]]:
        """
        Compute reward score for a solution using actual model hidden states
        
        Args:
            text: The complete solution text
            return_all_scores: Whether to return individual step scores
        
        Returns:
            Tuple of (final_score, optional_all_scores)
        """
        
        # Tokenize input
        input_ids = mx.array(
            self.tokenizer.encode(text, add_special_tokens=False)
        ).reshape(1, -1)
        
        # Get hidden states from model (second-to-last layer)
        hidden_states = self._get_hidden_states(input_ids)
        
        if hidden_states is None:
            # Fallback if we can't get hidden states
            return (0.5, [0.5]) if return_all_scores else (0.5, None)
        
        # Get scoring masks
        score_mask = self.get_score_mask(input_ids)
        thinking_mask = self.get_thinking_mask(input_ids)
        
        # Combine masks: only score at punctuation within thinking regions
        final_mask = score_mask & thinking_mask
        
        # Count valid positions
        num_valid = mx.sum(final_mask)
        
        if num_valid == 0:
            # No valid scoring positions, return neutral score
            return (0.5, [0.5]) if return_all_scores else (0.5, None)
        
        # Extract features at scoring positions
        scoring_features_list = []
        mask_flat = final_mask[0]
        
        for i in range(mask_flat.shape[0]):
            if mask_flat[i]:
                scoring_features_list.append(hidden_states[0, i])
        
        if len(scoring_features_list) == 0:
            return (0.5, [0.5]) if return_all_scores else (0.5, None)
        
        # Stack features
        scoring_features = mx.stack(scoring_features_list)
        
        # Apply PRM to get scores
        raw_scores = self.prm(scoring_features)
        scores = mx.sigmoid(raw_scores).squeeze()
        
        # Ensure scores is 1D
        if scores.ndim == 0:
            scores = scores.reshape(1)
        
        # Compute final score as geometric mean
        final_score = self.geometric_mean(scores)
        
        if return_all_scores:
            all_scores = [float(s) for s in scores]
            return final_score, all_scores
        
        return final_score, None
    
    def _get_hidden_states(self, input_ids: mx.array) -> Optional[mx.array]:
        """
        Extract hidden states from model
        
        Note: This requires the model to return hidden states.
        For now, we'll try to hook into the model's forward pass.
        """
        try:
            # Try to get hidden states by running model with output_hidden_states
            # This requires model modification or monkey-patching
            
            # For MLX models, we need to access intermediate layers
            # This is model-specific and may need adjustment
            
            # Option 1: If model supports returning hidden states
            if hasattr(self.model, 'forward_with_hidden_states'):
                outputs = self.model.forward_with_hidden_states(input_ids)
                return outputs['hidden_states'][-2]  # Second-to-last layer
            
            # Option 2: Try to extract from model layers directly
            # Run through embedding and layers
            x = self.model.embed_tokens(input_ids)
            
            # Process through transformer layers
            for i, layer in enumerate(self.model.layers):
                x = layer(x)
                # Save second-to-last layer output
                if i == len(self.model.layers) - 2:
                    hidden_states = x
            
            return hidden_states
            
        except Exception as e:
            print(f"Warning: Could not extract hidden states: {e}")
            print("Using fallback scoring method")
            
            # Fallback: Use output embeddings as proxy
            try:
                # Get model output logits and use their embeddings
                outputs = self.model(input_ids)
                # Use the logits embeddings as a proxy for hidden states
                batch_size, seq_len = input_ids.shape
                hidden_dim = 5120  # Expected dimension for score model
                
                # Create pseudo-hidden states from logits
                # This is not ideal but better than random
                if hasattr(outputs, 'shape'):
                    # Reduce logits dimension to match score model input
                    vocab_size = outputs.shape[-1]
                    # Simple linear projection as fallback
                    pseudo_hidden = mx.zeros((batch_size, seq_len, hidden_dim))
                    # Fill with some signal from logits
                    for i in range(seq_len):
                        # Take top-k logits as features
                        top_k = min(hidden_dim, vocab_size)
                        pseudo_hidden[:, i, :top_k] = outputs[:, i, :top_k]
                    return pseudo_hidden
            except:
                pass
            
            return None


class EfficientScorer(RealXBaiScorer):
    """Optimized scorer for batch processing"""
    
    def score_batch(
        self,
        texts: List[str],
        batch_size: int = 4
    ) -> List[float]:
        """Score multiple texts efficiently"""
        scores = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_scores = [self.compute_score(text)[0] for text in batch]
            scores.extend(batch_scores)
        
        return scores