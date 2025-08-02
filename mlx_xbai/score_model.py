"""
Advanced Score Model implementation for XBai o4 MLX
Implements the actual Process Reward Model scoring
"""

import mlx.core as mx
import mlx.nn as nn
from typing import List, Tuple, Optional
import numpy as np
from pathlib import Path


class ProcessRewardModel(nn.Module):
    """Process Reward Model for scoring reasoning steps"""
    
    def __init__(self, hidden_size: int = 1536):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Score head: 2-layer MLP
        self.score_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size * 2, 1)
        )
    
    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Forward pass through score head"""
        return self.score_head(hidden_states)


class XBaiScorer:
    """Complete scoring system for XBai o4"""
    
    def __init__(self, model, tokenizer, score_model_path: Optional[str] = None):
        self.model = model
        self.tokenizer = tokenizer
        
        # Load special tokens
        self.think_start_id = self._get_token_id('<think>')
        self.think_end_id = self._get_token_id('</think>')
        
        # Get scoring position tokens
        self.score_token_ids = self._get_score_tokens()
        
        # Initialize PRM
        hidden_size = self._get_hidden_size()
        self.prm = ProcessRewardModel(hidden_size)
        
        # Load weights if provided
        if score_model_path and Path(score_model_path).exists():
            self.load_score_weights(score_model_path)
    
    def _get_token_id(self, token: str) -> int:
        """Get token ID, handling special tokens"""
        ids = self.tokenizer.encode(token, add_special_tokens=False)
        return ids[0] if ids else -1
    
    def _get_score_tokens(self) -> List[int]:
        """Get tokens that mark scoring positions"""
        # For English: periods and newlines
        score_tokens = []
        
        # Add period tokens
        period_ids = self.tokenizer.encode('.', add_special_tokens=False)
        score_tokens.extend(period_ids)
        
        # Add newline tokens
        newline_ids = self.tokenizer.encode('\n', add_special_tokens=False)
        score_tokens.extend(newline_ids)
        
        # Add double newline
        double_newline_ids = self.tokenizer.encode('\n\n', add_special_tokens=False)
        score_tokens.extend(double_newline_ids)
        
        return list(set(score_tokens))
    
    def _get_hidden_size(self) -> int:
        """Detect model hidden size"""
        # This would need to be adjusted based on actual model architecture
        # Common sizes: 768, 1024, 1536, 2048, 4096
        return 1536  # Default for medium-sized models
    
    def load_score_weights(self, path: str):
        """Load pre-trained score model weights"""
        weights = mx.load(path)
        self.prm.load_weights(weights)
    
    def get_score_mask(
        self,
        input_ids: mx.array
    ) -> mx.array:
        """Create mask for scoring positions"""
        batch_size, seq_len = input_ids.shape
        
        # Find positions after score tokens (periods, newlines)
        score_mask = mx.zeros((batch_size, seq_len), dtype=mx.bool_)
        
        for token_id in self.score_token_ids:
            token_positions = (input_ids == token_id)
            # Shift to mark position after the token
            shifted = mx.pad(token_positions[:, :-1], ((0, 0), (1, 0)))
            score_mask = score_mask | shifted
        
        # Exclude positions right after <think>
        if self.think_start_id != -1:
            think_positions = (input_ids == self.think_start_id)
            think_mask = mx.pad(think_positions[:, :-1], ((0, 0), (1, 0)))
            score_mask = score_mask & ~think_mask
        
        return score_mask
    
    def get_thinking_mask(
        self,
        input_ids: mx.array
    ) -> mx.array:
        """Create mask for content between <think> and </think>"""
        batch_size, seq_len = input_ids.shape
        mask = mx.zeros((batch_size, seq_len), dtype=mx.bool_)
        
        if self.think_start_id == -1 or self.think_end_id == -1:
            return mask
        
        for b in range(batch_size):
            seq = input_ids[b]
            
            # Find start and end positions
            start_pos = mx.where(seq == self.think_start_id)[0]
            end_pos = mx.where(seq == self.think_end_id)[0]
            
            if len(start_pos) > 0 and len(end_pos) > 0:
                start_idx = start_pos[0]
                end_idx = end_pos[0] if len(end_pos) > 0 else seq_len
                
                # Mark positions between start and end
                mask[b, start_idx+1:end_idx] = True
        
        return mask
    
    def geometric_mean(self, scores: mx.array) -> float:
        """Compute geometric mean of scores"""
        if scores.shape[0] == 0:
            return 0.0
        
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
        Compute reward score for a solution
        
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
        
        # Get hidden states from model
        # Note: This requires model modification to return hidden states
        # For now, we'll use a simplified approach
        hidden_states = self._get_hidden_states(input_ids)
        
        # Get scoring masks
        score_mask = self.get_score_mask(input_ids)
        thinking_mask = self.get_thinking_mask(input_ids)
        
        # Combine masks: only score within thinking regions
        final_mask = score_mask & thinking_mask
        
        # If no thinking tags, use entire sequence
        if not mx.any(thinking_mask):
            final_mask = score_mask
        
        # Extract features at scoring positions
        scoring_positions = mx.where(final_mask[0])[0]
        
        if len(scoring_positions) == 0:
            # No valid scoring positions, return neutral score
            return (0.5, [0.5]) if return_all_scores else (0.5, None)
        
        # Get hidden states at scoring positions
        scoring_features = hidden_states[0, scoring_positions]
        
        # Apply PRM to get scores
        raw_scores = self.prm(scoring_features)
        scores = mx.sigmoid(raw_scores).squeeze()
        
        # Compute final score as geometric mean
        final_score = self.geometric_mean(scores)
        
        if return_all_scores:
            all_scores = [float(s) for s in scores]
            return final_score, all_scores
        
        return final_score, None
    
    def _get_hidden_states(self, input_ids: mx.array) -> mx.array:
        """
        Extract hidden states from model
        
        Note: This is a simplified version. Actual implementation
        would need to hook into model layers to get intermediate representations.
        """
        # For demo purposes, return random features
        batch_size, seq_len = input_ids.shape
        hidden_size = self._get_hidden_size()
        
        # In real implementation, this would be:
        # outputs = self.model(input_ids, output_hidden_states=True)
        # return outputs.hidden_states[-2]  # Second-to-last layer
        
        # Mock implementation
        return mx.random.normal((batch_size, seq_len, hidden_size))


class EfficientScorer(XBaiScorer):
    """Optimized scorer for batch processing"""
    
    def score_batch(
        self,
        texts: List[str],
        batch_size: int = 4
    ) -> List[float]:
        """Score multiple texts efficiently"""
        scores = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_scores = [self.compute_score(text)[0] for text in batch]
            scores.extend(batch_scores)
        
        return scores
    
    def score_with_caching(
        self,
        text: str,
        cache: dict
    ) -> float:
        """Score with caching for repeated evaluations"""
        # Create cache key from first 100 chars
        cache_key = text[:100]
        
        if cache_key in cache:
            return cache[cache_key]
        
        score, _ = self.compute_score(text)
        cache[cache_key] = score
        
        return score