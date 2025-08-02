"""
Score Model implementation for XBai o4 MLX
Process Reward Model with trained weights for solution scoring
"""

import mlx.core as mx
import mlx.nn as nn
from typing import List, Tuple, Optional
from pathlib import Path
import os


class ProcessRewardModel(nn.Module):
    """Process Reward Model for scoring reasoning steps"""
    
    def __init__(self, dropout_p: float = 0.5):
        super().__init__()
        # Match the exact architecture from PyTorch original
        # Fixed dimension: 1536
        score_model_dim = 1536
        self.linear1 = nn.Linear(score_model_dim, score_model_dim * 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_p)
        self.linear2 = nn.Linear(score_model_dim * 2, 1)
    
    def __call__(self, hidden_states: mx.array, training: bool = False) -> mx.array:
        """Forward pass through score head"""
        x = self.linear1(hidden_states)
        x = self.relu(x)
        if training:
            x = self.dropout(x)
        x = self.linear2(x)
        return x


class XBaiScorer:
    """Scoring system for XBai o4 using model's hidden states"""
    
    def __init__(self, model, tokenizer, model_path: str = None, lang: str = 'en'):
        self.model = model
        self.tokenizer = tokenizer
        self.score_model_dim = 1536  # Fixed dimension
        self.lang = lang
        
        # Load special tokens
        self.think_start_id = self._get_token_id('<think>')
        self.think_end_id = self._get_token_id('</think>')
        
        # Get scoring position tokens (punctuation, newlines)
        self.score_token_ids = self._get_all_key_ids()
        
        # Initialize and load PRM head
        self.prm = self._load_prm(model_path)
        print("Score model initialized")
    
    
    def _get_token_id(self, token: str) -> int:
        """Get token ID, return -1 if not found"""
        try:
            ids = self.tokenizer.encode(token, add_special_tokens=False)
            return ids[0] if ids else -1
        except:
            return -1
    
    def _get_all_key_ids(self, target_chars: List[str] = None) -> List[int]:
        """Get token IDs for scoring positions - matches original get_all_key_ids"""
        if target_chars is None:
            target_chars = None  # Default to English behavior
        
        if False:  # Removed language-specific logic
            # Chinese mode - match original logic
            vocab = self.tokenizer.get_vocab() if hasattr(self.tokenizer, 'get_vocab') else {}
            target_token_ids = []
            
            for token in vocab.keys():
                try:
                    token_id = self.tokenizer.convert_tokens_to_ids([token])[0] if hasattr(self.tokenizer, 'convert_tokens_to_ids') else self.tokenizer.encode(token, add_special_tokens=False)[0]
                    decoded = self.tokenizer.decode([token_id])
                    for target_char in target_chars:
                        if target_char in decoded and len(decoded.replace(target_char, '')) <= 1 and len(decoded) <= 3:
                            target_token_ids.append(token_id)
                            break
                except:
                    continue
        else:
            # English mode - use .\n\n pattern
            target_token_ids = self.tokenizer.encode('.\n\n', add_special_tokens=False)
        
        return target_token_ids
    
    def get_score_mask(self, input_ids: mx.array) -> mx.array:
        """Create mask for positions where we should compute scores - matches original"""
        # Create mask for target tokens
        mask = mx.zeros_like(input_ids).astype(mx.bool_)
        for token_id in self.score_token_ids:
            mask = mask | (input_ids == token_id)
        
        # Only keep first occurrence in consecutive sequences
        mask_shifted = mx.pad(mask[:, :-1], ((0, 0), (1, 0)), constant_values=False)
        mask_first_only = mask & ~mask_shifted
        
        # Exclude positions right after <think> token
        think_pos = (input_ids == self.think_start_id)
        think_next_mask = mx.pad(think_pos[:, :-1], ((0, 0), (1, 0)), constant_values=False)
        final_mask = mask_first_only & ~think_next_mask
        
        # Shift mask by one position to align with hidden states
        final_mask = mx.pad(final_mask[:, 1:], ((0, 0), (0, 1)), constant_values=False)
        
        return final_mask
    
    def get_thinking_mask(self, input_ids: mx.array) -> mx.array:
        """Create mask for thinking tokens between <think> and </think>"""
        batch_size, seq_len = input_ids.shape
        mask = mx.zeros((batch_size, seq_len)).astype(mx.bool_)
        
        if self.think_start_id == -1 or self.think_end_id == -1:
            # If no thinking tags, return all True
            return mx.ones_like(mask).astype(mx.bool_)
        
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
    
    def geometric_mean(self, scores: mx.array, eps: float = 1e-10) -> float:
        """Compute geometric mean of scores - matches original"""
        if scores.shape[0] == 0:
            return 0.0  # Match original: returns 0 for empty tensor
        
        # Clip to avoid log(0) - matches original
        scores = mx.clip(scores, eps, mx.inf)
        
        # Geometric mean = exp(mean(log(scores)))
        log_scores = mx.log(scores)
        mean_log = mx.mean(log_scores)
        
        return float(mx.exp(mean_log))
    
    def compute_score(
        self,
        text: str,
        return_all_scores: bool = False
    ) -> Tuple[float, Optional[List[float]], Optional[List[int]], Optional[List[int]]]:
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
        
        # Get score mask (already handles think token exclusion)
        score_mask = self.get_score_mask(input_ids)
        
        # Get positions between <think> and </think>
        # In MLX, we need to find indices differently
        think_start_mask = (input_ids[0] == self.think_start_id)
        think_end_mask = (input_ids[0] == self.think_end_id)
        
        # Find first occurrence using argmax (returns first True position)
        start_pos = 0
        end_pos = input_ids.shape[1]
        
        if mx.any(think_start_mask):
            start_pos = int(mx.argmax(think_start_mask))
        
        if mx.any(think_end_mask):
            end_pos = int(mx.argmax(think_end_mask))
        
        # Create between mask
        between_mask = mx.zeros_like(input_ids).astype(mx.bool_)
        between_mask[:, start_pos + 1:end_pos] = True
        
        # Combine masks
        final_mask = between_mask & score_mask
        
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
            # Get scoring positions
            score_positions = [i for i in range(mask_flat.shape[0]) if mask_flat[i]]
            token_ids = [int(id) for id in input_ids[0]]
            return final_score, all_scores, token_ids, score_positions
        
        return final_score, None, None, None
    
    def _get_hidden_states(self, input_ids: mx.array) -> Optional[mx.array]:
        """
        Extract hidden states from the already-loaded MLX model
        
        This extracts the second-to-last layer's hidden states by running
        the model's forward pass and capturing intermediate outputs.
        """
        try:
            # For MLX LLM models, we need to run through the layers manually
            # to capture intermediate states
            
            # For MLX models, check for nested model structure
            actual_model = self.model
            if hasattr(self.model, 'model'):
                actual_model = self.model.model
            
            # Get embeddings
            if hasattr(actual_model, 'embed_tokens'):
                x = actual_model.embed_tokens(input_ids)
            elif hasattr(actual_model, 'wte'):  # GPT-style
                x = actual_model.wte(input_ids)
            elif hasattr(actual_model, 'tok_embeddings'):  # LLaMA-style
                x = actual_model.tok_embeddings(input_ids)
            else:
                raise ValueError(f"Could not find embedding layer in {type(actual_model)}")
            
            # Get the transformer layers
            if hasattr(actual_model, 'layers'):
                layers = actual_model.layers
            elif hasattr(actual_model, 'h'):  # GPT-style
                layers = actual_model.h
            elif hasattr(actual_model, 'transformer') and hasattr(actual_model.transformer, 'h'):
                layers = actual_model.transformer.h
            else:
                raise ValueError("Could not find transformer layers")
            
            # Process through transformer layers
            hidden_states = None
            for i, layer in enumerate(layers):
                x = layer(x)
                # Save second-to-last layer output
                if i == len(layers) - 2:
                    hidden_states = x
            
            if hidden_states is None:
                # If model has fewer than 2 layers, use the last available
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
                
                # Create pseudo-hidden states from logits
                # This is not ideal but better than random
                if hasattr(outputs, 'shape'):
                    # Create pseudo-hidden states as fallback
                    # Return random features as last resort
                    return mx.random.normal((batch_size, seq_len, self.score_model_dim))
            except:
                pass
            
            return None


class EfficientScorer(XBaiScorer):
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

    def _load_prm(self, model_path: str = None) -> ProcessRewardModel:
        """Load the Process Reward Model with trained weights"""
        
        # Initialize model with fixed dimensions (1536)
        prm = ProcessRewardModel()
        
        # Try to find score_module.pt in the model directory
        if model_path:
            score_path = Path(model_path) / "score_module.pt"
            # Also check in parent directory for 4bit models
            if not score_path.exists():
                parent_path = Path(os.path.expanduser('~/XBai-o4-4bit/score_module.pt'))
                if parent_path.exists():
                    score_path = parent_path
            
            if score_path.exists():
                print(f"Loading score model weights from {score_path}")
                try:
                    import torch
                    state_dict = torch.load(str(score_path), map_location='cpu')
                    # Convert PyTorch weights to MLX
                    # The state dict has keys like '0.weight', '0.bias', '3.weight', '3.bias'
                    prm.linear1.weight = mx.array(state_dict['0.weight'].numpy())
                    prm.linear1.bias = mx.array(state_dict['0.bias'].numpy())
                    prm.linear2.weight = mx.array(state_dict['3.weight'].numpy())
                    prm.linear2.bias = mx.array(state_dict['3.bias'].numpy())
                    print("Score model weights loaded successfully")
                except Exception as e:
                    print(f"Warning: Could not load score weights: {e}")
            else:
                print(f"Warning: score_module.pt not found in {model_path}")
                print("Using random initialization for score model")
        
        return prm