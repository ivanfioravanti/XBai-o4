#!/usr/bin/env python
"""
XBai o4 MLX Inference with Low/Medium/High modes
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.utils import load_model


class XBaiInference:
    """Main inference class for XBai o4 with MLX"""
    
    def __init__(
        self,
        model_path: str,
        mode: str = "medium",
        temperature: float = 0.6,
        max_tokens: int = 32768,
        verbose: bool = False
    ):
        self.model_path = Path(model_path)
        self.mode = mode
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose
        
        # Set branch count based on mode
        self.branch_map = {
            "low": 2,
            "medium": 8,
            "high": 32
        }
        self.branch = self.branch_map.get(mode, 8)
        
        # Load model and tokenizer
        print(f"Loading model from {model_path}...")
        self.model, self.tokenizer = load(str(model_path))
        
        # Initialize score model
        self.score_model = ScoreModel(self.model, self.tokenizer)
        
        if verbose:
            print(f"Model loaded. Using {mode} mode with {self.branch} branches")
    
    def generate_candidates(
        self,
        prompt: str,
        n_candidates: int
    ) -> List[str]:
        """Generate multiple candidate solutions"""
        
        candidates = []
        
        # Format prompt with chat template
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        for i in range(n_candidates):
            if self.verbose:
                print(f"Generating candidate {i+1}/{n_candidates}...")
            
            # Generate with different seeds for diversity
            # Set seed for this generation
            mx.random.seed(int(time.time() * 1000) + i)
            
            # Create sampler with temperature
            sampler = make_sampler(temp=self.temperature)
            
            response = generate(
                self.model,
                self.tokenizer,
                prompt=formatted_prompt,
                max_tokens=self.max_tokens,
                sampler=sampler,
                verbose=False
            )
            
            candidates.append(formatted_prompt + response)
        
        return candidates
    
    def select_best_solution(
        self,
        candidates: List[str]
    ) -> Tuple[str, float, List[float]]:
        """Score all candidates and select the best one"""
        
        scores = []
        for i, candidate in enumerate(candidates):
            if self.verbose:
                print(f"Scoring candidate {i+1}/{len(candidates)}...")
            
            score = self.score_model.compute_score(candidate)
            scores.append(score)
        
        # Select best
        best_idx = int(mx.argmax(mx.array(scores)))
        best_solution = candidates[best_idx]
        best_score = scores[best_idx]
        
        # Extract actual response (remove prompt)
        messages = [{"role": "user", "content": "dummy"}]
        prompt_template = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        ).replace("dummy", "")
        
        # Find where the actual response starts
        response_start = best_solution.find(prompt_template.split("{")[0])
        if response_start != -1:
            response_start = best_solution.find("assistant\n", response_start)
            if response_start != -1:
                best_solution = best_solution[response_start + len("assistant\n"):]
        
        return best_solution, best_score, scores
    
    def run(self, prompt: str) -> Dict:
        """Run inference with selected mode"""
        
        start_time = time.time()
        
        # Generate candidates
        candidates = self.generate_candidates(prompt, self.branch)
        
        # Select best
        best_solution, best_score, all_scores = self.select_best_solution(candidates)
        
        elapsed = time.time() - start_time
        
        result = {
            "mode": self.mode,
            "branches": self.branch,
            "solution": best_solution,
            "score": float(best_score),
            "all_scores": [float(s) for s in all_scores],
            "time": elapsed
        }
        
        if self.verbose:
            print(f"\nCompleted in {elapsed:.2f}s")
            print(f"Best score: {best_score:.4f}")
            print(f"Score distribution: {all_scores}")
        
        return result


class ScoreModel:
    """Process Reward Model for scoring solutions"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
        # Get special token IDs
        self.think_start_id = self.tokenizer.encode('<think>', add_special_tokens=False)[0]
        self.think_end_id = self.tokenizer.encode('</think>', add_special_tokens=False)[0]
        
        # Get scoring position tokens (newlines, periods)
        self.score_tokens = self._get_score_tokens()
        
        # Initialize score head (2-layer MLP)
        self.score_dim = 1536  # Adjust based on model hidden size
        self.score_head = self._init_score_head()
    
    def _get_score_tokens(self) -> List[int]:
        """Get token IDs for scoring positions"""
        # For English, use period and newlines
        tokens = self.tokenizer.encode('.\n\n', add_special_tokens=False)
        return tokens
    
    def _init_score_head(self) -> nn.Sequential:
        """Initialize the scoring MLP"""
        return nn.Sequential(
            nn.Linear(self.score_dim, self.score_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.score_dim * 2, 1)
        )
    
    def compute_score(self, text: str) -> float:
        """Compute reward score for a solution"""
        
        # Tokenize
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        input_ids = mx.array(tokens).reshape(1, -1)
        
        # Get model hidden states (we need intermediate representations)
        # Note: This is a simplified version - actual implementation needs model modification
        # to return hidden states
        
        # For now, return a mock score based on solution length and structure
        # This should be replaced with actual PRM scoring
        score = self._mock_score(text)
        
        return score
    
    def _mock_score(self, text: str) -> float:
        """Temporary mock scoring function"""
        # Simple heuristic based on solution structure
        score = 0.5
        
        # Check for thinking tags
        if '<think>' in text and '</think>' in text:
            score += 0.1
        
        # Check for step-by-step reasoning
        lines = text.split('\n')
        if len(lines) > 5:
            score += 0.1
        
        # Check for mathematical notation
        if any(c in text for c in ['=', '+', '-', '*', '/']):
            score += 0.1
        
        # Check for conclusion
        if any(word in text.lower() for word in ['therefore', 'thus', 'answer', 'solution']):
            score += 0.1
        
        # Add some randomness for demo
        score += float(mx.random.uniform(low=-0.1, high=0.1, shape=()))
        
        return min(max(score, 0.0), 1.0)


def main():
    parser = argparse.ArgumentParser(description="XBai o4 MLX Inference")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the model"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["low", "medium", "high"],
        default="medium",
        help="Inference mode (low=2, medium=8, high=32 branches)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Input prompt"
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        help="File containing prompts (JSONL format)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.jsonl",
        help="Output file for results"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32768,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = XBaiInference(
        model_path=args.model_path,
        mode=args.mode,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        verbose=args.verbose
    )
    
    # Process prompts
    prompts = []
    if args.prompt:
        prompts = [args.prompt]
    elif args.prompt_file:
        with open(args.prompt_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                prompts.append(data.get('prompt', data.get('question', '')))
    else:
        # Interactive mode
        print("Enter your prompt (Ctrl+D to submit):")
        import sys
        prompt = sys.stdin.read().strip()
        prompts = [prompt]
    
    # Run inference
    results = []
    for i, prompt in enumerate(prompts):
        print(f"\nProcessing prompt {i+1}/{len(prompts)}...")
        print(f"Mode: {args.mode} ({inference.branch} branches)")
        
        result = inference.run(prompt)
        result['prompt'] = prompt
        results.append(result)
        
        print(f"\nBest solution (score: {result['score']:.4f}):")
        print("-" * 50)
        print(result['solution'])
        print("-" * 50)
    
    # Save results
    with open(args.output, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()