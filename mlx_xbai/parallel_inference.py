"""
Parallel inference with async generation and scoring
Optimized for MLX performance
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple, Optional
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from score_model import EfficientScorer


class ParallelInferenceEngine:
    """Manages parallel generation and scoring for optimal performance"""
    
    def __init__(
        self,
        model_path: str,
        max_workers: int = 4,
        cache_size: int = 100
    ):
        self.model_path = model_path
        self.max_workers = max_workers
        
        # Load model and tokenizer
        self.model, self.tokenizer = load(model_path)
        
        # Initialize scorer with the already-loaded model
        self.scorer = EfficientScorer(
            model=self.model,
            tokenizer=self.tokenizer,
            model_path=model_path
        )
        
        # Cache for score results
        self.score_cache = {}
        self.cache_size = cache_size
        
        # Thread pool for parallel operations
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def generate_single(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        seed: int
    ) -> str:
        """Generate a single response"""
        # Set seed for this generation
        mx.random.seed(seed)
        
        # Create sampler with temperature
        sampler = make_sampler(temp=temperature)
        
        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False
        )
        return prompt + response
    
    def generate_parallel(
        self,
        prompt: str,
        n_candidates: int,
        max_tokens: int = 32768,
        temperature: float = 0.6,
        verbose: bool = False
    ) -> Tuple[List[str], List[float], Dict]:
        """Generate multiple candidates (sequentially to avoid Metal conflicts)
        
        Returns:
            Tuple of (candidates, generation_times, timing_stats)
        """
        
        # Format prompt
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize prompt to get token count
        prompt_tokens = len(self.tokenizer.encode(formatted_prompt))
        
        # Generate sequentially to avoid Metal command buffer conflicts
        candidates = []
        generation_times = []
        tokens_generated = []
        base_seed = int(time.time() * 1000)
        
        for i in range(n_candidates):
            # Show progress bar (always show, not just in verbose mode)
            progress = i + 1
            bar_length = 30
            filled = int(bar_length * (i) / n_candidates)
            bar = '█' * filled + '░' * (bar_length - filled)
            print(f"\rGenerating: [{bar}] {i}/{n_candidates} candidates", end='', flush=True)
            
            gen_start = time.time()
            candidate = self.generate_single(
                formatted_prompt,
                max_tokens,
                temperature,
                base_seed + i
            )
            gen_time = time.time() - gen_start
            
            # Calculate tokens generated
            candidate_tokens = len(self.tokenizer.encode(candidate))
            output_tokens = candidate_tokens - prompt_tokens
            
            candidates.append(candidate)
            generation_times.append(gen_time)
            tokens_generated.append(output_tokens)
            
            # Update progress bar after completion
            filled = int(bar_length * progress / n_candidates)
            bar = '█' * filled + '░' * (bar_length - filled)
            print(f"\rGenerating: [{bar}] {progress}/{n_candidates} candidates", end='', flush=True)
        
        print()  # New line after progress bar
        
        # Calculate generation statistics
        total_gen_time = sum(generation_times)
        avg_gen_time = total_gen_time / n_candidates
        total_tokens = sum(tokens_generated)
        avg_tokens_per_sec = total_tokens / total_gen_time if total_gen_time > 0 else 0
        
        timing_stats = {
            "total_generation_time": total_gen_time,
            "avg_generation_time": avg_gen_time,
            "individual_times": generation_times,
            "tokens_generated": tokens_generated,
            "total_tokens": total_tokens,
            "avg_tokens_per_second": avg_tokens_per_sec,
            "prompt_tokens": prompt_tokens
        }
        
        return candidates, generation_times, timing_stats
    
    def score_parallel(
        self,
        candidates: List[str],
        verbose: bool = False
    ) -> List[float]:
        """Score multiple candidates in parallel"""
        
        # Check cache first
        scores = []
        uncached_candidates = []
        uncached_indices = []
        
        for i, candidate in enumerate(candidates):
            cache_key = candidate[:100]  # Use first 100 chars as key
            
            if cache_key in self.score_cache:
                scores.append(self.score_cache[cache_key])
            else:
                scores.append(None)
                uncached_candidates.append(candidate)
                uncached_indices.append(i)
        
        # Score uncached candidates
        if uncached_candidates:
            # Batch score for efficiency
            new_scores = self.scorer.score_batch(uncached_candidates, batch_size=4)
            
            # Update cache and results
            for idx, score in zip(uncached_indices, new_scores):
                scores[idx] = score
                cache_key = candidates[idx][:100]
                
                # Manage cache size
                if len(self.score_cache) >= self.cache_size:
                    # Remove oldest entry (simple FIFO)
                    first_key = next(iter(self.score_cache))
                    del self.score_cache[first_key]
                
                self.score_cache[cache_key] = score
        
        return scores
    
    def run_inference(
        self,
        prompt: str,
        mode: str = "medium",
        max_tokens: int = 32768,
        temperature: float = 0.6,
        verbose: bool = False
    ) -> Dict:
        """
        Run complete inference pipeline
        
        Args:
            prompt: Input prompt
            mode: "low", "medium", or "high"
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            verbose: Print progress
        
        Returns:
            Dictionary with results
        """
        
        # Determine branch count
        branch_map = {"low": 2, "medium": 8, "high": 32}
        n_branches = branch_map.get(mode, 8)
        
        start_time = time.time()
        
        # Phase 1: Generate candidates with timing stats
        candidates, gen_times, gen_stats = self.generate_parallel(
            prompt,
            n_branches,
            max_tokens,
            temperature,
            verbose=verbose
        )
        
        gen_time = time.time() - start_time
        
        # Phase 2: Score ALL candidates
        if verbose:
            print(f"Scoring {n_branches} candidates...")
        
        score_start = time.time()
        scores = self.score_parallel(candidates, verbose=verbose)
        score_time = time.time() - score_start
        
        # Calculate scoring speed
        avg_score_time = score_time / n_branches if n_branches > 0 else 0
        
        # Phase 3: Select best
        best_idx = int(mx.argmax(mx.array(scores)))
        best_candidate = candidates[best_idx]
        best_score = scores[best_idx]
        
        # Extract response (remove prompt part)
        messages = [{"role": "user", "content": prompt}]
        prompt_template = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Process all candidates to extract responses and pair with scores
        all_responses = []
        response_details = []
        for i, candidate in enumerate(candidates):
            if candidate.startswith(prompt_template):
                response_text = candidate[len(prompt_template):]
            else:
                response_text = candidate
            all_responses.append(response_text)
            response_details.append({
                "index": i,
                "response": response_text,
                "score": float(scores[i]),
                "generation_time": gen_times[i],
                "tokens_generated": gen_stats["tokens_generated"][i],
                "tokens_per_second": gen_stats["tokens_generated"][i] / gen_times[i] if gen_times[i] > 0 else 0
            })
        
        # Sort responses by score (descending)
        sorted_responses = sorted(response_details, key=lambda x: x["score"], reverse=True)
        
        # Get best response
        response = all_responses[best_idx]
        
        total_time = time.time() - start_time
        
        return {
            "mode": mode,
            "branches": n_branches,
            "response": response,
            "best_response": {
                "text": response,
                "score": float(best_score),
                "index": best_idx,
                "generation_time": gen_times[best_idx],
                "tokens_generated": gen_stats["tokens_generated"][best_idx]
            },
            "all_scores": [float(s) for s in scores],
            "all_responses": all_responses,
            "response_details": response_details,
            "sorted_responses": sorted_responses,
            "generation_stats": gen_stats,
            "timing": {
                "generation": gen_time,
                "scoring": score_time,
                "total": total_time,
                "avg_generation_per_candidate": gen_stats["avg_generation_time"],
                "avg_scoring_per_candidate": avg_score_time,
                "tokens_per_second": gen_stats["avg_tokens_per_second"]
            }
        }
    
    def cleanup(self):
        """Clean up resources"""
        self.executor.shutdown(wait=True)


class AsyncInferenceEngine(ParallelInferenceEngine):
    """Async version for better concurrency"""
    
    async def generate_async(
        self,
        prompt: str,
        n_candidates: int,
        max_tokens: int,
        temperature: float
    ) -> List[str]:
        """Async generation of candidates"""
        
        loop = asyncio.get_event_loop()
        
        # Format prompt
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Create async tasks
        tasks = []
        base_seed = int(time.time() * 1000)
        
        for i in range(n_candidates):
            task = loop.run_in_executor(
                self.executor,
                self.generate_single,
                formatted_prompt,
                max_tokens,
                temperature,
                base_seed + i
            )
            tasks.append(task)
        
        # Wait for all to complete
        candidates = await asyncio.gather(*tasks)
        return candidates
    
    async def score_async(
        self,
        candidates: List[str]
    ) -> List[float]:
        """Async scoring of candidates"""
        
        loop = asyncio.get_event_loop()
        
        # Run scoring in executor
        scores = await loop.run_in_executor(
            self.executor,
            self.score_parallel,
            candidates
        )
        
        return scores
    
    async def run_inference_async(
        self,
        prompt: str,
        mode: str = "medium",
        max_tokens: int = 32768,
        temperature: float = 0.6,
        verbose: bool = False
    ) -> Dict:
        """Async inference pipeline"""
        
        # Determine branch count
        branch_map = {"low": 2, "medium": 8, "high": 32}
        n_branches = branch_map.get(mode, 8)
        
        start_time = time.time()
        
        # Generate and score concurrently where possible
        if verbose:
            print(f"Running async inference with {n_branches} branches...")
        
        # Generate candidates
        candidates = await self.generate_async(
            prompt,
            n_branches,
            max_tokens,
            temperature
        )
        
        gen_time = time.time() - start_time
        
        # Score candidates
        score_start = time.time()
        scores = await self.score_async(candidates)
        score_time = time.time() - score_start
        
        # Select best
        best_idx = int(mx.argmax(mx.array(scores)))
        best_candidate = candidates[best_idx]
        best_score = scores[best_idx]
        
        # Extract response
        messages = [{"role": "user", "content": prompt}]
        prompt_template = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        if best_candidate.startswith(prompt_template):
            response = best_candidate[len(prompt_template):]
        else:
            response = best_candidate
        
        total_time = time.time() - start_time
        
        return {
            "mode": mode,
            "branches": n_branches,
            "response": response,
            "score": float(best_score),
            "all_scores": [float(s) for s in scores],
            "timing": {
                "generation": gen_time,
                "scoring": score_time,
                "total": total_time
            }
        }


def benchmark_modes(
    model_path: str,
    prompt: str,
    runs: int = 3
):
    """Benchmark performance across different modes"""
    
    engine = ParallelInferenceEngine(model_path)
    modes = ["low", "medium", "high"]
    
    results = {}
    
    for mode in modes:
        print(f"\nBenchmarking {mode} mode...")
        
        times = []
        scores = []
        
        for run in range(runs):
            result = engine.run_inference(
                prompt,
                mode=mode,
                verbose=False
            )
            
            times.append(result["timing"]["total"])
            scores.append(result["score"])
        
        results[mode] = {
            "avg_time": float(mx.mean(mx.array(times))),
            "avg_score": float(mx.mean(mx.array(scores))),
            "branches": result["branches"]
        }
    
    # Print comparison
    print("\n" + "="*50)
    print("Mode Comparison:")
    print("="*50)
    
    for mode, stats in results.items():
        print(f"\n{mode.upper()} Mode:")
        print(f"  Branches: {stats['branches']}")
        print(f"  Avg Time: {stats['avg_time']:.2f}s")
        print(f"  Avg Score: {stats['avg_score']:.4f}")
    
    engine.cleanup()
    
    return results