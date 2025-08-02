#!/usr/bin/env python
"""
XBai o4 MLX CLI - Easy interface for low/medium/high mode inference
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional
import asyncio

from parallel_inference import ParallelInferenceEngine, AsyncInferenceEngine, benchmark_modes


def print_header():
    """Print CLI header"""
    print("\n" + "="*60)
    print("  XBai o4 MLX - Test-Time Scaling Inference")
    print("  Modes: low (2x), medium (8x), high (32x)")
    print("="*60 + "\n")


def print_result(result: dict, verbose: bool = False):
    """Pretty print inference result"""
    print(f"\n{'='*50}")
    print(f"Mode: {result['mode'].upper()} ({result['branches']} branches)")
    
    # Handle both old and new result format
    if 'best_response' in result:
        # New format with detailed stats
        best_score = result['best_response']['score']
        best_idx = result['best_response']['index']
    else:
        # Fallback to old format
        best_score = result.get('score', result.get('all_scores', [0])[0] if 'all_scores' in result else 0)
        best_idx = result.get('best_idx', 0)
    
    print(f"Best Score: {best_score:.4f}")
    print(f"Total Time: {result['timing']['total']:.2f}s")
    
    # Always show key timing stats (even in non-verbose mode)
    if 'generation_phase' in result['timing']:
        print(f"  Generation: {result['timing']['generation_phase']:.2f}s")
    else:
        print(f"  Generation: {result['timing'].get('generation', 0):.2f}s")
    
    if 'scoring_phase' in result['timing']:
        print(f"  Scoring: {result['timing']['scoring_phase']:.2f}s")
    else:
        print(f"  Scoring: {result['timing'].get('scoring', 0):.2f}s")
    
    # Show tokens per second if available
    if 'tokens_per_second' in result['timing']:
        print(f"  Speed: {result['timing']['tokens_per_second']:.1f} tokens/sec")
    
    # Show score distribution
    if 'all_scores' in result:
        scores = result['all_scores']
        min_score = min(scores)
        max_score = max(scores)
        print(f"Score Range: [{min_score:.4f} - {max_score:.4f}]")
    
    # Show total tokens generated
    if 'generation_stats' in result:
        total_tokens = result['generation_stats'].get('total_tokens', 0)
        print(f"Total Tokens Generated: {total_tokens}")
    
    if verbose:
        # Verbose mode: show detailed stats
        if 'generation_stats' in result:
            stats = result['generation_stats']
            print(f"\nGeneration Details:")
            print(f"  Prompt tokens: {stats.get('prompt_tokens', 'N/A')}")
            print(f"  Total tokens generated: {stats.get('total_tokens', 'N/A')}")
            print(f"  Avg tokens/response: {stats.get('total_tokens', 0) / result['branches']:.0f}")
        
        if 'cache_hits' in result:
            print(f"  Cache hits: {result['cache_hits']}")
        
        # Show sorted responses if available
        if 'sorted_responses' in result:
            print(f"\n{'='*50}")
            print(f"TOP 5 RESPONSES (by score):")
            print("="*50)
            
            for i, resp_detail in enumerate(result['sorted_responses'][:5]):
                is_best = resp_detail['index'] == best_idx
                marker = " ✅ SELECTED" if is_best else ""
                print(f"\n{i+1}. Score: {resp_detail['score']:.4f} | Time: {resp_detail['generation_time']:.1f}s | Tokens: {resp_detail.get('tokens_generated', 'N/A')}{marker}")
                if i < 3:  # Show first 3 responses in verbose mode
                    print("-"*30)
                    # Get response text from all_responses if available
                    if 'all_responses' in result and resp_detail['index'] < len(result['all_responses']):
                        response_text = result['all_responses'][resp_detail['index']]
                        # Don't truncate the selected response, truncate others
                        if is_best:
                            # Show full selected response
                            print(response_text)
                        else:
                            # Truncate non-selected responses
                            if len(response_text) > 300:
                                response_text = response_text[:300] + "... [truncated]"
                            print(response_text)
        
        # Show all candidates with scores (old format support)
        elif 'all_responses' in result:
            print(f"\n{'='*50}")
            print(f"ALL CANDIDATE RESPONSES ({result['branches']} total):")
            print("="*50)
            
            for i, (response, score) in enumerate(zip(result['all_responses'], result['all_scores'])):
                is_best = i == best_idx
                marker = " ✅" if is_best else " ❌"
                print(f"\nCandidate {i+1} (Score: {score:.4f}){marker}:")
                print("-"*50)
                print(response)
                print("-"*50)
    else:
        # Non-verbose: show only the selected response (always in full)
        print(f"\n{'='*50}")
        print("SELECTED RESPONSE (Full):")
        print("-"*50)
        print(result['response'])
        print("-"*50)


def interactive_mode(engine: ParallelInferenceEngine, args):
    """Run interactive chat mode"""
    print_header()
    print("Interactive mode started. Type 'exit' to quit.")
    print("Commands: /mode [low|medium|high], /help")
    
    current_mode = args.mode
    
    while True:
        print(f"\n[{current_mode.upper()}] > ", end="")
        user_input = input().strip()
        
        if user_input.lower() == 'exit':
            break
        
        if user_input.startswith('/'):
            # Handle commands
            parts = user_input.split()
            cmd = parts[0]
            
            if cmd == '/mode' and len(parts) > 1:
                new_mode = parts[1].lower()
                if new_mode in ['low', 'medium', 'high']:
                    current_mode = new_mode
                    print(f"Switched to {current_mode} mode")
                else:
                    print("Invalid mode. Use: low, medium, or high")
            
            elif cmd == '/help':
                print("\nCommands:")
                print("  /mode [low|medium|high] - Switch inference mode")
                print("  /help - Show this help")
                print("  exit - Quit the program")
            
            else:
                print(f"Unknown command: {cmd}")
            
            continue
        
        # Run inference
        print(f"\nProcessing with {current_mode} mode...")
        
        try:
            result = engine.run_inference(
                user_input,
                mode=current_mode,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                verbose=args.verbose
            )
            
            print_result(result, args.verbose)
            
        except Exception as e:
            print(f"Error: {e}")


def batch_mode(engine: ParallelInferenceEngine, args):
    """Process batch of prompts from file"""
    input_file = Path(args.input)
    output_file = Path(args.output)
    
    if not input_file.exists():
        print(f"Error: Input file {input_file} not found")
        sys.exit(1)
    
    # Load prompts
    prompts = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():
                if input_file.suffix == '.jsonl':
                    data = json.loads(line)
                    prompt = data.get('prompt', data.get('question', ''))
                else:
                    prompt = line.strip()
                
                if prompt:
                    prompts.append(prompt)
    
    print(f"Loaded {len(prompts)} prompts from {input_file}")
    
    # Process each prompt
    results = []
    for i, prompt in enumerate(prompts, 1):
        print(f"\nProcessing {i}/{len(prompts)}...")
        
        try:
            result = engine.run_inference(
                prompt,
                mode=args.mode,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                verbose=args.verbose
            )
            
            result['prompt'] = prompt
            results.append(result)
            
            if args.verbose:
                print_result(result, verbose=True)
            else:
                print(f"  Score: {result['score']:.4f}, Time: {result['timing']['total']:.2f}s")
            
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                'prompt': prompt,
                'error': str(e)
            })
    
    # Save results
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"\nResults saved to {output_file}")
    
    # Print summary
    successful = [r for r in results if 'score' in r]
    if successful:
        avg_score = sum(r['score'] for r in successful) / len(successful)
        avg_time = sum(r['timing']['total'] for r in successful) / len(successful)
        
        print(f"\nSummary:")
        print(f"  Successful: {len(successful)}/{len(results)}")
        print(f"  Average score: {avg_score:.4f}")
        print(f"  Average time: {avg_time:.2f}s")


async def async_batch_mode(engine: AsyncInferenceEngine, args):
    """Process batch asynchronously for better performance"""
    input_file = Path(args.input)
    output_file = Path(args.output)
    
    # Load prompts
    prompts = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():
                if input_file.suffix == '.jsonl':
                    data = json.loads(line)
                    prompt = data.get('prompt', data.get('question', ''))
                else:
                    prompt = line.strip()
                
                if prompt:
                    prompts.append(prompt)
    
    print(f"Processing {len(prompts)} prompts asynchronously...")
    
    # Create async tasks
    tasks = []
    for prompt in prompts:
        task = engine.run_inference_async(
            prompt,
            mode=args.mode,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            verbose=False
        )
        tasks.append(task)
    
    # Run all tasks
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    output_results = []
    for prompt, result in zip(prompts, results):
        if isinstance(result, Exception):
            output_results.append({
                'prompt': prompt,
                'error': str(result)
            })
        else:
            result['prompt'] = prompt
            output_results.append(result)
    
    # Save results
    with open(output_file, 'w') as f:
        for result in output_results:
            f.write(json.dumps(result) + '\n')
    
    print(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="XBai o4 MLX - Easy inference with low/medium/high modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode with medium setting
  xbai_cli --model-path /path/to/model --mode medium
  
  # Process single prompt with high mode
  xbai_cli --model-path /path/to/model --mode high --prompt "Solve: 2^10 + 3^5"
  
  # Batch processing
  xbai_cli --model-path /path/to/model --mode medium --input prompts.jsonl --output results.jsonl
  
  # Benchmark all modes
  xbai_cli --model-path /path/to/model --benchmark --prompt "What is 2+2?"
        """
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to XBai o4 model"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["low", "medium", "high"],
        default="medium",
        help="Inference mode: low (2 branches), medium (8), high (32)"
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        help="Single prompt to process"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        help="Input file with prompts (JSONL or text)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="results.jsonl",
        help="Output file for results"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature (default: 0.6)"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32768,
        help="Maximum tokens to generate"
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum parallel workers"
    )
    
    parser.add_argument(
        "--async",
        dest="async_mode",
        action="store_true",
        help="Use async processing for batch mode"
    )
    
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark all modes"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output (shows all candidates, timing details, and scores)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.benchmark and not args.prompt and not args.input:
        # No input specified, run interactive mode
        args.interactive = True
    else:
        args.interactive = False
    
    # Initialize engine
    if args.async_mode and args.input:
        engine = AsyncInferenceEngine(
            args.model_path,
            max_workers=args.max_workers
        )
    else:
        engine = ParallelInferenceEngine(
            args.model_path,
            max_workers=args.max_workers
        )
    
    try:
        if args.benchmark:
            # Benchmark mode
            prompt = args.prompt or "What is the derivative of x^2 + 3x + 2?"
            benchmark_modes(args.model_path, prompt, runs=3)
        
        elif args.interactive:
            # Interactive mode
            interactive_mode(engine, args)
        
        elif args.input:
            # Batch mode
            if args.async_mode:
                asyncio.run(async_batch_mode(engine, args))
            else:
                batch_mode(engine, args)
        
        elif args.prompt:
            # Single prompt mode
            print_header()
            print(f"Processing with {args.mode} mode...")
            
            result = engine.run_inference(
                args.prompt,
                mode=args.mode,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                verbose=args.verbose
            )
            
            print_result(result, args.verbose)
            
            # Save to file if output specified
            if args.output != "results.jsonl":
                result['prompt'] = args.prompt
                with open(args.output, 'w') as f:
                    f.write(json.dumps(result) + '\n')
                print(f"\nResult saved to {args.output}")
    
    finally:
        engine.cleanup()


if __name__ == "__main__":
    main()