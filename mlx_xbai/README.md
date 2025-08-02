# XBai o4 MLX Implementation

MLX-optimized implementation of XBai o4 with low/medium/high inference modes for Apple Silicon.

## Features

- **Three inference modes**: Low (2x), Medium (8x), High (32x) test-time scaling
- **Parallel generation**: Efficient candidate generation using MLX
- **Process Reward Model**: Integrated scoring system for selecting best solutions
- **Interactive CLI**: Easy-to-use command-line interface
- **Batch processing**: Process multiple prompts efficiently
- **Async support**: Asynchronous processing for better performance

## Installation

```bash
# Install MLX and dependencies
pip install mlx mlx-lm numpy

# Clone XBai o4
git clone https://github.com/MetaStoneTec/XBai-o4.git
cd XBai-o4/mlx_xbai

# Make CLI executable
chmod +x xbai_cli.py
```

## Quick Start

### Interactive Mode

```bash
python xbai_cli.py --model-path /path/to/xbai-o4-model --mode medium
```

In interactive mode, you can:
- Type prompts and get responses
- Switch modes with `/mode [low|medium|high]`
- Type `exit` to quit

### Single Prompt

```bash
# Low mode (fast, 2 candidates)
python xbai_cli.py --model-path /path/to/model --mode low \
  --prompt "What is the derivative of x^2?"

# Medium mode (balanced, 8 candidates)
python xbai_cli.py --model-path /path/to/model --mode medium \
  --prompt "Solve: 2^10 + 3^5"

# High mode (best quality, 32 candidates)
python xbai_cli.py --model-path /path/to/model --mode high \
  --prompt "Prove that sqrt(2) is irrational"
```

### Batch Processing

```bash
# Process prompts from file
python xbai_cli.py --model-path /path/to/model --mode medium \
  --input prompts.jsonl --output results.jsonl

# Async batch processing (faster)
python xbai_cli.py --model-path /path/to/model --mode medium \
  --input prompts.jsonl --output results.jsonl --async
```

### Benchmark Modes

```bash
# Compare performance across all modes
python xbai_cli.py --model-path /path/to/model --benchmark \
  --prompt "What is 2+2?"
```

## How It Works

1. **Generation Phase**: Creates multiple solution candidates in parallel
2. **Scoring Phase**: Each candidate is evaluated by the Process Reward Model
3. **Selection Phase**: The highest-scoring solution is selected

### Mode Comparison

| Mode | Branches | Speed | Quality | Use Case |
|------|----------|-------|---------|----------|
| Low | 2 | Fast (~5s) | Good | Quick answers, simple problems |
| Medium | 8 | Moderate (~15s) | Better | Most problems, balanced approach |
| High | 32 | Slow (~45s) | Best | Complex reasoning, critical tasks |

## Python API

```python
from parallel_inference import ParallelInferenceEngine

# Initialize engine
engine = ParallelInferenceEngine(
    model_path="/path/to/xbai-o4",
    max_workers=4
)

# Run inference
result = engine.run_inference(
    prompt="Solve: x^2 - 5x + 6 = 0",
    mode="medium",  # or "low", "high"
    temperature=0.6,
    max_tokens=32768,
    verbose=True
)

print(f"Solution: {result['response']}")
print(f"Score: {result['score']}")
print(f"Time: {result['timing']['total']}s")

# Clean up
engine.cleanup()
```

## Input Format

### JSONL Format
```json
{"prompt": "What is 2+2?"}
{"prompt": "Solve: x^2 = 16"}
{"question": "Prove that e^(iπ) + 1 = 0"}
```

### Plain Text Format
```
What is 2+2?
Solve: x^2 = 16
Prove that e^(iπ) + 1 = 0
```

## Output Format

```json
{
  "mode": "medium",
  "branches": 8,
  "response": "The answer is 4",
  "score": 0.8756,
  "all_scores": [0.7234, 0.8756, 0.6891, ...],
  "timing": {
    "generation": 12.34,
    "scoring": 2.56,
    "total": 14.90
  },
  "prompt": "What is 2+2?"
}
```

## Advanced Options

```bash
# Custom temperature
python xbai_cli.py --model-path /path/to/model --mode medium \
  --prompt "..." --temperature 0.8

# Limit max tokens
python xbai_cli.py --model-path /path/to/model --mode low \
  --prompt "..." --max-tokens 4096

# Increase parallel workers
python xbai_cli.py --model-path /path/to/model --mode high \
  --prompt "..." --max-workers 8

# Verbose output
python xbai_cli.py --model-path /path/to/model --mode medium \
  --prompt "..." --verbose
```

## Performance Tips

1. **Use appropriate mode**: Start with low mode, upgrade only when needed
2. **Batch processing**: Use `--async` flag for multiple prompts
3. **Adjust workers**: Increase `--max-workers` on systems with more cores
4. **Cache results**: The system automatically caches scores for repeated content

## Limitations

- The current implementation uses a simplified scoring model (full PRM integration pending)
- Requires model weights in MLX-compatible format
- Performance depends on available RAM and compute

## Contributing

Contributions welcome! Areas for improvement:
- Full Process Reward Model integration
- Model quantization support
- Streaming generation
- Web UI interface

## License

Same as XBai o4 main repository

## Citation

```bibtex
@misc{wang2025testtimescalingreflectivegenerative,
  title={Test-Time Scaling with Reflective Generative Model}, 
  author={Zixiao Wang and others},
  year={2025},
  eprint={2507.01951},
  archivePrefix={arXiv}
}
```