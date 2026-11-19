---
title: "How Much VRAM Do Open-Weight LLMs Actually Need at 8K vs. 32K Context?"
published: true
tags: localllama, ollama, vllm, machinelearning
canonical_url: https://stigla610.github.io/ai-cost-engine/vram-electricity-calculator.html
---

Most VRAM cheat sheets only count model weights at tiny context lengths. Once you crank your context window up to 32K–128K tokens for coding or RAG, the KV cache adds gigabytes of extra VRAM.

Here are the exact requirements (weights + KV-cache headroom) across top open-weight models from [TokenBench](https://stigla610.github.io/ai-cost-engine/vram-electricity-calculator.html):

| Open-Weight Model | Params | Q4 @ 8K Context | Q4 @ 32K Context | Q8 @ 8K Context | FP16 @ 8K |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [DeepSeek: DeepSeek V3](https://stigla610.github.io/ai-cost-engine/vram/deepseek-deepseek-chat-vram-requirements.html) | ~671B | **436.2 GB** | 486.5 GB | 771.6 GB | 1476.2 GB |
| [Meta: Llama 4 Maverick](https://stigla610.github.io/ai-cost-engine/vram/meta-llama-llama-4-maverick-vram-requirements.html) | ~400B | **260.0 GB** | 290.0 GB | 460.0 GB | 880.0 GB |
| [Qwen: Qwen3 32B](https://stigla610.github.io/ai-cost-engine/vram/qwen-qwen3-32b-vram-requirements.html) | ~32B | **20.8 GB** | 23.2 GB | 36.8 GB | 70.4 GB |
| [Mistral Large](https://stigla610.github.io/ai-cost-engine/vram/mistralai-mistral-large-vram-requirements.html) | ~123B | **80.0 GB** | 89.2 GB | 141.4 GB | 270.6 GB |
| [NVIDIA: Nemotron 3 Nano 30B A3B](https://stigla610.github.io/ai-cost-engine/vram/nvidia-nemotron-3-nano-30b-a3b-vram-requirements.html) | ~30B | **19.5 GB** | 21.8 GB | 34.5 GB | 66.0 GB |
| [Cohere: Command A](https://stigla610.github.io/ai-cost-engine/vram/cohere-command-a-vram-requirements.html) | ~111B | **72.2 GB** | 80.5 GB | 127.6 GB | 244.2 GB |
| [Z.ai: GLM 4.5](https://stigla610.github.io/ai-cost-engine/vram/z-ai-glm-4-5-vram-requirements.html) | ~355B | **230.8 GB** | 257.4 GB | 408.2 GB | 781.0 GB |
| [MoonshotAI: Kimi K2 0711](https://stigla610.github.io/ai-cost-engine/vram/moonshotai-kimi-k2-vram-requirements.html) | ~1000B | **650.0 GB** | 725.0 GB | 1150.0 GB | 2200.0 GB |
| [Microsoft: Phi 4](https://stigla610.github.io/ai-cost-engine/vram/microsoft-phi-4-vram-requirements.html) | ~14B | **9.1 GB** | 10.2 GB | 16.1 GB | 30.8 GB |
| [DeepSeek: R1 0528](https://stigla610.github.io/ai-cost-engine/vram/deepseek-deepseek-r1-0528-vram-requirements.html) | ~671B | **436.2 GB** | 486.5 GB | 771.6 GB | 1476.2 GB |
| [Meta: Llama 4 Scout](https://stigla610.github.io/ai-cost-engine/vram/meta-llama-llama-4-scout-vram-requirements.html) | ~109B | **70.9 GB** | 79.0 GB | 125.3 GB | 239.8 GB |
| [Qwen2.5 72B Instruct](https://stigla610.github.io/ai-cost-engine/vram/qwen-qwen-2-5-72b-instruct-vram-requirements.html) | ~72B | **46.8 GB** | 52.2 GB | 82.8 GB | 158.4 GB |
| [Mistral: Mistral Small 3.2 24B](https://stigla610.github.io/ai-cost-engine/vram/mistralai-mistral-small-3-2-24b-instruct-vram-requirements.html) | ~24B | **15.6 GB** | 17.4 GB | 27.6 GB | 52.8 GB |
| [NVIDIA: Nemotron 3 Super](https://stigla610.github.io/ai-cost-engine/vram/nvidia-nemotron-3-super-120b-a12b-vram-requirements.html) | ~120B | **78.0 GB** | 87.0 GB | 138.0 GB | 264.0 GB |
| [Cohere: Command A+](https://stigla610.github.io/ai-cost-engine/vram/cohere-command-a-plus-vram-requirements.html) | ~111B | **72.2 GB** | 80.5 GB | 127.6 GB | 244.2 GB |
| [Z.ai: GLM 4.5 Air](https://stigla610.github.io/ai-cost-engine/vram/z-ai-glm-4-5-air-vram-requirements.html) | ~106B | **68.9 GB** | 76.9 GB | 121.9 GB | 233.2 GB |

### Auto-Check Your GPU in 1 Second
Run the zero-dependency Python CLI (`nvidia-smi` + Apple Silicon Unified Memory detection) to see which models fit on your hardware right now with copy-paste `ollama run` and `vllm serve` commands:

```bash
curl -sL https://stigla610.github.io/ai-cost-engine/tokenbench.py | python3 -
```

Interactive calculator + electricity (`$/kWh`) estimator: [https://stigla610.github.io/ai-cost-engine/vram-electricity-calculator.html](https://stigla610.github.io/ai-cost-engine/vram-electricity-calculator.html)
