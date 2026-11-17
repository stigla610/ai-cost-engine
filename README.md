# ⚡ TokenBench — AI API Pricing, LLM VRAM & Cloud GPU Break-Even Engine

[![Live Demo](https://img.shields.io/badge/Live_Calculators-TokenBench-38bdf8?style=for-the-badge)](https://stigla610.github.io/ai-cost-engine/)
[![Models Tracked](https://img.shields.io/badge/Models_Tracked-440%2B-10b981?style=for-the-badge)](https://stigla610.github.io/ai-cost-engine/index.html)
[![Auto-Sync](https://img.shields.io/badge/Updated-Daily_via_GitHub_Actions-818cf8?style=for-the-badge)](https://stigla610.github.io/ai-cost-engine/gpu-cloud-pricing.html)

**TokenBench** is an open, daily-updated benchmark and calculator suite for comparing **LLM API token pricing**, estimating **local GPU VRAM requirements** (FP16, Q8_0, and 4-bit GGUF/AWQ/EXL2 quantization + KV cache overhead), and finding the exact **daily token break-even point** between serverless APIs and dedicated Cloud GPUs.

👉 **Launch the Live Web App:** **[https://stigla610.github.io/ai-cost-engine/](https://stigla610.github.io/ai-cost-engine/)**

---

## 🛠️ Interactive Calculators & Directories

* **[Batch Workload API Cost Simulator](https://stigla610.github.io/ai-cost-engine/index.html)** — Slide your daily input/output token volume to rank 260+ models (OpenAI, Anthropic, Google Gemini, DeepSeek, Meta Llama, Qwen, Mistral, xAI Grok) from cheapest to most expensive per month.
* **[LLM VRAM & Home GPU Electricity Calculator](https://stigla610.github.io/ai-cost-engine/vram-electricity-calculator.html)** — Calculate exact GPU memory needed for 1B–405B+ models across context lengths (`2K` to `128K` tokens) and compare your local PC's `$/kWh` power bill against cloud GPU instances.
* **[Cloud GPU Hourly Price Matrix](https://stigla610.github.io/ai-cost-engine/gpu-cloud-pricing.html)** — Compare on-demand RTX 4090, NVIDIA A100 (80GB), and NVIDIA H100 (80GB) hourly and monthly rates.

---

## 📊 Quick Reference: LLM VRAM Requirements by Model Size

Memory estimates include model weights plus baseline 8K context KV-cache overhead:

| Parameter Size | Example Open-Weight Models | 4-Bit (Q4_K_M / AWQ) | 8-Bit (Q8_0 / FP8) | 16-Bit (FP16 / BF16) | Recommended Hardware |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **7B – 8B** | [Llama 3.1 8B](https://stigla610.github.io/ai-cost-engine/vram/meta-llama-llama-3-1-8b-instruct-vram-requirements.html), Qwen3 8B, Ministral 8B | **5.2 GB** | **9.2 GB** | **17.6 GB** | RTX 3060 12GB / RTX 4060 Ti 16GB |
| **14B** | [Microsoft Phi-4](https://stigla610.github.io/ai-cost-engine/vram/microsoft-phi-4-vram-requirements.html), [Qwen3 14B](https://stigla610.github.io/ai-cost-engine/vram/qwen-qwen3-14b-vram-requirements.html) | **9.1 GB** | **16.1 GB** | **30.8 GB** | RTX 4070 Ti 16GB / RTX 4090 24GB |
| **24B – 32B** | [Qwen3 32B](https://stigla610.github.io/ai-cost-engine/vram/qwen-qwen3-32b-vram-requirements.html), [Mistral Small 24B](https://stigla610.github.io/ai-cost-engine/vram/mistralai-mistral-small-3-2-24b-instruct-vram-requirements.html), [Gemma 4 31B](https://stigla610.github.io/ai-cost-engine/vram/google-gemma-4-31b-it-vram-requirements.html) | **15.6 – 20.8 GB** | **27.6 – 36.8 GB** | **52.8 – 70.4 GB** | 1x RTX 3090 / 4090 (24GB) |
| **70B – 72B** | [Llama 3.3 70B](https://stigla610.github.io/ai-cost-engine/vram/meta-llama-llama-3-3-70b-instruct-vram-requirements.html), [Qwen2.5 72B](https://stigla610.github.io/ai-cost-engine/vram/qwen-qwen-2-5-72b-instruct-vram-requirements.html) | **45.5 – 46.8 GB** | **80.5 – 82.8 GB** | **154.0 – 158.4 GB** | 2x RTX 3090/4090 (48GB) or 1x A100 80GB |
| **109B – 123B** | [Llama 4 Scout](https://stigla610.github.io/ai-cost-engine/vram/meta-llama-llama-4-scout-vram-requirements.html), [Mistral Large](https://stigla610.github.io/ai-cost-engine/vram/mistralai-mistral-large-vram-requirements.html), [gpt-oss-120b](https://stigla610.github.io/ai-cost-engine/vram/openai-gpt-oss-120b-vram-requirements.html) | **70.8 – 80.0 GB** | **125.4 – 141.5 GB** | **239.8 – 270.6 GB** | 1x–2x A100 / H100 80GB or Mac Ultra |
| **235B – 400B** | [Qwen3 235B MoE](https://stigla610.github.io/ai-cost-engine/vram/qwen-qwen3-235b-a22b-2507-vram-requirements.html), [Llama 4 Maverick](https://stigla610.github.io/ai-cost-engine/vram/meta-llama-llama-4-maverick-vram-requirements.html) | **152.8 – 260.0 GB** | **270.2 – 460.0 GB** | **517.0 – 880.0 GB** | 2x–4x H100 80GB Cluster |
| **671B MoE** | [DeepSeek V3](https://stigla610.github.io/ai-cost-engine/vram/deepseek-deepseek-chat-vram-requirements.html), [DeepSeek R1](https://stigla610.github.io/ai-cost-engine/vram/deepseek-deepseek-r1-0528-vram-requirements.html) | **436.2 GB** | **771.6 GB** | **1,476.2 GB** | 6x–8x H100 / A100 80GB Node |

---

## ⚖️ Popular Head-to-Head API Cost Comparisons

* [OpenAI GPT-5 vs Anthropic Claude Sonnet 4.5](https://stigla610.github.io/ai-cost-engine/compare/openai-gpt-5-vs-anthropic-claude-sonnet-4-5.html)
* [OpenAI GPT-5 vs Google Gemini 2.5 Pro](https://stigla610.github.io/ai-cost-engine/compare/openai-gpt-5-vs-google-gemini-2-5-pro.html)
* [OpenAI GPT-5 vs DeepSeek V3](https://stigla610.github.io/ai-cost-engine/compare/openai-gpt-5-vs-deepseek-deepseek-chat.html)
* [OpenAI GPT-5 vs Meta Llama 4 Maverick](https://stigla610.github.io/ai-cost-engine/compare/openai-gpt-5-vs-meta-llama-llama-4-maverick.html)
* [OpenAI GPT-5 vs Qwen3 32B](https://stigla610.github.io/ai-cost-engine/compare/openai-gpt-5-vs-qwen-qwen3-32b.html)

---

## 🚀 Building Locally

```bash
python build_site.py
```

Generates all static HTML calculators, JSON-LD structured data, SVG brand assets, and `sitemap.xml` inside `./dist`.
