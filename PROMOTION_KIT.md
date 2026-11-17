# 🚀 Zero-Budget Traffic & Backlink Seeding Kit

Use these ready-to-copy templates to seed initial developer traffic and high-authority backlinks so Google and Bing index all 1,540+ pages.

---

## 1. Reddit Post — `r/LocalLLaMA` / `r/ollama` / `r/selfhosted`

**Title:**
`I built an open, zero-bloat calculator for LLM VRAM (FP16/Q8/Q4 + KV cache), Home GPU Electricity ($/kWh), and API vs Self-Hosting break-even across 440+ models`

**Body:**
```markdown
Hey r/LocalLLaMA,

Whenever a new model drops, I always end up doing napkin math to figure out:
1. Whether it'll fit on my local VRAM in Q4_K_M or Q8_0 once I crank the context window up to 32k–64k
2. What my home rig's 24/7 power draw is actually costing me per month vs. serverless API tokens or renting an on-demand GPU pod

So I built a fast, static calculator directory that auto-syncs daily with OpenRouter's model list (440+ models, separating open-weight models from closed APIs):

- **VRAM + KV Cache & Home Electricity Calculator:** https://stigla610.github.io/ai-cost-engine/vram-electricity-calculator.html
- **Batch Workload Simulator (ranks 260+ models by monthly cost for your exact daily token volume):** https://stigla610.github.io/ai-cost-engine/
- **GitHub Repo:** https://github.com/stigla610/ai-cost-engine

Features added based on community feedback:
- Hardware quick-presets (RTX 4060 Ti 16GB, 3090/4090 24GB, 5090 32GB, Dual 3090 48GB, Mac Studio 64GB/192GB, A100/H100 80GB)
- Proper MoE total-parameter VRAM sizing (DeepSeek V3/R1 671B, Qwen3 235B-A22B, Llama 4 Maverick/Scout, Mixtral 8x22B)
- One-click shareable deep links (`?p=70&ctx=32768&w=550&kwh=0.18`) and Markdown table export for pasting into issues/comments

Let me know if any model parameter counts or KV-cache multipliers look off and I'll patch the generator script!
```

---

## 2. Hacker News — `Show HN` Submission

* **Title:** `Show HN: TokenBench – Open LLM API Cost, VRAM, and GPU Break-Even Calculator`
* **URL:** `https://stigla610.github.io/ai-cost-engine/`
* **First Comment:**
```text
Hi HN, I built TokenBench to answer a question that kept coming up in production AI projects: "At what daily token volume does it actually make sense to stop paying per-token API rates and run vLLM/Ollama on a dedicated GPU?"

The site pulls live pricing and context windows daily for 440+ models across OpenAI, Anthropic, Google, DeepSeek, Meta Llama, Qwen, and Mistral. You can drag the daily input/output token sliders on the main page to sort all models by your projected monthly bill, or use the VRAM & electricity calculator to estimate FP16/Q8/Q4 memory + KV cache overhead and local $/kWh power costs.

Source code (single zero-dependency Python static generator running on GitHub Actions): https://github.com/stigla610/ai-cost-engine
```

---

## 3. Hugging Face Space (Takes 60 Seconds — High-Authority Backlink)

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Name it `llm-vram-cost-calculator`, select **Static** as the Space SDK, and make it **Public**.
3. Upload the two files inside the [`./huggingface-space/`](./huggingface-space/) folder (`README.md` and `index.html`).
4. Hugging Face Spaces get indexed rapidly by Google and pass strong domain authority to `https://stigla610.github.io/ai-cost-engine/`.
