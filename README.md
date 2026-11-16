# TokenBench — Automated AI & Cloud GPU Cost Calculator Engine

This project is a **100% automated programmatic SEO & affiliate web application** that costs **$0/month** to run and updates itself daily.

## How It Makes Money Automatically
1. **Live Data Ingestion (`build_site.py`)**: Every day, the script pulls real-time pricing and context window data for 250+ AI models (OpenAI, Anthropic, Google Gemini, DeepSeek, Llama, Qwen, Mistral) from OpenRouter's free public API.
2. **402 Programmatic Calculator & Comparison Pages (`dist/`)**:
   - **250 Individual Model Cost Calculators** (`dist/models/*.html`) with interactive daily/monthly token sliders and "API vs. Self-Hosted GPU Break-Even" recommendations.
   - **150 Head-to-Head Model Comparisons** (`dist/compare/*-vs-*.html`) targeting high-intent search queries (e.g., *"GPT-4o vs DeepSeek R1 API pricing"*).
   - **Cloud GPU Pricing Directory** (`dist/gpu-cloud-pricing.html`) comparing H100, A100, and RTX 4090 hourly rental costs.
   - **Auto-generated `sitemap.xml` & FAQ JSON-LD Schema** for search engine indexing.
3. **Recurring Cloud GPU & VPS Affiliate Commissions**: Every calculator and comparison page recommends Cloud GPU/VPS infrastructure where developers can claim free credits or rent GPUs, embedding your referral links from [`config.json`](./config.json).
4. **Zero-Maintenance Daily Cloud Autopilot (`.github/workflows/daily-autopilot.yml`)**: Runs on GitHub Actions every day at 06:00 UTC for free, rebuilds the site with any new models or price drops, and deploys straight to GitHub Pages—even when your PC is turned off.

---

## One-Time Setup (5 Minutes, $0 Cost)

### Step 1: Add Your Free Referral / Affiliate IDs in `config.json`
Open [`config.json`](./config.json) (or tell Antigravity your referral links and I will update it for you):
- **RunPod** (Recurring % of all GPU spend): Sign up free at `https://www.runpod.io` → Referrals → Copy your `?ref=...` link.
- **Vast.ai** (Ongoing GPU rental commission): Sign up free at `https://cloud.vast.ai` → Affiliate/Referral link.
- **Vultr** (Up to $100 per signup): `https://www.vultr.com/company/referral-program/`
- **DigitalOcean** ($25–$200 per signup): `https://www.digitalocean.com/referral-program`
- **Hetzner** (€10 per referral): `https://www.hetzner.com`

### Step 2: Push to a Free GitHub Repository & Enable GitHub Pages
1. Create a free public repository on GitHub named `ai-cost-engine`.
2. In GitHub repository **Settings → Pages**, set **Build and deployment → Source** to **GitHub Actions**.
3. Push this folder to GitHub:
   ```powershell
   git remote add origin https://github.com/YOUR_USERNAME/ai-cost-engine.git
   git push -u origin main
   ```
That's it! GitHub Actions will automatically build and host your site at `https://YOUR_USERNAME.github.io/ai-cost-engine/` and update it every single day at 06:00 UTC with zero input from you.
