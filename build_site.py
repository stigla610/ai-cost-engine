import urllib.request
import json
import os
import re
import html
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
DIST_DIR = os.path.join(BASE_DIR, "dist")

def slugify(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def fetch_models():
    url = "https://openrouter.ai/api/v1/models"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "TokenBench-AutoBuilder/1.0"}
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    models = []
    seen_slugs = set()
    for item in raw.get("data", []):
        model_id = item.get("id", "")
        name = item.get("name", model_id)
        pricing = item.get("pricing") or {}
        try:
            prompt_per_tok = float(pricing.get("prompt", 0) or 0)
            completion_per_tok = float(pricing.get("completion", 0) or 0)
        except (ValueError, TypeError):
            continue

        # Convert per-token to per-1M tokens
        input_1m = round(prompt_per_tok * 1_000_000, 4)
        output_1m = round(completion_per_tok * 1_000_000, 4)

        # Skip broken negative pricing entries
        if input_1m < 0 or output_1m < 0:
            continue

        context_length = int(item.get("context_length") or 4096)
        slug = slugify(model_id)
        if not slug or slug in seen_slugs:
            continue
        seen_slugs.add(slug)

        provider = model_id.split("/")[0].title() if "/" in model_id else "AI"
        is_open_weight = any(
            kw in model_id.lower()
            for kw in ["llama", "qwen", "mistral", "mixtral", "deepseek", "gemma", "phi-", "yi-", "nemotron"]
        )

        # Recommend a GPU tier based on model name hints
        if any(k in model_id.lower() for k in ["405b", "671b", "r1", "v3"]):
            gpu_tier_idx = 3
        elif any(k in model_id.lower() for k in ["70b", "72b", "65b", "120b", "large"]):
            gpu_tier_idx = 1
        elif any(k in model_id.lower() for k in ["32b", "34b", "27b", "mixtral"]):
            gpu_tier_idx = 1
        else:
            gpu_tier_idx = 0

        models.append({
            "id": model_id,
            "name": name,
            "slug": slug,
            "provider": provider,
            "input_1m": input_1m,
            "output_1m": output_1m,
            "blended_1m": round((input_1m * 0.7) + (output_1m * 0.3), 4),
            "context_length": context_length,
            "description": (item.get("description") or f"Live API token pricing, context window specs, and self-hosted GPU break-even analysis for {name}.")[:280],
            "is_open_weight": is_open_weight,
            "gpu_tier_idx": gpu_tier_idx,
        })

    # Prioritize paid commercial & top open-weight models so calculators are meaningful
    paid_models = [m for m in models if m["input_1m"] > 0 or m["output_1m"] > 0]
    priority_keywords = ["gpt-4", "gpt-5", "claude", "gemini", "deepseek", "llama-3", "qwen", "mistral", "o1", "o3", "grok", "command"]
    def sort_key(m):
        score = 0
        for i, kw in enumerate(priority_keywords):
            if kw in m["id"].lower():
                score -= (len(priority_keywords) - i) * 10
        return (score, m["name"])

    paid_models.sort(key=sort_key)
    return paid_models

def shared_head(title, description, canonical_url, json_ld=None):
    schema_tag = ""
    if json_ld:
        schema_tag = f'<script type="application/ld+json">{json.dumps(json_ld)}</script>'
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{html.escape(title)}</title>
  <meta name="description" content="{html.escape(description)}" />
  <link rel="canonical" href="{html.escape(canonical_url)}" />
  <meta property="og:title" content="{html.escape(title)}" />
  <meta property="og:description" content="{html.escape(description)}" />
  <meta property="og:type" content="website" />
  <meta property="og:url" content="{html.escape(canonical_url)}" />
  {schema_tag}
  <style>
    :root {{
      --bg: #0b0f19;
      --card: #131b2e;
      --card-hover: #1a2540;
      --border: #23304f;
      --text: #e2e8f0;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --accent-green: #10b981;
      --accent-purple: #818cf8;
      --warn: #f59e0b;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
    }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .container {{ max-width: 1180px; margin: 0 auto; padding: 0 20px; }}
    header {{
      border-bottom: 1px solid var(--border);
      background: rgba(11, 15, 25, 0.9);
      position: sticky;
      top: 0;
      z-index: 50;
      backdrop-filter: blur(8px);
    }}
    .nav {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 16px 0;
      flex-wrap: wrap;
      gap: 12px;
    }}
    .brand {{
      font-size: 1.25rem;
      font-weight: 800;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .brand span {{ color: var(--accent); }}
    .nav-links {{ display: flex; gap: 20px; font-size: 0.95rem; align-items: center; flex-wrap: wrap; }}
    .btn-cta {{
      background: linear-gradient(135deg, #0284c7, #4f46e5);
      color: #fff !important;
      padding: 9px 16px;
      border-radius: 8px;
      font-weight: 600;
      display: inline-block;
      transition: transform 0.15s ease, opacity 0.15s ease;
      text-decoration: none !important;
    }}
    .btn-cta:hover {{ transform: translateY(-1px); opacity: 0.95; }}
    .btn-green {{
      background: linear-gradient(135deg, #059669, #10b981);
    }}
    .hero {{ padding: 42px 0 28px; }}
    .hero h1 {{ font-size: 2.2rem; line-height: 1.25; margin-bottom: 12px; color: #fff; }}
    .hero p {{ color: var(--muted); font-size: 1.08rem; max-width: 820px; }}
    .grid-2 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); gap: 20px; margin: 24px 0; }}
    .grid-3 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 18px; margin: 24px 0; }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 22px;
    }}
    .card h2, .card h3 {{ color: #fff; margin-bottom: 10px; }}
    .badge {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 0.78rem;
      font-weight: 600;
      background: rgba(56, 189, 248, 0.15);
      color: var(--accent);
      margin-bottom: 10px;
    }}
    .badge-green {{
      background: rgba(16, 185, 129, 0.15);
      color: var(--accent-green);
    }}
    .stat-row {{
      display: flex;
      justify-content: space-between;
      padding: 10px 0;
      border-bottom: 1px solid var(--border);
      font-size: 0.95rem;
    }}
    .stat-row:last-child {{ border-bottom: none; }}
    .price-big {{ font-size: 1.8rem; font-weight: 800; color: var(--accent-green); }}
    .slider-group {{ margin-bottom: 16px; }}
    .slider-group label {{ display: flex; justify-content: space-between; font-weight: 600; margin-bottom: 6px; font-size: 0.92rem; }}
    input[type="range"] {{ width: 100%; accent-color: var(--accent); cursor: pointer; }}
    input[type="text"], input[type="number"], select {{
      width: 100%;
      padding: 10px 14px;
      background: #0b0f19;
      border: 1px solid var(--border);
      color: #fff;
      border-radius: 8px;
      font-size: 0.95rem;
    }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 0.92rem; }}
    th, td {{ padding: 12px 10px; text-align: left; border-bottom: 1px solid var(--border); }}
    th {{ color: var(--muted); font-weight: 600; background: rgba(255,255,255,0.02); }}
    tr:hover td {{ background: rgba(255,255,255,0.02); }}
    .affiliate-box {{
      background: linear-gradient(135deg, rgba(14, 165, 233, 0.12), rgba(79, 70, 229, 0.12));
      border: 1px solid var(--accent);
      border-radius: 12px;
      padding: 20px;
      margin: 24px 0;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 16px;
    }}
    footer {{
      margin-top: 60px;
      border-top: 1px solid var(--border);
      padding: 32px 0;
      color: var(--muted);
      font-size: 0.88rem;
    }}
  </style>
</head>
"""

def shared_nav(prefix=""):
    return f"""<body>
<header>
  <div class="container nav">
    <a href="{prefix}index.html" class="brand">⚡ Token<span>Bench</span></a>
    <nav class="nav-links">
      <a href="{prefix}index.html">API Cost Calculator</a>
      <a href="{prefix}gpu-cloud-pricing.html">Cloud GPU Rates</a>
      <a href="{prefix}index.html#comparisons">Model Comparisons</a>
      <a href="{prefix}gpu-cloud-pricing.html" class="btn-cta">Rent Cloud GPUs (From $0.20/hr)</a>
    </nav>
  </div>
</header>
"""

def shared_footer(config, updated_str, prefix=""):
    return f"""<footer>
  <div class="container">
    <p><strong>{html.escape(config['site_name'])}</strong> — Automatically updated daily with live API pricing &amp; cloud GPU benchmarks. Last auto-sync: {updated_str}.</p>
    <p style="margin-top:8px;">links to cloud GPU &amp; infrastructure providers may include partner referral links at zero extra cost to you, which keeps this calculator 100% free.</p>
    <p style="margin-top:8px;">
      <a href="{prefix}index.html">All AI Models</a> ·
      <a href="{prefix}gpu-cloud-pricing.html">Cloud GPU Price Directory</a> ·
      <a href="{prefix}sitemap.xml">XML Sitemap</a>
    </p>
  </div>
</footer>
</body>
</html>
"""

def build_model_page(m, all_models, config, updated_str):
    base_url = config["base_url"].rstrip("/")
    canonical = f"{base_url}/models/{m['slug']}.html"
    gpu = config["gpu_tiers"][m["gpu_tier_idx"]]
    aff = config["affiliates"][gpu["provider_key"]]
    alt_key = "vastai" if gpu["provider_key"] != "vastai" else "runpod"
    alt_aff = config["affiliates"][alt_key]

    # Calculate daily break-even tokens when self-hosting vs API
    daily_gpu_cost = round(gpu["hourly_usd"] * 24, 2)
    blended = max(m["blended_1m"], 0.05)
    breakeven_m_tokens_day = round(daily_gpu_cost / blended, 1)

    title = f"{m['name']} API Cost Calculator & Self-Host GPU Break-Even (2026)"
    desc = f"Calculate exact {m['name']} token costs (${m['input_1m']}/1M input, ${m['output_1m']}/1M output), context window ({m['context_length']:,} tokens), and Cloud GPU self-hosting break-even."

    json_ld = {
        "@context": "https://schema.org",
        "@type": "FAQPage",
        "mainEntity": [
            {
                "@type": "Question",
                "name": f"How much does the {m['name']} API cost per 1M tokens?",
                "acceptedAnswer": {
                    "@type": "Answer",
                    "text": f"{m['name']} costs ${m['input_1m']} per 1M input tokens and ${m['output_1m']} per 1M output tokens, with a context window of {m['context_length']:,} tokens."
                }
            },
            {
                "@type": "Question",
                "name": f"When is it cheaper to self-host {m['name']} on a Cloud GPU?",
                "acceptedAnswer": {
                    "@type": "Answer",
                    "text": f"Renting a dedicated {gpu['name']} costs roughly ${gpu['hourly_usd']}/hour (${daily_gpu_cost}/day). If your workload exceeds {breakeven_m_tokens_day}M blended tokens per day, self-hosting or using dedicated GPU instances on {aff['name']} can lower your unit costs."
                }
            }
        ]
    }

    # Find 6 related models for internal SEO linking
    related = [x for x in all_models if x["slug"] != m["slug"]][:8]
    related_links = "".join(
        f'<li><a href="{x["slug"]}.html">{html.escape(x["name"])} Cost Calculator (${x["input_1m"]}/1M in)</a></li>'
        for x in related
    )

    page = shared_head(title, desc, canonical, json_ld)
    page += shared_nav(prefix="../")
    page += f"""
<main class="container">
  <section class="hero">
    <span class="badge">{html.escape(m['provider'])} Model Pricing &amp; Calculator</span>
    <h1>{html.escape(m['name'])} API Cost Calculator</h1>
    <p>{html.escape(m['description'])}</p>
  </section>

  <div class="grid-2">
    <div class="card">
      <h2>Interactive Monthly Spend Estimator</h2>
      <p style="color:var(--muted);font-size:0.9rem;margin-bottom:16px;">Adjust sliders to calculate your daily and monthly bill for <strong>{html.escape(m['name'])}</strong>.</p>

      <div class="slider-group">
        <label><span>Input Tokens per Day</span> <span id="inVal">2,000,000</span></label>
        <input type="range" id="inSlider" min="100000" max="50000000" step="100000" value="2000000" oninput="recalc()" />
      </div>

      <div class="slider-group">
        <label><span>Output Tokens per Day</span> <span id="outVal">500,000</span></label>
        <input type="range" id="outSlider" min="50000" max="20000000" step="50000" value="500000" oninput="recalc()" />
      </div>

      <div class="stat-row">
        <span>Estimated Daily API Cost</span>
        <strong id="dailyCost">$0.00</strong>
      </div>
      <div class="stat-row">
        <span>Estimated Monthly API Cost (30 days)</span>
        <strong id="monthlyCost" class="price-big">$0.00</strong>
      </div>
      <div class="stat-row">
        <span>24/7 Dedicated {html.escape(gpu['name'])} GPU Cost</span>
        <strong id="gpuMonthlyCost">${round(daily_gpu_cost * 30, 2):,.2f}/mo</strong>
      </div>
      <div id="verdictBox" style="margin-top:16px;padding:12px;border-radius:8px;background:rgba(16,185,129,0.12);border:1px solid var(--accent-green);font-size:0.92rem;"></div>
    </div>

    <div class="card">
      <h2>Official Token Pricing &amp; Specs</h2>
      <div class="stat-row">
        <span>Model ID</span>
        <code>{html.escape(m['id'])}</code>
      </div>
      <div class="stat-row">
        <span>Input Price (per 1M tokens)</span>
        <strong>${m['input_1m']:.4f}</strong>
      </div>
      <div class="stat-row">
        <span>Output Price (per 1M tokens)</span>
        <strong>${m['output_1m']:.4f}</strong>
      </div>
      <div class="stat-row">
        <span>Max Context Window</span>
        <strong>{m['context_length']:,} tokens</strong>
      </div>
      <div class="stat-row">
        <span>Recommended Self-Host GPU Tier</span>
        <strong>{html.escape(gpu['name'])} (${gpu['hourly_usd']}/hr)</strong>
      </div>
      <div class="stat-row">
        <span>API vs. GPU Daily Break-Even</span>
        <strong>~{breakeven_m_tokens_day}M tokens/day</strong>
      </div>

      <div style="margin-top:20px;display:flex;gap:12px;flex-wrap:wrap;">
        <a href="{html.escape(aff['url'])}" target="_blank" rel="nofollow sponsored" class="btn-cta">{html.escape(aff['cta_text'])}</a>
        <a href="{html.escape(alt_aff['url'])}" target="_blank" rel="nofollow sponsored" class="btn-cta btn-green">{html.escape(alt_aff['cta_text'])}</a>
      </div>
    </div>
  </div>

  <div class="affiliate-box">
    <div>
      <span class="badge badge-green">{html.escape(aff['badge'])}</span>
      <h3>Stop Overpaying for High-Volume Inference</h3>
      <p style="color:var(--muted);max-width:700px;">Running {html.escape(m['name'])} or open-weight alternatives at scale? Rent on-demand H100, A100, or RTX 4090 GPUs by the second with zero long-term commitment.</p>
    </div>
    <a href="{html.escape(aff['url'])}" target="_blank" rel="nofollow sponsored" class="btn-cta">{html.escape(aff['cta_text'])} →</a>
  </div>

  <div class="card" style="margin-top:24px;">
    <h2>Compare {html.escape(m['name'])} With Other Popular Models</h2>
    <ul style="margin-top:12px;padding-left:20px;display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:8px;">
      {related_links}
    </ul>
  </div>
</main>
<script>
  const IN_RATE = {m['input_1m']};
  const OUT_RATE = {m['output_1m']};
  const GPU_MONTHLY = {round(daily_gpu_cost * 30, 2)};
  function recalc() {{
    const inTok = parseFloat(document.getElementById('inSlider').value);
    const outTok = parseFloat(document.getElementById('outSlider').value);
    document.getElementById('inVal').textContent = inTok.toLocaleString();
    document.getElementById('outVal').textContent = outTok.toLocaleString();
    const daily = ((inTok / 1e6) * IN_RATE) + ((outTok / 1e6) * OUT_RATE);
    const monthly = daily * 30;
    document.getElementById('dailyCost').textContent = '$' + daily.toFixed(2);
    document.getElementById('monthlyCost').textContent = '$' + monthly.toFixed(2);
    const verdict = document.getElementById('verdictBox');
    if (monthly > GPU_MONTHLY) {{
      const save = (monthly - GPU_MONTHLY).toFixed(0);
      verdict.innerHTML = '💡 <strong>Self-Hosting Wins:</strong> At this volume, renting a dedicated GPU saves roughly <strong>$' + save + '/month</strong> compared to pay-per-token API billing.';
    }} else {{
      verdict.innerHTML = '✅ <strong>Serverless API / Serverless GPU Wins:</strong> At this volume, pay-per-token API or serverless GPU endpoints are cheaper than a 24/7 dedicated instance.';
    }}
  }}
  recalc();
</script>
"""
    page += shared_footer(config, updated_str, prefix="../")
    return page

def build_comparison_page(m1, m2, config, updated_str):
    base_url = config["base_url"].rstrip("/")
    comp_slug = f"{m1['slug']}-vs-{m2['slug']}"
    canonical = f"{base_url}/compare/{comp_slug}.html"
    title = f"{m1['name']} vs {m2['name']}: API Pricing & Token Cost Comparison (2026)"
    desc = f"Side-by-side price comparison: {m1['name']} (${m1['input_1m']}/1M in) vs {m2['name']} (${m2['input_1m']}/1M in). Calculate which LLM is cheaper for your workload."

    cheaper = m1 if m1["blended_1m"] <= m2["blended_1m"] else m2
    aff_runpod = config["affiliates"]["runpod"]
    aff_do = config["affiliates"]["vastai"]

    page = shared_head(title, desc, canonical)
    page += shared_nav(prefix="../")
    page += f"""
<main class="container">
  <section class="hero">
    <span class="badge">Head-to-Head Pricing Benchmark</span>
    <h1>{html.escape(m1['name'])} vs. {html.escape(m2['name'])}</h1>
    <p>Interactive cost comparison based on live API token rates and context window limits. Overall lower blended token cost: <strong>{html.escape(cheaper['name'])}</strong>.</p>
  </section>

  <div class="card">
    <h2>Side-by-Side Specs &amp; Token Rates</h2>
    <table>
      <thead>
        <tr>
          <th>Metric</th>
          <th><a href="../models/{m1['slug']}.html">{html.escape(m1['name'])}</a></th>
          <th><a href="../models/{m2['slug']}.html">{html.escape(m2['name'])}</a></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Provider</td>
          <td>{html.escape(m1['provider'])}</td>
          <td>{html.escape(m2['provider'])}</td>
        </tr>
        <tr>
          <td>Input Cost (per 1M tokens)</td>
          <td><strong>${m1['input_1m']:.4f}</strong></td>
          <td><strong>${m2['input_1m']:.4f}</strong></td>
        </tr>
        <tr>
          <td>Output Cost (per 1M tokens)</td>
          <td><strong>${m1['output_1m']:.4f}</strong></td>
          <td><strong>${m2['output_1m']:.4f}</strong></td>
        </tr>
        <tr>
          <td>Context Window</td>
          <td>{m1['context_length']:,} tokens</td>
          <td>{m2['context_length']:,} tokens</td>
        </tr>
        <tr>
          <td>Monthly Cost @ 10M In / 2M Out per day</td>
          <td><strong>${((m1['input_1m']*10 + m1['output_1m']*2)*30):,.2f}/mo</strong></td>
          <td><strong>${((m2['input_1m']*10 + m2['output_1m']*2)*30):,.2f}/mo</strong></td>
        </tr>
      </tbody>
    </table>
  </div>

  <div class="affiliate-box">
    <div>
      <span class="badge badge-green">Cloud Infrastructure Deal</span>
      <h3>Deploy Your AI App or Open-Source LLM for Less</h3>
      <p style="color:var(--muted);">Claim free cloud credits or rent serverless GPUs to cut your monthly AI infrastructure bill by up to 70%.</p>
    </div>
    <div style="display:flex;gap:12px;flex-wrap:wrap;">
      <a href="{html.escape(aff_runpod['url'])}" target="_blank" rel="nofollow sponsored" class="btn-cta">{html.escape(aff_runpod['cta_text'])}</a>
      <a href="{html.escape(aff_do['url'])}" target="_blank" rel="nofollow sponsored" class="btn-cta btn-green">{html.escape(aff_do['cta_text'])}</a>
    </div>
  </div>
</main>
"""
    page += shared_footer(config, updated_str, prefix="../")
    return comp_slug, page

def build_gpu_page(config, updated_str):
    base_url = config["base_url"].rstrip("/")
    canonical = f"{base_url}/gpu-cloud-pricing.html"
    title = "Cloud GPU Price Comparison (2026): H100, A100, RTX 4090 Hourly Rates"
    desc = "Compare hourly Cloud GPU rental prices across RunPod, Vast.ai, Vultr, DigitalOcean, and Hetzner. Find the cheapest GPU for LLM inference and fine-tuning."

    rows = ""
    for gpu in config["gpu_tiers"]:
        aff = config["affiliates"][gpu["provider_key"]]
        rows += f"""
        <tr>
          <td><strong>{html.escape(gpu['name'])}</strong></td>
          <td>{gpu['vram_gb']} GB</td>
          <td><strong style="color:var(--accent-green)">${gpu['hourly_usd']:.2f}/hr</strong> (${gpu['hourly_usd']*24*30:,.0f}/mo)</td>
          <td>~{gpu['throughput_tok_sec']} tok/s</td>
          <td>{html.escape(gpu['best_for'])}</td>
          <td><a href="{html.escape(aff['url'])}" target="_blank" rel="nofollow sponsored" class="btn-cta">{html.escape(aff['cta_text'])}</a></td>
        </tr>
        """

    cards = ""
    for key, aff in config["affiliates"].items():
        cards += f"""
        <div class="card">
          <span class="badge badge-green">{html.escape(aff['badge'])}</span>
          <h3>{html.escape(aff['name'])}</h3>
          <p style="color:var(--muted);font-size:0.92rem;margin:10px 0 18px;">High-performance cloud compute and GPU instances ready for AI agents, vLLM, Ollama, and production APIs.</p>
          <a href="{html.escape(aff['url'])}" target="_blank" rel="nofollow sponsored" class="btn-cta">{html.escape(aff['cta_text'])} →</a>
        </div>
        """

    page = shared_head(title, desc, canonical)
    page += shared_nav(prefix="")
    page += f"""
<main class="container">
  <section class="hero">
    <span class="badge">Live Cloud GPU &amp; VPS Benchmark</span>
    <h1>Cloud GPU Hourly Price Comparison (H100, A100, RTX 4090)</h1>
    <p>Compare on-demand GPU instances and claim free developer cloud credits for self-hosting LLMs, Stable Diffusion, and AI agents.</p>
  </section>

  <div class="card">
    <h2>On-Demand Cloud GPU Pricing Matrix</h2>
    <div style="overflow-x:auto;">
      <table>
        <thead>
          <tr>
            <th>GPU Instance</th>
            <th>VRAM</th>
            <th>On-Demand Rate</th>
            <th>Est. Speed</th>
            <th>Best Workload</th>
            <th>Deploy</th>
          </tr>
        </thead>
        <tbody>
          {rows}
        </tbody>
      </table>
    </div>
  </div>

  <h2 style="margin-top:36px;color:#fff;">Verified Cloud GPU &amp; VPS Free Credit Offers</h2>
  <div class="grid-3">
    {cards}
  </div>
</main>
"""
    page += shared_footer(config, updated_str, prefix="")
    return page

def build_index_page(models, comparisons, config, updated_str):
    base_url = config["base_url"].rstrip("/")
    canonical = f"{base_url}/index.html"
    title = f"{config['site_name']} — {config['site_tagline']}"
    desc = "Compare live API token pricing across 250+ AI models (OpenAI, Anthropic, DeepSeek, Meta Llama, Google Gemini) and calculate Cloud GPU self-hosting break-even costs."

    table_rows = ""
    for m in models[:120]:
        table_rows += f"""
        <tr class="model-row" data-name="{html.escape(m['name'].lower())} {html.escape(m['provider'].lower())}">
          <td><a href="models/{m['slug']}.html"><strong>{html.escape(m['name'])}</strong></a><br/><small style="color:var(--muted)">{html.escape(m['id'])}</small></td>
          <td>{html.escape(m['provider'])}</td>
          <td>${m['input_1m']:.4f}</td>
          <td>${m['output_1m']:.4f}</td>
          <td>{m['context_length']:,}</td>
          <td><a href="models/{m['slug']}.html">Calculator →</a></td>
        </tr>
        """

    comp_links = ""
    for c_slug, m1, m2 in comparisons[:60]:
        comp_links += f"""
        <a href="compare/{c_slug}.html" class="card" style="display:block;padding:14px 18px;text-decoration:none;">
          <strong style="color:#fff;">{html.escape(m1['name'])} vs {html.escape(m2['name'])}</strong>
          <div style="color:var(--muted);font-size:0.85rem;margin-top:4px;">${m1['input_1m']}/1M vs ${m2['input_1m']}/1M input tokens →</div>
        </a>
        """

    aff_main = config["affiliates"]["runpod"]
    aff_sec = config["affiliates"]["vultr"]

    page = shared_head(title, desc, canonical)
    page += shared_nav(prefix="")
    page += f"""
<main class="container">
  <section class="hero">
    <span class="badge">Auto-Updated Daily · {len(models)}+ Live AI Models Tracked</span>
    <h1>AI API Token Cost Calculator &amp; Cloud GPU Break-Even Engine</h1>
    <p>Instantly compare per-million token costs across OpenAI, Anthropic, Google, DeepSeek, Qwen, and Llama models—and see the exact point where renting a Cloud GPU saves you money.</p>
  </section>

  <div class="affiliate-box">
    <div>
      <span class="badge badge-green">Featured Infrastructure Deals</span>
      <h3>Building an AI Product or Running Local LLMs?</h3>
      <p style="color:var(--muted);">Deploy dedicated GPUs from $0.20/hr or claim up to $300 in free cloud compute credits below.</p>
    </div>
    <div style="display:flex;gap:12px;flex-wrap:wrap;">
      <a href="{html.escape(aff_main['url'])}" target="_blank" rel="nofollow sponsored" class="btn-cta">{html.escape(aff_main['cta_text'])}</a>
      <a href="{html.escape(aff_sec['url'])}" target="_blank" rel="nofollow sponsored" class="btn-cta btn-green">{html.escape(aff_sec['cta_text'])}</a>
    </div>
  </div>

  <div class="card">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;margin-bottom:14px;">
      <h2>Live AI Model Pricing Directory (Per 1M Tokens)</h2>
      <input type="text" id="searchInput" placeholder="Filter models (e.g. gpt-4, claude, deepseek, llama)..." style="max-width:360px;" oninput="filterTable()" />
    </div>
    <div style="overflow-x:auto;">
      <table>
        <thead>
          <tr>
            <th>Model Name</th>
            <th>Provider</th>
            <th>Input / 1M</th>
            <th>Output / 1M</th>
            <th>Context Window</th>
            <th>Cost Calculator</th>
          </tr>
        </thead>
        <tbody id="modelTableBody">
          {table_rows}
        </tbody>
      </table>
    </div>
  </div>

  <section id="comparisons" style="margin-top:42px;">
    <h2 style="color:#fff;margin-bottom:10px;">Popular Head-to-Head Model Cost Comparisons</h2>
    <p style="color:var(--muted);margin-bottom:18px;">Click any pair below to view side-by-side token calculators and monthly cost breakdowns.</p>
    <div class="grid-3">
      {comp_links}
    </div>
  </section>
</main>
<script>
  function filterTable() {{
    const q = document.getElementById('searchInput').value.toLowerCase().trim();
    const rows = document.querySelectorAll('.model-row');
    rows.forEach(r => {{
      const name = r.getAttribute('data-name') || '';
      r.style.display = name.includes(q) ? '' : 'none';
    }});
  }}
</script>
"""
    page += shared_footer(config, updated_str, prefix="")
    return page

def main():
    config = load_config()
    updated_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print("Fetching live AI model pricing data...")
    all_models = fetch_models()
    max_models = int(config.get("max_individual_model_pages", 250))
    models = all_models[:max_models]
    print(f"Selected {len(models)} models for programmatic calculator pages.")

    os.makedirs(os.path.join(DIST_DIR, "models"), exist_ok=True)
    os.makedirs(os.path.join(DIST_DIR, "compare"), exist_ok=True)

    urls = []
    base_url = config["base_url"].rstrip("/")

    # 1. Build individual model calculator pages
    for m in models:
        html_out = build_model_page(m, models, config, updated_str)
        out_path = os.path.join(DIST_DIR, "models", f"{m['slug']}.html")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html_out)
        urls.append(f"{base_url}/models/{m['slug']}.html")

    # 2. Build comparison pages for top models
    max_comp = int(config.get("max_comparison_pages", 150))
    top_for_comp = models[:25]
    comparisons = []
    for i in range(len(top_for_comp)):
        for j in range(i + 1, len(top_for_comp)):
            if len(comparisons) >= max_comp:
                break
            m1, m2 = top_for_comp[i], top_for_comp[j]
            c_slug, c_html = build_comparison_page(m1, m2, config, updated_str)
            c_path = os.path.join(DIST_DIR, "compare", f"{c_slug}.html")
            with open(c_path, "w", encoding="utf-8") as f:
                f.write(c_html)
            comparisons.append((c_slug, m1, m2))
            urls.append(f"{base_url}/compare/{c_slug}.html")
        if len(comparisons) >= max_comp:
            break

    # 3. Build Cloud GPU comparison page
    gpu_html = build_gpu_page(config, updated_str)
    with open(os.path.join(DIST_DIR, "gpu-cloud-pricing.html"), "w", encoding="utf-8") as f:
        f.write(gpu_html)
    urls.insert(0, f"{base_url}/gpu-cloud-pricing.html")

    # 4. Build Main Index Hub
    index_html = build_index_page(models, comparisons, config, updated_str)
    with open(os.path.join(DIST_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(index_html)
    urls.insert(0, f"{base_url}/index.html")

    # 5. Build sitemap.xml & robots.txt & .nojekyll
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    sitemap_entries = "\n".join(
        f"  <url><loc>{html.escape(u)}</loc><lastmod>{today}</lastmod><changefreq>daily</changefreq></url>"
        for u in urls
    )
    sitemap_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
{sitemap_entries}
</urlset>
"""
    with open(os.path.join(DIST_DIR, "sitemap.xml"), "w", encoding="utf-8") as f:
        f.write(sitemap_xml)

    robots_txt = f"User-agent: *\nAllow: /\nSitemap: {base_url}/sitemap.xml\n"
    with open(os.path.join(DIST_DIR, "robots.txt"), "w", encoding="utf-8") as f:
        f.write(robots_txt)

    with open(os.path.join(DIST_DIR, ".nojekyll"), "w", encoding="utf-8") as f:
        f.write("")

    print(f"Successfully generated {len(urls)} total SEO pages in {DIST_DIR}")

if __name__ == "__main__":
    main()
