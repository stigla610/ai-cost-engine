import urllib.request
import json
import os
import re
import sys
import html
import shutil
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
DIST_DIR = os.path.join(BASE_DIR, "dist")

PROVIDER_NAMES = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "google": "Google",
    "deepseek": "DeepSeek",
    "meta-llama": "Meta Llama",
    "meta": "Meta",
    "qwen": "Qwen",
    "mistralai": "Mistral AI",
    "x-ai": "xAI",
    "cohere": "Cohere",
    "microsoft": "Microsoft",
    "nvidia": "NVIDIA",
    "amazon": "Amazon",
    "perplexity": "Perplexity",
    "minimax": "MiniMax",
    "moonshotai": "Moonshot AI",
    "z-ai": "Z.AI (GLM)",
    "thudm": "Zhipu GLM",
    "01-ai": "01.AI",
    "nousresearch": "Nous Research",
    "allenai": "AllenAI",
    "ibm-granite": "IBM Granite",
    "baidu": "Baidu",
    "bytedance": "ByteDance",
    "bytedance-seed": "ByteDance Seed",
    "tencent": "Tencent",
    "xiaomi": "Xiaomi",
    "liquid": "Liquid AI",
    "cognitivecomputations": "Cognitive Computations",
    "sao10k": "Sao10K",
    "thedrummer": "TheDrummer",
    "gryphe": "Gryphe",
    "aion-labs": "Aion Labs",
    "inflection": "Inflection",
    "ai21": "AI21 Labs",
}

def slugify(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def format_price(val):
    if round(val, 2) == round(val, 4):
        return f"${val:,.2f}"
    if round(val, 3) == round(val, 4):
        return f"${val:,.3f}"
    return f"${val:,.4f}"

def clean_provider(model_id):
    if "/" not in model_id:
        return "AI"
    raw = model_id.split("/")[0].lstrip("~").strip().lower()
    if raw in PROVIDER_NAMES:
        return PROVIDER_NAMES[raw]
    return raw.replace("-", " ").title()

def clean_model_name(name, model_id):
    cleaned = name.strip().lstrip("~").strip()
    # Fix awkward double spaces
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned

def is_open_weight(model_id, name):
    mid = model_id.lower().lstrip("~")
    combined = f"{mid} {name.lower()}"

    # Batch, pure image, audio, or realtime endpoints are API-specific
    if any(k in mid for k in [":batch", "-image", "-audio", "-realtime"]):
        return False

    # OpenAI gpt-oss models are explicitly open-weight
    if "gpt-oss" in mid:
        return True

    # Closed proprietary API providers
    if mid.startswith(("openai/", "anthropic/", "x-ai/", "amazon/", "perplexity/", "ai21/", "inflection/", "openrouter/")):
        return False

    # Google: Gemma is open-weight, Gemini is closed proprietary API
    if mid.startswith("google/"):
        return "gemma" in mid

    # Meta: meta-llama is open-weight; meta/muse-spark-open or glimmer-30b are open
    if mid.startswith("meta/"):
        return "open" in mid or "30b" in mid or "llama" in mid
    if mid.startswith("meta-llama/"):
        return True

    # Qwen: models with explicit parameter sizes or open series (qwq, coder, vl, qwen-2.5, qwen3-14b/32b/etc.) are open-weight;
    # commercial API tiers (max, plus, turbo, flash without parameter counts) are hosted APIs
    if mid.startswith("qwen/"):
        if re.search(r"\b\d+(?:\.\d+)?[bt]\b", combined) or "qwq" in mid or "qwen-2.5" in mid or "qwen2.5" in mid:
            return True
        if any(k in mid for k in ["max", "plus", "turbo", "flash"]):
            return False
        return True

    # Mistral: mistral-medium is closed API; Mixtral, Mistral Small/Large/Nemo/Saba, Ministral, Codestral, Devstral, Voxtral, Pixtral are open-weight
    if mid.startswith("mistralai/"):
        if "mistral-medium" in mid:
            return False
        return True

    # DeepSeek: V3, R1, Chat, Reasoner, Coder, Prover are open-weight; synthetic flash-latest/pro-latest aliases are API tiers
    if mid.startswith("deepseek/"):
        if model_id.startswith("~") or "flash" in mid or "v4-pro" in mid or "pro-latest" in mid:
            return False
        return True

    # Other open-weight providers (including Kimi K2, Z.AI GLM, Cohere Command R/A, Microsoft Phi, NVIDIA Nemotron)
    if mid.startswith((
        "microsoft/phi", "microsoft/wizard", "nvidia/", "nousresearch/",
        "cognitivecomputations/", "sao10k/", "thedrummer/", "gryphe/",
        "allenai/", "ibm-granite/", "liquid/", "01-ai/", "moonshotai/kimi",
        "z-ai/glm", "thudm/glm", "cohere/command-r", "cohere/command-a"
    )):
        return True

    # If model explicitly states a parameter count in billions (e.g. 7b, 8b, 14b, 27b, 30b, 32b, 70b, 72b, 120b, 235b, 405b), it is typically open-weight
    if re.search(r"\b\d+(?:\.\d+)?b\b", combined):
        return True

    return False

def estimate_params_b(model_id, name):
    combined = f"{model_id} {name}".lower()

    # 1. Check MoE NxMB pattern first (e.g. 8x7b -> 47B, 8x22b -> 141B)
    m_moe = re.search(r"\b(\d+)x(\d+(?:\.\d+)?)b\b", combined)
    if m_moe:
        n_exp = float(m_moe.group(1))
        b_exp = float(m_moe.group(2))
        return round(n_exp * b_exp * 0.8, 1)

    # 2. Check Trillion parameter pattern (e.g. 2.4t -> 2400B)
    m_tril = re.search(r"\b(\d+(?:\.\d+)?)t\b", combined)
    if m_tril:
        val_t = float(m_tril.group(1))
        if 0.5 <= val_t <= 10:
            return round(val_t * 1000.0, 1)

    # 3. Check total Billion parameter pattern (ignoring active parameter tags like a3b, a17b, a22b)
    m_bil = re.search(r"(?<![a-z0-9])(\d+(?:\.\d+)?)b\b", combined)
    if m_bil:
        val = float(m_bil.group(1))
        if 0.5 <= val <= 2000:
            return val

    # 4. Known specific model families without NNb in their display name
    if "deepseek" in combined:
        if re.search(r"\b(flash|lite|mini|nano|small)\b", combined):
            return 14.0
        if any(k in combined for k in ["v3", "r1", "chat", "reasoner", "prover", "v4"]):
            return 671.0

    if "phi-4" in combined or "phi-3" in combined:
        if re.search(r"\b(mini|small|multimodal)\b", combined):
            return 3.8
        return 14.0
    if "llama-4-maverick" in combined:
        return 400.0
    if "llama-4-scout" in combined:
        return 109.0
    if "mistral-large" in combined or "pixtral-large" in combined:
        return 123.0
    if "mistral-medium" in combined:
        return 70.0
    if any(k in combined for k in ["mistral-small", "devstral", "voxtral-small"]):
        return 24.0
    if "codestral" in combined:
        return 22.0
    if "mistral-nemo" in combined or "mistral-saba" in combined:
        return 12.0
    if "command-a" in combined:
        return 111.0
    if "command-r-plus" in combined:
        return 104.0
    if "command-r" in combined:
        return 35.0
    if "kimi-k2" in combined:
        return 1000.0
    if "glm-4.5-air" in combined:
        return 106.0
    if "glm-4.5" in combined or "glm-5" in combined:
        return 355.0

    # 5. Check Nano / Lite / Micro / Mini / Flash / Haiku / Small with word boundaries BEFORE generic provider names
    if re.search(r"\b(nano|micro|lite)\b", combined):
        return 4.0
    if re.search(r"\b(mini|flash|haiku|small)\b", combined):
        return 8.0

    # 6. Flagship / Large / Pro tiers (for open-weight equivalent comparison)
    if re.search(r"\b(opus|ultra|pro|max|large|gpt-5|gpt-4|claude|o1|o3|grok)\b", combined):
        return 70.0

    return 32.0

def fetch_models():
    url = "https://openrouter.ai/api/v1/models"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "TokenBench-AutoBuilder/2.1"}
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    models = []
    seen_slugs = set()
    for item in raw.get("data", []):
        model_id = item.get("id", "")
        raw_name = item.get("name", model_id)
        name = clean_model_name(raw_name, model_id)
        pricing = item.get("pricing") or {}
        try:
            prompt_per_tok = float(pricing.get("prompt", 0) or 0)
            completion_per_tok = float(pricing.get("completion", 0) or 0)
        except (ValueError, TypeError):
            continue

        input_1m = round(prompt_per_tok * 1_000_000, 4)
        output_1m = round(completion_per_tok * 1_000_000, 4)
        if input_1m < 0 or output_1m < 0:
            continue

        context_length = int(item.get("context_length") or 4096)
        slug = slugify(model_id)
        if not slug or slug in seen_slugs:
            continue
        seen_slugs.add(slug)

        provider = clean_provider(model_id)
        raw_prov = model_id.split("/")[0].lstrip("~").lower() if "/" in model_id else "other"
        params_b = estimate_params_b(model_id, name)
        open_weight = is_open_weight(model_id, name)
        is_secondary = any(k in model_id.lower() for k in [":batch", ":free", "-image", "-audio", "-realtime", "guard", "safeguard"]) or model_id.startswith("~")

        if params_b >= 100:
            gpu_tier_idx = 3
        elif params_b >= 65:
            gpu_tier_idx = 2
        elif params_b >= 25:
            gpu_tier_idx = 1
        else:
            gpu_tier_idx = 0

        models.append({
            "id": model_id,
            "name": name,
            "slug": slug,
            "provider": provider,
            "raw_provider": raw_prov,
            "is_open_weight": open_weight,
            "is_secondary": is_secondary,
            "params_b": params_b,
            "vram_fp16": round(params_b * 2.2, 1),
            "vram_int8": round(params_b * 1.15, 1),
            "vram_int4": round(params_b * 0.65, 1),
            "input_1m": input_1m,
            "output_1m": output_1m,
            "blended_1m": round((input_1m * 0.7) + (output_1m * 0.3), 4),
            "context_length": context_length,
            "description": (item.get("description") or f"Live API token pricing, VRAM requirements, and Cloud GPU break-even analysis for {name}.")[:280],
            "gpu_tier_idx": gpu_tier_idx,
        })

    paid_models = [m for m in models if m["input_1m"] > 0 or m["output_1m"] > 0]

    # Interleave across major families so the top models represent OpenAI, Anthropic, Google, DeepSeek, Llama, Qwen, Mistral, xAI, etc.
    priority_families = [
        "openai", "anthropic", "google", "deepseek",
        "meta-llama", "qwen", "mistralai", "x-ai",
        "nvidia", "cohere", "z-ai", "moonshotai", "meta", "microsoft"
    ]

    flagship_exact = {
        "openai/gpt-5": 0, "openai/gpt-4o": 1, "openai/gpt-4.1": 2, "openai/gpt-4o-mini": 3,
        "openai/o3": 4, "openai/gpt-oss-120b": 5, "openai/gpt-5-mini": 6, "openai/gpt-4": 7,
        "anthropic/claude-sonnet-4.5": 0, "anthropic/claude-opus-4.5": 1, "anthropic/claude-haiku-4.5": 2,
        "anthropic/claude-sonnet-4.6": 3, "anthropic/claude-opus-4.6": 4, "anthropic/claude-sonnet-5": 5,
        "google/gemini-2.5-pro": 0, "google/gemini-2.5-flash": 1, "google/gemma-4-31b-it": 2,
        "google/gemma-3-27b-it": 3, "google/gemma-3-12b-it": 4,
        "deepseek/deepseek-chat": 0, "deepseek/deepseek-r1-0528": 1, "deepseek/deepseek-v3.2": 2,
        "deepseek/deepseek-chat-v3.1": 3,
        "meta-llama/llama-4-maverick": 0, "meta-llama/llama-4-scout": 1,
        "meta-llama/llama-3.3-70b-instruct": 2, "meta-llama/llama-3.1-70b-instruct": 3,
        "meta-llama/llama-3.1-8b-instruct": 4,
        "qwen/qwen3-32b": 0, "qwen/qwen-2.5-72b-instruct": 1, "qwen/qwen3-235b-a22b-2507": 2,
        "qwen/qwen-2.5-coder-32b-instruct": 3, "qwen/qwen3-14b": 4,
        "mistralai/mistral-large": 0, "mistralai/mistral-small-3.2-24b-instruct": 1,
        "mistralai/codestral-2508": 2, "mistralai/mixtral-8x22b-instruct": 3,
    }

    def within_family_key(m):
        mid = m["id"].lower()
        dated = bool(re.search(r"-\d{4}-\d{2}-\d{2}", mid))
        old_gen = "gpt-3.5" in mid or "0613" in mid
        flagship_rank = flagship_exact.get(mid, 100)
        return (
            1 if m["is_secondary"] else 0,
            flagship_rank,
            1 if old_gen else 0,
            1 if dated else 0,
            m["name"]
        )

    buckets = {fam: [] for fam in priority_families}
    other_bucket = []
    for m in paid_models:
        rp = m["raw_provider"]
        if rp in buckets:
            buckets[rp].append(m)
        else:
            other_bucket.append(m)

    for fam in buckets:
        buckets[fam].sort(key=within_family_key)
    other_bucket.sort(key=within_family_key)

    # Separate primary vs secondary (:batch / ~ alias) so primary models always come first
    interleaved_primary = []
    interleaved_secondary = []

    primary_buckets = [[m for m in buckets[f] if not m["is_secondary"]] for f in priority_families]
    primary_buckets.append([m for m in other_bucket if not m["is_secondary"]])

    max_len = max((len(b) for b in primary_buckets), default=0)
    for idx in range(max_len):
        for b in primary_buckets:
            if idx < len(b):
                interleaved_primary.append(b[idx])

    for f in priority_families:
        interleaved_secondary.extend([m for m in buckets[f] if m["is_secondary"]])
    interleaved_secondary.extend([m for m in other_bucket if m["is_secondary"]])

    return interleaved_primary + interleaved_secondary

def shared_head(title, description, canonical_url, config, json_ld=None):
    schema_tag = ""
    if json_ld:
        schema_tag = f'<script type="application/ld+json">{json.dumps(json_ld)}</script>'
    gsc = config.get("google_site_verification", "").strip()
    gsc_tag = f'<meta name="google-site-verification" content="{html.escape(gsc)}" />' if gsc else ""
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
  {gsc_tag}
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
    html, body {{
      max-width: 100%;
      overflow-x: hidden;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
    }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .container {{ max-width: 1180px; margin: 0 auto; padding: 0 20px; width: 100%; }}
    header {{
      border-bottom: 1px solid var(--border);
      background: rgba(11, 15, 25, 0.92);
      position: sticky;
      top: 0;
      z-index: 50;
      backdrop-filter: blur(8px);
    }}
    .nav {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 14px 0;
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
    .nav-links {{ display: flex; gap: 16px; font-size: 0.92rem; align-items: center; flex-wrap: wrap; }}
    .btn-cta {{
      background: linear-gradient(135deg, #0284c7, #4f46e5);
      color: #fff !important;
      padding: 9px 16px;
      border-radius: 8px;
      font-weight: 600;
      display: inline-block;
      transition: transform 0.15s ease, opacity 0.15s ease;
      text-decoration: none !important;
      white-space: nowrap;
    }}
    .btn-cta:hover {{ transform: translateY(-1px); opacity: 0.95; }}
    .btn-green {{
      background: linear-gradient(135deg, #059669, #10b981);
    }}
    .hero {{ padding: 38px 0 24px; }}
    .hero h1 {{ font-size: 2.15rem; line-height: 1.25; margin-bottom: 12px; color: #fff; word-break: break-word; }}
    .hero p {{ color: var(--muted); font-size: 1.05rem; max-width: 840px; }}
    .grid-2 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 340px), 1fr)); gap: 20px; margin: 24px 0; }}
    .grid-3 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 260px), 1fr)); gap: 18px; margin: 24px 0; }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 22px;
      min-width: 0;
    }}
    .card h2, .card h3 {{ color: #fff; margin-bottom: 10px; }}
    .badge {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 0.76rem;
      font-weight: 600;
      background: rgba(56, 189, 248, 0.15);
      color: var(--accent);
      margin-bottom: 10px;
    }}
    .badge-green {{
      background: rgba(16, 185, 129, 0.15);
      color: var(--accent-green);
    }}
    .badge-warn {{
      background: rgba(245, 158, 11, 0.15);
      color: var(--warn);
    }}
    .stat-row {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 10px 0;
      border-bottom: 1px solid var(--border);
      font-size: 0.94rem;
      flex-wrap: wrap;
    }}
    .stat-row:last-child {{ border-bottom: none; }}
    .price-big {{ font-size: 1.75rem; font-weight: 800; color: var(--accent-green); }}
    .slider-group {{ margin-bottom: 16px; }}
    .slider-group label {{ display: flex; justify-content: space-between; font-weight: 600; margin-bottom: 6px; font-size: 0.92rem; gap: 8px; flex-wrap: wrap; }}
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
    .table-wrap {{
      width: 100%;
      overflow-x: auto;
      -webkit-overflow-scrolling: touch;
      margin-top: 12px;
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.91rem; min-width: 640px; }}
    th, td {{ padding: 12px 10px; text-align: left; border-bottom: 1px solid var(--border); }}
    th {{ color: var(--muted); font-weight: 600; background: rgba(255,255,255,0.02); white-space: nowrap; }}
    tr:hover td {{ background: rgba(255,255,255,0.02); }}
    .filter-bar {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 14px;
    }}
    .filter-pill {{
      background: #0b0f19;
      border: 1px solid var(--border);
      color: var(--muted);
      padding: 6px 12px;
      border-radius: 999px;
      font-size: 0.82rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s ease;
    }}
    .filter-pill:hover, .filter-pill.active {{
      border-color: var(--accent);
      color: #fff;
      background: rgba(56, 189, 248, 0.15);
    }}
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
    .notice-box {{
      background: rgba(245, 158, 11, 0.1);
      border: 1px solid rgba(245, 158, 11, 0.4);
      border-radius: 10px;
      padding: 14px 18px;
      margin-bottom: 20px;
      font-size: 0.92rem;
      color: #fde68a;
    }}
    footer {{
      margin-top: 60px;
      border-top: 1px solid var(--border);
      padding: 32px 0;
      color: var(--muted);
      font-size: 0.88rem;
    }}
    @media (max-width: 768px) {{
      .container {{ padding: 0 14px; }}
      .hero {{ padding: 24px 0 16px; }}
      .hero h1 {{ font-size: 1.55rem; }}
      .hero p {{ font-size: 0.95rem; }}
      .nav {{ flex-direction: column; align-items: flex-start; gap: 10px; }}
      .nav-links {{ width: 100%; gap: 8px 12px; font-size: 0.84rem; }}
      .btn-cta {{ padding: 8px 12px; font-size: 0.84rem; white-space: normal; text-align: center; }}
      .card {{ padding: 16px; }}
      .grid-2, .grid-3 {{ grid-template-columns: 1fr; gap: 14px; }}
      table {{ min-width: 540px; font-size: 0.84rem; }}
      th, td {{ padding: 9px 8px; }}
    }}
  </style>
</head>
"""

def shared_nav(prefix="", config=None):
    runpod_url = config["affiliates"]["runpod"]["url"] if config else "#"
    return f"""<body>
<header>
  <div class="container nav">
    <a href="{prefix}index.html" class="brand">⚡ Token<span>Bench</span></a>
    <nav class="nav-links">
      <a href="{prefix}index.html">API Cost Calculator</a>
      <a href="{prefix}vram-electricity-calculator.html">VRAM &amp; Electricity Calculator</a>
      <a href="{prefix}gpu-cloud-pricing.html">Cloud GPU Rates</a>
      <a href="{prefix}index.html#comparisons">Model Comparisons</a>
      <a href="{html.escape(runpod_url)}" target="_blank" rel="nofollow sponsored" class="btn-cta">Rent Cloud GPUs (From $0.20/hr)</a>
    </nav>
  </div>
</header>
"""

def shared_footer(config, updated_str, prefix=""):
    return f"""<footer>
  <div class="container">
    <p><strong>{html.escape(config['site_name'])}</strong> — Automatically updated daily with live API pricing, VRAM calculators &amp; cloud GPU benchmarks. Last auto-sync: {updated_str}.</p>
    <p style="margin-top:8px;">Links to cloud GPU &amp; infrastructure providers include partner referral links at zero extra cost to you, keeping our calculators 100% free.</p>
    <p style="margin-top:8px;">
      <a href="{prefix}index.html">All AI Models</a> ·
      <a href="{prefix}vram-electricity-calculator.html">VRAM &amp; Local GPU Electricity Calculator</a> ·
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

    daily_gpu_cost = round(gpu["hourly_usd"] * 24, 2)
    blended = max(m["blended_1m"], 0.05)
    breakeven_m_tokens_day = round(daily_gpu_cost / blended, 1)

    in_fmt = format_price(m["input_1m"])
    out_fmt = format_price(m["output_1m"])

    if m["is_open_weight"]:
        title = f"{m['name']} API Cost Calculator & Self-Host GPU Break-Even (2026)"
        desc = f"Calculate exact {m['name']} token costs ({in_fmt}/1M input, {out_fmt}/1M output), local VRAM requirements ({m['vram_int4']}GB Q4), and Cloud GPU self-hosting break-even."
        faq_q2 = f"When is it cheaper to self-host {m['name']} on a Cloud GPU?"
        faq_a2 = f"Renting a dedicated {gpu['name']} costs roughly ${gpu['hourly_usd']}/hour (${daily_gpu_cost}/day). If your workload exceeds {breakeven_m_tokens_day}M blended tokens per day, self-hosting {m['name']} on {aff['name']} can significantly reduce your monthly inference bill."
        weights_badge = '<span class="badge badge-green">Open-Weight / Self-Hostable</span>'
        vram_label = "VRAM Needed (4-Bit Q4 / FP16)"
        vram_link_text = f"{m['vram_int4']} GB / {m['vram_fp16']} GB →"
        gpu_row_label = f"24/7 Dedicated {html.escape(gpu['name'])} Self-Host Cost"
        verdict_win_prefix = f"Self-Hosting {html.escape(m['name'])} Wins"
    else:
        title = f"{m['name']} API Cost Calculator & Open-Weight GPU Break-Even (2026)"
        desc = f"Calculate exact {m['name']} API token costs ({in_fmt}/1M input, {out_fmt}/1M output) and compare against self-hosting an open-weight GPU alternative."
        faq_q2 = f"Can you self-host {m['name']} or switch to a Cloud GPU?"
        faq_a2 = f"{m['name']} is a proprietary API model, so its weights are not downloadable. However, if your API spend exceeds ${round(daily_gpu_cost * 30):,}/month (~{breakeven_m_tokens_day}M tokens/day), running an equivalent open-weight model (Llama, Qwen, or DeepSeek) on a dedicated {gpu['name']} (${gpu['hourly_usd']}/hr) on {aff['name']} can cut costs substantially."
        weights_badge = '<span class="badge badge-warn">Proprietary Cloud API</span>'
        vram_label = f"Open-Weight Eq. VRAM (~{m['params_b']:g}B Class)"
        vram_link_text = f"~{m['vram_int4']} GB Q4 Equivalent Guide →"
        gpu_row_label = f"24/7 {html.escape(gpu['name'])} (Open-Weight Alternative)"
        verdict_win_prefix = "Switching to Open-Weight GPU Wins"

    json_ld = {
        "@context": "https://schema.org",
        "@type": "FAQPage",
        "mainEntity": [
            {
                "@type": "Question",
                "name": f"How much does the {m['name']} API cost per 1M tokens?",
                "acceptedAnswer": {
                    "@type": "Answer",
                    "text": f"{m['name']} costs {in_fmt} per 1M input tokens and {out_fmt} per 1M output tokens, with a context window of {m['context_length']:,} tokens."
                }
            },
            {
                "@type": "Question",
                "name": faq_q2,
                "acceptedAnswer": {
                    "@type": "Answer",
                    "text": faq_a2
                }
            }
        ]
    }

    related = [x for x in all_models if x["slug"] != m["slug"] and not x["is_secondary"]][:8]
    related_links = "".join(
        f'<li><a href="{x["slug"]}.html">{html.escape(x["name"])} Cost Calculator ({format_price(x["input_1m"])}/1M in)</a></li>'
        for x in related
    )

    page = shared_head(title, desc, canonical, config, json_ld)
    page += shared_nav(prefix="../", config=config)
    page += f"""
<main class="container">
  <section class="hero">
    <div style="display:flex;gap:8px;flex-wrap:wrap;">
      <span class="badge">{html.escape(m['provider'])} Model Pricing &amp; Calculator</span>
      {weights_badge}
    </div>
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
        <span>{gpu_row_label}</span>
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
        <span>Provider</span>
        <strong>{html.escape(m['provider'])}</strong>
      </div>
      <div class="stat-row">
        <span>Weight Availability</span>
        <strong>{"Open-Weight (Downloadable GGUF / Safetensors)" if m["is_open_weight"] else "Proprietary Cloud API (Closed Weights)"}</strong>
      </div>
      <div class="stat-row">
        <span>Input Price (per 1M tokens)</span>
        <strong>{in_fmt}</strong>
      </div>
      <div class="stat-row">
        <span>Output Price (per 1M tokens)</span>
        <strong>{out_fmt}</strong>
      </div>
      <div class="stat-row">
        <span>Max Context Window</span>
        <strong>{m['context_length']:,} tokens</strong>
      </div>
      <div class="stat-row">
        <span>{vram_label}</span>
        <strong><a href="../vram/{m['slug']}-vram-requirements.html">{vram_link_text}</a></strong>
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
      <h3>Stop Overpaying for High-Volume AI Inference</h3>
      <p style="color:var(--muted);max-width:700px;">{"Running " + html.escape(m['name']) + " at scale? Deploy pre-configured vLLM or Ollama pods on dedicated H100, A100, or RTX 4090 GPUs by the second." if m["is_open_weight"] else "Spending heavily on " + html.escape(m['name']) + " API calls? Benchmark open-weight alternatives (Llama 4, Qwen 3, DeepSeek V3/R1) on dedicated H100, A100, or RTX 4090 GPUs from $0.20/hr."}</p>
    </div>
    <a href="{html.escape(aff['url'])}" target="_blank" rel="nofollow sponsored" class="btn-cta">{html.escape(aff['cta_text'])} →</a>
  </div>

  <div class="card" style="margin-top:24px;">
    <h2>Compare {html.escape(m['name'])} With Other Popular Models</h2>
    <ul style="margin-top:12px;padding-left:20px;display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%, 260px),1fr));gap:8px;">
      {related_links}
    </ul>
  </div>
</main>
<script>
  const IN_RATE = {m['input_1m']};
  const OUT_RATE = {m['output_1m']};
  const GPU_MONTHLY = {round(daily_gpu_cost * 30, 2)};
  const IS_OPEN = {"true" if m["is_open_weight"] else "false"};
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
      if (IS_OPEN) {{
        verdict.innerHTML = '💡 <strong>{verdict_win_prefix}:</strong> At this volume, renting a dedicated Cloud GPU saves roughly <strong>$' + save + '/month</strong> compared to pay-per-token API billing.';
      }} else {{
        verdict.innerHTML = '💡 <strong>{verdict_win_prefix}:</strong> Since this model is a closed API, migrating high-volume tasks to a self-hosted open-weight equivalent (Llama / Qwen / DeepSeek) on a dedicated GPU can save roughly <strong>$' + save + '/month</strong>.';
      }}
    }} else {{
      verdict.innerHTML = '✅ <strong>Pay-Per-Token API Wins:</strong> At this volume, staying on the serverless API (or serverless GPU endpoints) is cheaper than renting a 24/7 dedicated GPU instance.';
    }}
  }}
  recalc();
</script>
"""
    page += shared_footer(config, updated_str, prefix="../")
    return page

def build_vram_page(m, config, updated_str):
    base_url = config["base_url"].rstrip("/")
    vram_slug = f"{m['slug']}-vram-requirements"
    canonical = f"{base_url}/vram/{vram_slug}.html"
    gpu = config["gpu_tiers"][m["gpu_tier_idx"]]
    aff_runpod = config["affiliates"]["runpod"]
    aff_vast = config["affiliates"]["vastai"]

    in_fmt = format_price(m["input_1m"])
    out_fmt = format_price(m["output_1m"])
    slider_max = max(192, int((m["vram_fp16"] // 32 + 2) * 32))

    if m["is_open_weight"]:
        title = f"{m['name']} VRAM Requirements & GPU Hardware Calculator (FP16, Q8, Q4)"
        desc = f"How much VRAM do you need to run {m['name']} locally? See exact memory requirements for FP16 ({m['vram_fp16']} GB), 8-bit ({m['vram_int8']} GB), and 4-bit GGUF/AWQ ({m['vram_int4']} GB) plus Cloud GPU rental costs."
        hero_h1 = f"{html.escape(m['name'])} VRAM Requirements &amp; GPU Guide"
        hero_p = f"Find out if your local GPU can run <strong>{html.escape(m['name'])}</strong> (~{m['params_b']:g}B parameters) in FP16, 8-bit, or 4-bit GGUF/EXL2/AWQ quantization—or rent the exact Cloud GPU needed from ${gpu['hourly_usd']}/hr."
        notice_html = ""
        card1_title = f"VRAM Required by Quantization Level (~{m['params_b']:g}B Params)"
        card2_title = f"Can My Local GPU Run {html.escape(m['name'])}?"
        faq_a = f"Running {m['name']} (~{m['params_b']:g}B parameters) requires approximately {m['vram_fp16']} GB VRAM in 16-bit (FP16/BF16), {m['vram_int8']} GB VRAM in 8-bit (Q8_0) quantization, and {m['vram_int4']} GB VRAM in 4-bit (Q4_K_M / AWQ) quantization including KV cache overhead."
    else:
        title = f"Can You Run {m['name']} Locally? VRAM Equivalent & Open-Weight Alternatives"
        desc = f"{m['name']} is a proprietary Cloud API. See the exact GPU VRAM required ({m['vram_int4']} GB Q4 / {m['vram_fp16']} GB FP16) to run an equivalent ~{m['params_b']:g}B open-weight alternative locally or on RunPod/Vast.ai."
        hero_h1 = f"{html.escape(m['name'])}: Local VRAM Equivalent &amp; Self-Hosting Guide"
        hero_p = f"<strong>{html.escape(m['name'])}</strong> is a closed-weight commercial API hosted by {html.escape(m['provider'])}. Below are the exact VRAM specifications needed to self-host a comparable <strong>~{m['params_b']:g}B-class open-weight LLM</strong> (such as Meta Llama, Qwen, Mistral, or DeepSeek) in FP16, Q8, or Q4 quantization."
        notice_html = f"""
  <div class="notice-box">
    <strong>ℹ️ Closed-Weight API Notice:</strong> {html.escape(m['name'])} weights are proprietary to {html.escape(m['provider'])} and cannot be downloaded as local GGUF/Safetensors files. The calculator below shows the hardware and VRAM required to run an <strong>equivalent ~{m['params_b']:g}B parameter open-weight alternative</strong> locally or on a dedicated Cloud GPU.
  </div>"""
        card1_title = f"Equivalent Open-Weight VRAM (~{m['params_b']:g}B Class Model)"
        card2_title = f"Can My GPU Run a ~{m['params_b']:g}B Open-Weight Alternative?"
        faq_a = f"{m['name']} is a closed-source API and cannot be downloaded directly. To run a comparable ~{m['params_b']:g}B parameter open-weight model locally, you need roughly {m['vram_fp16']} GB VRAM in FP16, {m['vram_int8']} GB VRAM in 8-bit (Q8_0), or {m['vram_int4']} GB VRAM in 4-bit (Q4_K_M / AWQ) quantization."

    json_ld = {
        "@context": "https://schema.org",
        "@type": "FAQPage",
        "mainEntity": [
            {
                "@type": "Question",
                "name": f"How much VRAM is needed to run {m['name']} or an open-weight equivalent?",
                "acceptedAnswer": {
                    "@type": "Answer",
                    "text": faq_a
                }
            }
        ]
    }

    page = shared_head(title, desc, canonical, config, json_ld)
    page += shared_nav(prefix="../", config=config)
    page += f"""
<main class="container">
  <section class="hero">
    <div style="display:flex;gap:8px;flex-wrap:wrap;">
      <span class="badge">LLM VRAM &amp; Hardware Calculator</span>
      {'<span class="badge badge-green">Open-Weight Model</span>' if m["is_open_weight"] else '<span class="badge badge-warn">Proprietary API · Open-Weight Equivalent</span>'}
    </div>
    <h1>{hero_h1}</h1>
    <p>{hero_p}</p>
  </section>
  {notice_html}
  <div class="grid-2">
    <div class="card">
      <h2>{card1_title}</h2>
      <div class="stat-row">
        <span>Estimated Parameter Class</span>
        <strong>~{m['params_b']:g}B Parameters</strong>
      </div>
      <div class="stat-row">
        <span>16-Bit Full Precision (FP16 / BF16)</span>
        <strong>{m['vram_fp16']} GB VRAM</strong>
      </div>
      <div class="stat-row">
        <span>8-Bit Quantization (Q8_0 / FP8)</span>
        <strong>{m['vram_int8']} GB VRAM</strong>
      </div>
      <div class="stat-row">
        <span>4-Bit Quantization (Q4_K_M / AWQ / EXL2)</span>
        <strong style="color:var(--accent-green)">{m['vram_int4']} GB VRAM</strong>
      </div>
      <div class="stat-row">
        <span>Recommended Cloud GPU</span>
        <strong>{html.escape(gpu['name'])} (${gpu['hourly_usd']}/hr)</strong>
      </div>
      <div class="stat-row">
        <span>Cloud API Token Pricing</span>
        <strong><a href="../models/{m['slug']}.html">{in_fmt}/1M In · {out_fmt}/1M Out →</a></strong>
      </div>
      <div style="margin-top:20px;display:flex;gap:12px;flex-wrap:wrap;">
        <a href="{html.escape(aff_runpod['url'])}" target="_blank" rel="nofollow sponsored" class="btn-cta">{html.escape(aff_runpod['cta_text'])}</a>
        <a href="{html.escape(aff_vast['url'])}" target="_blank" rel="nofollow sponsored" class="btn-cta btn-green">{html.escape(aff_vast['cta_text'])}</a>
      </div>
    </div>

    <div class="card">
      <h2>{card2_title}</h2>
      <div class="slider-group">
        <label><span>Select Your Local GPU VRAM</span> <span id="myVramLabel">24 GB VRAM</span></label>
        <input type="range" id="myVram" min="8" max="{slider_max}" step="8" value="24" oninput="checkVram()" />
      </div>
      <div id="vramVerdict" style="margin-top:16px;padding:14px;border-radius:8px;background:rgba(56,189,248,0.12);border:1px solid var(--accent);"></div>
      <p style="color:var(--muted);font-size:0.88rem;margin-top:14px;">Don't have enough local VRAM? Rent a pre-configured Ollama / vLLM PyTorch pod on RunPod or Vast.ai in 30 seconds with zero upfront hardware cost.</p>
    </div>
  </div>
</main>
<script>
  const FP16 = {m['vram_fp16']};
  const INT8 = {m['vram_int8']};
  const INT4 = {m['vram_int4']};
  function checkVram() {{
    const v = parseFloat(document.getElementById('myVram').value);
    document.getElementById('myVramLabel').textContent = v + ' GB VRAM';
    const box = document.getElementById('vramVerdict');
    if (v >= FP16) {{
      box.innerHTML = '✅ <strong>Full Speed (FP16 / BF16):</strong> ' + v + ' GB VRAM can run this model class at full 16-bit precision (' + FP16 + ' GB needed) with generous KV-cache context headroom.';
    }} else if (v >= INT8) {{
      box.innerHTML = '🟢 <strong>Near-Lossless 8-Bit (Q8_0 / FP8):</strong> ' + v + ' GB VRAM comfortably runs 8-bit quantization (' + INT8 + ' GB needed) or 4-bit quantization (' + INT4 + ' GB needed) with large context windows. For full unquantized FP16 (' + FP16 + ' GB), rent a Cloud GPU below.';
    }} else if (v >= INT4) {{
      box.innerHTML = '⚡ <strong>Runs 4-Bit Quantized (Q4_K_M / AWQ):</strong> ' + v + ' GB VRAM can run this model class using 4-bit quantization (' + INT4 + ' GB needed). For 8-bit (' + INT8 + ' GB) or full FP16 (' + FP16 + ' GB), rent a Cloud GPU below.';
    }} else {{
      box.innerHTML = '❌ <strong>Not Enough Local VRAM:</strong> Even 4-bit quantization requires ~' + INT4 + ' GB VRAM (you selected ' + v + ' GB). Rent an RTX 4090, A100, or H100 on RunPod / Vast.ai for pennies per hour instead of buying expensive hardware.';
    }}
  }}
  checkVram();
</script>
"""
    page += shared_footer(config, updated_str, prefix="../")
    return vram_slug, page

def build_vram_hub_page(models, config, updated_str):
    base_url = config["base_url"].rstrip("/")
    canonical = f"{base_url}/vram-electricity-calculator.html"
    title = "LLM VRAM Calculator & Local GPU Electricity vs Cloud Cost (2026)"
    desc = "Calculate exact VRAM requirements (FP16, Q8, Q4) for any LLM including context window KV cache, and compare your home GPU electricity bill ($/kWh) against Cloud GPUs."

    aff_runpod = config["affiliates"]["runpod"]
    aff_vast = config["affiliates"]["vastai"]

    # Feature genuinely open-weight models first in the VRAM directory grid
    open_models = [m for m in models if m["is_open_weight"] and not m["is_secondary"]]
    other_models = [m for m in models if not m["is_open_weight"] and not m["is_secondary"]]
    featured_vram_models = (open_models + other_models)[:96]

    vram_links = "".join(
        f'<a href="vram/{m["slug"]}-vram-requirements.html" class="card" style="display:block;padding:14px 18px;text-decoration:none;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;gap:6px;"><strong style="color:#fff;">{html.escape(m["name"])}</strong>'
        f'<span class="badge {"badge-green" if m["is_open_weight"] else "badge-warn"}" style="margin:0;font-size:0.7rem;">{"Open-Weight" if m["is_open_weight"] else "API Eq."}</span></div>'
        f'<div style="color:var(--muted);font-size:0.84rem;margin-top:6px;">~{m["params_b"]:g}B · 4-bit: {m["vram_int4"]} GB · FP16: {m["vram_fp16"]} GB →</div></a>'
        for m in featured_vram_models
    )

    page = shared_head(title, desc, canonical, config)
    page += shared_nav(prefix="", config=config)
    page += f"""
<main class="container">
  <section class="hero">
    <span class="badge">VRAM &amp; Power Bill Calculator</span>
    <h1>LLM VRAM &amp; Home GPU Electricity vs. Cloud Calculator</h1>
    <p>Calculate how much GPU memory your LLM requires across FP16, 8-bit, and 4-bit quantization (including KV cache context overhead) and compare your local PC's power bill against on-demand Cloud GPUs.</p>
  </section>

  <div class="grid-2">
    <div class="card">
      <h2>Interactive LLM VRAM Calculator</h2>
      <div class="slider-group">
        <label><span>Model Size (Billion Parameters)</span> <span id="paramVal">70B</span></label>
        <input type="range" id="paramSlider" min="1" max="405" step="1" value="70" oninput="calcHub()" />
      </div>
      <div class="slider-group">
        <label><span>Context Window (KV Cache Tokens)</span> <span id="ctxVal">8,192 tokens</span></label>
        <input type="range" id="ctxSlider" min="2048" max="131072" step="2048" value="8192" oninput="calcHub()" />
      </div>
      <div class="stat-row">
        <span>FP16 / BF16 Full Precision</span>
        <strong id="hubFp16">154.0 GB VRAM</strong>
      </div>
      <div class="stat-row">
        <span>8-Bit Quantized (Q8_0 / FP8)</span>
        <strong id="hubInt8">80.5 GB VRAM</strong>
      </div>
      <div class="stat-row">
        <span>4-Bit Quantized (Q4_K_M / AWQ)</span>
        <strong id="hubInt4" style="color:var(--accent-green)">45.5 GB VRAM</strong>
      </div>
      <div class="stat-row">
        <span>Recommended GPU Tier</span>
        <strong id="hubGpuRec" style="color:var(--accent)">NVIDIA A100 80GB</strong>
      </div>
    </div>

    <div class="card">
      <h2>Home GPU Electricity Bill Calculator</h2>
      <div class="slider-group">
        <label><span>System Power Draw Under Load (Watts)</span> <span id="wattVal">550 W</span></label>
        <input type="range" id="wattSlider" min="150" max="1800" step="50" value="550" oninput="calcHub()" />
      </div>
      <div class="slider-group">
        <label><span>Electricity Rate ($ / kWh)</span> <span id="kwhVal">$0.18 / kWh</span></label>
        <input type="range" id="kwhSlider" min="0.05" max="0.60" step="0.01" value="0.18" oninput="calcHub()" />
      </div>
      <div class="slider-group">
        <label><span>Active Inference Hours per Day</span> <span id="hrsVal">24 hrs/day</span></label>
        <input type="range" id="hrsSlider" min="1" max="24" step="1" value="24" oninput="calcHub()" />
      </div>
      <div class="stat-row">
        <span>Monthly Energy Consumption</span>
        <strong id="kwhMonthly">396.0 kWh/mo</strong>
      </div>
      <div class="stat-row">
        <span>Monthly Home GPU Power Cost</span>
        <strong id="elecMonthly" class="price-big">$71.28/mo</strong>
      </div>
      <div style="margin-top:16px;display:flex;gap:12px;flex-wrap:wrap;">
        <a href="{html.escape(aff_runpod['url'])}" target="_blank" rel="nofollow sponsored" class="btn-cta">{html.escape(aff_runpod['cta_text'])}</a>
        <a href="{html.escape(aff_vast['url'])}" target="_blank" rel="nofollow sponsored" class="btn-cta btn-green">{html.escape(aff_vast['cta_text'])}</a>
      </div>
    </div>
  </div>

  <h2 style="color:#fff;margin-top:36px;margin-bottom:8px;">Open-Weight &amp; Popular Model VRAM Guides</h2>
  <p style="color:var(--muted);margin-bottom:16px;">Select any model below to view quantization memory tiers and check if your GPU can run it.</p>
  <div class="grid-3">
    {vram_links}
  </div>
</main>
<script>
  function calcHub() {{
    const p = parseFloat(document.getElementById('paramSlider').value);
    const ctx = parseFloat(document.getElementById('ctxSlider').value);
    document.getElementById('paramVal').textContent = p + 'B';
    document.getElementById('ctxVal').textContent = ctx.toLocaleString() + ' tokens';

    // KV cache scales roughly with parameter size and context length beyond 8k baseline
    const kvExtraGb = Math.max(0, (ctx - 8192) / 8192) * (p * 0.025);
    const fp16 = (p * 2.2) + (kvExtraGb * 2);
    const int8 = (p * 1.15) + (kvExtraGb * 1.2);
    const int4 = (p * 0.65) + kvExtraGb;

    document.getElementById('hubFp16').textContent = fp16.toFixed(1) + ' GB VRAM';
    document.getElementById('hubInt8').textContent = int8.toFixed(1) + ' GB VRAM';
    document.getElementById('hubInt4').textContent = int4.toFixed(1) + ' GB VRAM';

    let rec = '1x RTX 4090 (24GB)';
    if (int4 > 140) rec = '4x–8x H100 80GB Cluster';
    else if (int4 > 75) rec = '2x H100 / A100 (160GB VRAM)';
    else if (int4 > 22) rec = '1x A100 / H100 (80GB VRAM)';
    document.getElementById('hubGpuRec').textContent = rec;

    const w = parseFloat(document.getElementById('wattSlider').value);
    const kwh = parseFloat(document.getElementById('kwhSlider').value);
    const hrs = parseFloat(document.getElementById('hrsSlider').value);
    document.getElementById('wattVal').textContent = w + ' W';
    document.getElementById('kwhVal').textContent = '$' + kwh.toFixed(2) + ' / kWh';
    document.getElementById('hrsVal').textContent = hrs + ' hrs/day';
    const monthlyKwh = (w / 1000) * hrs * 30;
    const monthlyCost = monthlyKwh * kwh;
    document.getElementById('kwhMonthly').textContent = monthlyKwh.toFixed(1) + ' kWh/mo';
    document.getElementById('elecMonthly').textContent = '$' + monthlyCost.toFixed(2) + '/mo';
  }}
  calcHub();
</script>
"""
    page += shared_footer(config, updated_str, prefix="")
    return page

def build_comparison_page(m1, m2, config, updated_str):
    base_url = config["base_url"].rstrip("/")
    comp_slug = f"{m1['slug']}-vs-{m2['slug']}"
    canonical = f"{base_url}/compare/{comp_slug}.html"
    title = f"{m1['name']} vs {m2['name']}: API Pricing & Token Cost Comparison (2026)"
    desc = f"Interactive price comparison: {m1['name']} ({format_price(m1['input_1m'])}/1M in) vs {m2['name']} ({format_price(m2['input_1m'])}/1M in). Calculate which LLM is cheaper for your workload."

    cheaper = m1 if m1["blended_1m"] <= m2["blended_1m"] else m2
    aff_runpod = config["affiliates"]["runpod"]
    aff_do = config["affiliates"]["vastai"]

    vram_cell_1 = f'<a href="../vram/{m1["slug"]}-vram-requirements.html">{m1["vram_int4"]} GB (Open-Weight)</a>' if m1["is_open_weight"] else f'<a href="../vram/{m1["slug"]}-vram-requirements.html">API Only (~{m1["vram_int4"]} GB eq.)</a>'
    vram_cell_2 = f'<a href="../vram/{m2["slug"]}-vram-requirements.html">{m2["vram_int4"]} GB (Open-Weight)</a>' if m2["is_open_weight"] else f'<a href="../vram/{m2["slug"]}-vram-requirements.html">API Only (~{m2["vram_int4"]} GB eq.)</a>'

    page = shared_head(title, desc, canonical, config)
    page += shared_nav(prefix="../", config=config)
    page += f"""
<main class="container">
  <section class="hero">
    <span class="badge">Head-to-Head Pricing Benchmark</span>
    <h1>{html.escape(m1['name'])} vs. {html.escape(m2['name'])}</h1>
    <p>Interactive cost comparison based on live API token rates and context window limits. Overall lower blended token cost: <strong>{html.escape(cheaper['name'])}</strong>.</p>
  </section>

  <div class="grid-2">
    <div class="card">
      <h2>Interactive Side-by-Side Cost Calculator</h2>
      <p style="color:var(--muted);font-size:0.9rem;margin-bottom:14px;">Slide your daily token volume to compare monthly bills between both models in real time.</p>
      <div class="slider-group">
        <label><span>Daily Input Tokens</span> <span id="compInVal">5,000,000</span></label>
        <input type="range" id="compInSlider" min="100000" max="50000000" step="100000" value="5000000" oninput="calcComp()" />
      </div>
      <div class="slider-group">
        <label><span>Daily Output Tokens</span> <span id="compOutVal">1,000,000</span></label>
        <input type="range" id="compOutSlider" min="50000" max="20000000" step="50000" value="1000000" oninput="calcComp()" />
      </div>
      <div class="stat-row">
        <span><strong>{html.escape(m1['name'])}</strong> Monthly Cost</span>
        <strong id="m1Monthly" style="font-size:1.25rem;">$0.00/mo</strong>
      </div>
      <div class="stat-row">
        <span><strong>{html.escape(m2['name'])}</strong> Monthly Cost</span>
        <strong id="m2Monthly" style="font-size:1.25rem;">$0.00/mo</strong>
      </div>
      <div id="compWinnerBox" style="margin-top:14px;padding:12px;border-radius:8px;background:rgba(16,185,129,0.12);border:1px solid var(--accent-green);font-size:0.92rem;"></div>
    </div>

    <div class="card">
      <h2>Side-by-Side Specs &amp; Token Rates</h2>
      <div class="table-wrap">
        <table style="min-width:420px;">
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
              <td>Model Type</td>
              <td>{"Open-Weight" if m1["is_open_weight"] else "Proprietary API"}</td>
              <td>{"Open-Weight" if m2["is_open_weight"] else "Proprietary API"}</td>
            </tr>
            <tr>
              <td>Input Cost / 1M</td>
              <td><strong>{format_price(m1['input_1m'])}</strong></td>
              <td><strong>{format_price(m2['input_1m'])}</strong></td>
            </tr>
            <tr>
              <td>Output Cost / 1M</td>
              <td><strong>{format_price(m1['output_1m'])}</strong></td>
              <td><strong>{format_price(m2['output_1m'])}</strong></td>
            </tr>
            <tr>
              <td>Context Window</td>
              <td>{m1['context_length']:,} tokens</td>
              <td>{m2['context_length']:,} tokens</td>
            </tr>
            <tr>
              <td>4-Bit Quantized VRAM</td>
              <td>{vram_cell_1}</td>
              <td>{vram_cell_2}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <div class="affiliate-box">
    <div>
      <span class="badge badge-green">Cloud Infrastructure Deal</span>
      <h3>Deploy Your AI App or Open-Source LLM for Less</h3>
      <p style="color:var(--muted);">Rent serverless or dedicated GPUs on RunPod and Vast.ai to cut your monthly AI infrastructure bill by up to 70%.</p>
    </div>
    <div style="display:flex;gap:12px;flex-wrap:wrap;">
      <a href="{html.escape(aff_runpod['url'])}" target="_blank" rel="nofollow sponsored" class="btn-cta">{html.escape(aff_runpod['cta_text'])}</a>
      <a href="{html.escape(aff_do['url'])}" target="_blank" rel="nofollow sponsored" class="btn-cta btn-green">{html.escape(aff_do['cta_text'])}</a>
    </div>
  </div>
</main>
<script>
  const M1_NAME = {json.dumps(m1['name'])};
  const M2_NAME = {json.dumps(m2['name'])};
  const M1_IN = {m1['input_1m']};
  const M1_OUT = {m1['output_1m']};
  const M2_IN = {m2['input_1m']};
  const M2_OUT = {m2['output_1m']};
  function calcComp() {{
    const inTok = parseFloat(document.getElementById('compInSlider').value);
    const outTok = parseFloat(document.getElementById('compOutSlider').value);
    document.getElementById('compInVal').textContent = inTok.toLocaleString();
    document.getElementById('compOutVal').textContent = outTok.toLocaleString();
    const cost1 = (((inTok / 1e6) * M1_IN) + ((outTok / 1e6) * M1_OUT)) * 30;
    const cost2 = (((inTok / 1e6) * M2_IN) + ((outTok / 1e6) * M2_OUT)) * 30;
    document.getElementById('m1Monthly').textContent = '$' + cost1.toLocaleString(undefined, {{minimumFractionDigits: 2, maximumFractionDigits: 2}}) + '/mo';
    document.getElementById('m2Monthly').textContent = '$' + cost2.toLocaleString(undefined, {{minimumFractionDigits: 2, maximumFractionDigits: 2}}) + '/mo';
    const diff = Math.abs(cost1 - cost2).toLocaleString(undefined, {{minimumFractionDigits: 2, maximumFractionDigits: 2}});
    const winner = cost1 <= cost2 ? M1_NAME : M2_NAME;
    document.getElementById('compWinnerBox').innerHTML = '🏆 <strong>' + winner + ' is cheaper:</strong> Saves you <strong>$' + diff + '/month</strong> at this token volume.';
  }}
  calcComp();
</script>
"""
    page += shared_footer(config, updated_str, prefix="../")
    return comp_slug, page

def build_gpu_page(config, updated_str):
    base_url = config["base_url"].rstrip("/")
    canonical = f"{base_url}/gpu-cloud-pricing.html"
    title = "Cloud GPU Price Comparison (2026): H100, A100, RTX 4090 Hourly Rates"
    desc = "Compare hourly Cloud GPU rental prices across RunPod, Vast.ai, Vultr, and DigitalOcean. Find the cheapest GPU for LLM inference and fine-tuning."

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
        desc_text = aff.get(
            "description",
            "High-performance cloud compute and GPU instances ready for AI agents, vLLM, Ollama, and production APIs."
        )
        cards += f"""
        <div class="card">
          <span class="badge badge-green">{html.escape(aff['badge'])}</span>
          <h3>{html.escape(aff['name'])}</h3>
          <p style="color:var(--muted);font-size:0.92rem;margin:10px 0 18px;">{html.escape(desc_text)}</p>
          <a href="{html.escape(aff['url'])}" target="_blank" rel="nofollow sponsored" class="btn-cta">{html.escape(aff['cta_text'])} →</a>
        </div>
        """

    page = shared_head(title, desc, canonical, config)
    page += shared_nav(prefix="", config=config)
    page += f"""
<main class="container">
  <section class="hero">
    <span class="badge">Live Cloud GPU &amp; VPS Benchmark</span>
    <h1>Cloud GPU Hourly Price Comparison (H100, A100, RTX 4090)</h1>
    <p>Compare on-demand GPU instances and claim free developer cloud credits for self-hosting LLMs, Stable Diffusion, and AI agents.</p>
  </section>

  <div class="card">
    <h2>On-Demand Cloud GPU Pricing Matrix</h2>
    <div class="table-wrap">
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

  <h2 style="margin-top:36px;color:#fff;">Verified Cloud GPU &amp; VPS Partner Offers</h2>
  <div class="grid-3">
    {cards}
  </div>
</main>
"""
    page += shared_footer(config, updated_str, prefix="")
    return page

def build_index_page(models, comparisons, config, updated_str, total_pages):
    base_url = config["base_url"].rstrip("/")
    canonical = f"{base_url}/index.html"
    title = f"{config['site_name']} — {config['site_tagline']}"
    desc = "Compare live API token pricing across 300+ AI models (OpenAI, Anthropic, DeepSeek, Meta Llama, Qwen, Mistral, Google Gemini), calculate LLM VRAM requirements, and compare Cloud GPU rental rates."

    # Display all clean primary models in the index directory so every family (Llama, Qwen, Mistral, DeepSeek, OpenAI, Anthropic, Google, xAI) is searchable
    display_models = [m for m in models if not m["is_secondary"]][:260]

    table_rows = ""
    for m in display_models:
        search_terms = f"{m['name'].lower()} {m['provider'].lower()} {m['id'].lower()} {'open-weight' if m['is_open_weight'] else 'proprietary api'}"
        if m["is_open_weight"]:
            vram_col = f'<a href="vram/{m["slug"]}-vram-requirements.html"><strong>{m["vram_int4"]} GB</strong></a> <small style="color:var(--accent-green)">(Open)</small>'
        else:
            vram_col = f'<a href="vram/{m["slug"]}-vram-requirements.html" style="color:var(--muted)" title="Proprietary API — click for open-weight equivalent VRAM">API (~{m["vram_int4"]} GB eq.)</a>'

        table_rows += f"""
        <tr class="model-row" data-name="{html.escape(search_terms)}" data-open="{'1' if m['is_open_weight'] else '0'}">
          <td><a href="models/{m['slug']}.html"><strong>{html.escape(m['name'])}</strong></a><br/><small style="color:var(--muted)">{html.escape(m['id'])}</small></td>
          <td>{html.escape(m['provider'])}</td>
          <td>{format_price(m['input_1m'])}</td>
          <td>{format_price(m['output_1m'])}</td>
          <td>{m['context_length']:,}</td>
          <td>{vram_col}</td>
          <td><a href="models/{m['slug']}.html">Calculator →</a></td>
        </tr>
        """

    comp_links = ""
    for c_slug, m1, m2 in comparisons[:90]:
        comp_links += f"""
        <a href="compare/{c_slug}.html" class="card" style="display:block;padding:14px 18px;text-decoration:none;">
          <strong style="color:#fff;">{html.escape(m1['name'])} vs {html.escape(m2['name'])}</strong>
          <div style="color:var(--muted);font-size:0.85rem;margin-top:4px;">{format_price(m1['input_1m'])}/1M vs {format_price(m2['input_1m'])}/1M input tokens →</div>
        </a>
        """

    aff_main = config["affiliates"]["runpod"]
    aff_sec = config["affiliates"]["vastai"]

    page = shared_head(title, desc, canonical, config)
    page += shared_nav(prefix="", config=config)
    page += f"""
<main class="container">
  <section class="hero">
    <span class="badge">Auto-Updated Daily · {total_pages:,}+ Live AI Pricing &amp; VRAM Calculators</span>
    <h1>AI API Token Cost, VRAM &amp; Cloud GPU Break-Even Engine</h1>
    <p>Instantly compare per-million token costs across OpenAI, Anthropic, Google, DeepSeek, Qwen, Mistral, and Meta Llama models—and see the exact point where renting a Cloud GPU saves you money.</p>
  </section>

  <div class="affiliate-box">
    <div>
      <span class="badge badge-green">Featured Infrastructure Deals</span>
      <h3>Building an AI Product or Running Open-Weight LLMs?</h3>
      <p style="color:var(--muted);">Deploy dedicated RTX 4090, A100, and H100 GPUs from $0.20/hr with instant PyTorch, vLLM, and Ollama templates.</p>
    </div>
    <div style="display:flex;gap:12px;flex-wrap:wrap;">
      <a href="{html.escape(aff_main['url'])}" target="_blank" rel="nofollow sponsored" class="btn-cta">{html.escape(aff_main['cta_text'])}</a>
      <a href="{html.escape(aff_sec['url'])}" target="_blank" rel="nofollow sponsored" class="btn-cta btn-green">{html.escape(aff_sec['cta_text'])}</a>
    </div>
  </div>

  <div class="card">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;margin-bottom:14px;">
      <h2>Live AI Model Pricing &amp; VRAM Directory (Per 1M Tokens)</h2>
      <input type="text" id="searchInput" placeholder="Filter models (e.g. gpt-5, claude, deepseek, llama, qwen, mistral)..." style="max-width:380px;" oninput="filterTable()" />
    </div>
    <div class="filter-bar">
      <button type="button" class="filter-pill active" onclick="setQuickFilter('', this)">All ({len(display_models)})</button>
      <button type="button" class="filter-pill" onclick="setQuickFilter('OPEN_ONLY', this)">🟢 Open-Weight Only</button>
      <button type="button" class="filter-pill" onclick="setQuickFilter('openai', this)">OpenAI</button>
      <button type="button" class="filter-pill" onclick="setQuickFilter('anthropic', this)">Anthropic</button>
      <button type="button" class="filter-pill" onclick="setQuickFilter('google', this)">Google</button>
      <button type="button" class="filter-pill" onclick="setQuickFilter('deepseek', this)">DeepSeek</button>
      <button type="button" class="filter-pill" onclick="setQuickFilter('llama', this)">Meta Llama</button>
      <button type="button" class="filter-pill" onclick="setQuickFilter('qwen', this)">Qwen</button>
      <button type="button" class="filter-pill" onclick="setQuickFilter('mistral', this)">Mistral</button>
      <button type="button" class="filter-pill" onclick="setQuickFilter('grok', this)">xAI Grok</button>
    </div>
    <div class="table-wrap">
      <table id="pricingTable">
        <thead>
          <tr>
            <th>Model Name</th>
            <th>Provider</th>
            <th>Input / 1M</th>
            <th>Output / 1M</th>
            <th>Context</th>
            <th>Q4 VRAM / Type</th>
            <th>Calculator</th>
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
    <p style="color:var(--muted);margin-bottom:18px;">Click any pair below to view interactive side-by-side token calculators and monthly cost breakdowns.</p>
    <div class="grid-3">
      {comp_links}
    </div>
  </section>
</main>
<script>
  let currentMode = '';
  function setQuickFilter(q, btn) {{
    document.querySelectorAll('.filter-pill').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const input = document.getElementById('searchInput');
    if (q === 'OPEN_ONLY') {{
      currentMode = 'OPEN_ONLY';
      input.value = '';
    }} else {{
      currentMode = '';
      input.value = q;
    }}
    filterTable();
  }}
  function filterTable() {{
    const q = document.getElementById('searchInput').value.toLowerCase().trim();
    const rows = document.querySelectorAll('.model-row');
    rows.forEach(r => {{
      const name = r.getAttribute('data-name') || '';
      const isOpen = r.getAttribute('data-open') === '1';
      const matchesOpen = (currentMode !== 'OPEN_ONLY') || isOpen;
      const matchesText = !q || name.includes(q);
      r.style.display = (matchesOpen && matchesText) ? '' : 'none';
    }});
  }}
</script>
"""
    page += shared_footer(config, updated_str, prefix="")
    return page

def submit_indexnow(config, urls):
    key = config.get("indexnow_key", "").strip()
    if not key:
        return
    base_url = config["base_url"].rstrip("/")
    host = base_url.split("://")[-1].split("/")[0]
    payload = {
        "host": host,
        "key": key,
        "keyLocation": f"{base_url}/{key}.txt",
        "urlList": urls[:1000]
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://api.indexnow.org/indexnow",
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            print(f"IndexNow ping succeeded with HTTP {resp.status} for {len(payload['urlList'])} URLs!")
    except Exception as e:
        print(f"IndexNow ping note: {e}")

def main():
    config = load_config()
    updated_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print("Fetching live AI model pricing data...")
    all_models = fetch_models()
    max_models = int(config.get("max_individual_model_pages", 450))
    models = all_models[:max_models]
    print(f"Selected {len(models)} models ({sum(1 for m in models if m['is_open_weight'])} open-weight) for programmatic calculator pages.")

    # Clean old comparison files so only balanced cross-provider comparisons exist
    comp_dir = os.path.join(DIST_DIR, "compare")
    if os.path.exists(comp_dir):
        shutil.rmtree(comp_dir)

    os.makedirs(os.path.join(DIST_DIR, "models"), exist_ok=True)
    os.makedirs(comp_dir, exist_ok=True)
    os.makedirs(os.path.join(DIST_DIR, "vram"), exist_ok=True)

    urls = []
    base_url = config["base_url"].rstrip("/")

    # 1. Build individual model calculator pages
    for m in models:
        html_out = build_model_page(m, models, config, updated_str)
        out_path = os.path.join(DIST_DIR, "models", f"{m['slug']}.html")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html_out)
        urls.append(f"{base_url}/models/{m['slug']}.html")

    # 2. Build VRAM calculator pages for all models so every model's VRAM link works
    max_vram = max(len(models), int(config.get("max_vram_pages", 450)))
    for m in models[:max_vram]:
        v_slug, v_html = build_vram_page(m, config, updated_str)
        v_path = os.path.join(DIST_DIR, "vram", f"{v_slug}.html")
        with open(v_path, "w", encoding="utf-8") as f:
            f.write(v_html)
        urls.append(f"{base_url}/vram/{v_slug}.html")

    # 3. Build comparison pages for top primary models (diverse across OpenAI, Anthropic, Google, DeepSeek, Llama, Qwen, Mistral, xAI)
    max_comp = int(config.get("max_comparison_pages", 650))
    primary_for_comp = [m for m in models if not m["is_secondary"]][:38]
    comparisons = []
    for i in range(len(primary_for_comp)):
        for j in range(i + 1, len(primary_for_comp)):
            if len(comparisons) >= max_comp:
                break
            m1, m2 = primary_for_comp[i], primary_for_comp[j]
            c_slug, c_html = build_comparison_page(m1, m2, config, updated_str)
            c_path = os.path.join(comp_dir, f"{c_slug}.html")
            with open(c_path, "w", encoding="utf-8") as f:
                f.write(c_html)
            comparisons.append((c_slug, m1, m2))
            urls.append(f"{base_url}/compare/{c_slug}.html")
        if len(comparisons) >= max_comp:
            break

    # 4. Build VRAM & Electricity Hub Page
    vram_hub_html = build_vram_hub_page(models, config, updated_str)
    with open(os.path.join(DIST_DIR, "vram-electricity-calculator.html"), "w", encoding="utf-8") as f:
        f.write(vram_hub_html)
    urls.insert(0, f"{base_url}/vram-electricity-calculator.html")

    # 5. Build Cloud GPU comparison page
    gpu_html = build_gpu_page(config, updated_str)
    with open(os.path.join(DIST_DIR, "gpu-cloud-pricing.html"), "w", encoding="utf-8") as f:
        f.write(gpu_html)
    urls.insert(0, f"{base_url}/gpu-cloud-pricing.html")

    # 6. Build Main Index Hub
    index_html = build_index_page(models, comparisons, config, updated_str, len(urls) + 1)
    with open(os.path.join(DIST_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(index_html)
    urls.insert(0, f"{base_url}/index.html")

    # 7. Build sitemap.xml & robots.txt & .nojekyll & IndexNow key file
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

    indexnow_key = config.get("indexnow_key", "").strip()
    if indexnow_key:
        with open(os.path.join(DIST_DIR, f"{indexnow_key}.txt"), "w", encoding="utf-8") as f:
            f.write(indexnow_key)

    print(f"Successfully generated {len(urls)} total SEO pages in {DIST_DIR}")
    if "--ping-indexnow" in sys.argv:
        submit_indexnow(config, urls)

if __name__ == "__main__":
    main()
