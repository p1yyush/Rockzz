"""VeriNews AI backend.
Run:  python app.py   then open http://localhost:5000
Optional: set ANTHROPIC_API_KEY to power the chatbot with Claude (otherwise built-in answers are used)."""
import os, re, urllib.parse
import xml.etree.ElementTree as ET
import requests
from flask import Flask, request, jsonify, send_from_directory
try:
    from flask_cors import CORS
except ImportError:          # keep working even if flask-cors isn't installed
    CORS = None

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=BASE, static_url_path="")
if CORS:
    CORS(app)
else:
    @app.after_request
    def _cors(resp):
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return resp

# ---------- ML model (optional: falls back to a heuristic if pkl files are missing) ----------
vec = model = None
try:
    import joblib
    vec = joblib.load(os.path.join(BASE, "tfidf.pkl"))
    model = joblib.load(os.path.join(BASE, "model.pkl"))
    REAL_IDX = list(model.classes_).index(1)   # label 1 = real. Change if your labels differ.
    print("ML model loaded.")
except Exception as e:
    print("ML model not found, using heuristic style score. (%s)" % e)

def style_score(text):
    """Returns 0-100: how much the writing looks like real news."""
    if model is not None:
        return float(model.predict_proba(vec.transform([text]))[0][REAL_IDX]) * 100
    s = 70
    s -= len(re.findall(r"\b[A-Z]{4,}\b", text)) * 8 + text.count("!") * 6
    if re.search(r"shocking|share this|you won't believe|secret|miracle", text, re.I): s -= 25
    if re.search(r"said|officials|according to|announced|\d", text, re.I): s += 12
    return max(3.0, min(97.0, float(s)))

# ---------- Source check via Google News RSS ----------
TRUSTED = ["the hindu", "reuters", "bbc", "ndtv", "times of india", "indian express", "hindustan times",
           "ap news", "associated press", "the guardian", "pti", "the economic times", "mint", "dd news"]
FACT = ["alt news", "boom", "snopes", "factly", "pib fact check", "factcheck", "fact check"]

def classify(outlet, title):
    o, t = outlet.lower(), title.lower()
    if any(f in o for f in FACT) or "fact check" in t or "fact-check" in t: return "factcheck"
    if any(x in o for x in TRUSTED): return "trusted"
    return "other"

def search_news(text, limit=8):
    query = " ".join(re.sub(r"[^\w\s]", " ", text).split()[:14])
    url = "https://news.google.com/rss/search?q=%s&hl=en-IN&gl=IN&ceid=IN:en" % urllib.parse.quote(query)
    try:
        r = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        items = ET.fromstring(r.content).findall(".//item")[:limit]
    except Exception as e:
        print("News search failed:", e)
        return []
    out = []
    for it in items:
        title = it.findtext("title", "")
        src = it.find("source")
        outlet = (src.text if src is not None and src.text else title.rsplit(" - ", 1)[-1]).strip()
        title = title.rsplit(" - ", 1)[0] if " - " in title else title
        out.append({"title": title, "url": it.findtext("link", ""), "outlet": outlet,
                    "type": classify(outlet, title)})
    # trusted + fact-checkers first
    order = {"factcheck": 0, "trusted": 1, "other": 2}
    return sorted(out, key=lambda x: order[x["type"]])

# ---------- Latest news by category (Google News RSS) ----------
from email.utils import parsedate_to_datetime
from datetime import timezone
RSS = "https://news.google.com/rss"
LOC = "hl=en-IN&gl=IN&ceid=IN:en"
TOPICS = {"world": "WORLD", "india": "NATION", "business": "BUSINESS", "technology": "TECHNOLOGY",
          "entertainment": "ENTERTAINMENT", "sports": "SPORTS", "science": "SCIENCE", "health": "HEALTH"}
SEARCHES = {"politics": "politics OR election OR parliament OR government when:2d"}

def feed_url(cat):
    if cat in TOPICS: return "%s/headlines/section/topic/%s?%s" % (RSS, TOPICS[cat], LOC)
    if cat in SEARCHES: return "%s/search?q=%s&%s" % (RSS, urllib.parse.quote(SEARCHES[cat]), LOC)
    return "%s?%s" % (RSS, LOC)

@app.get("/news")
def news():
    cat = request.args.get("category", "top").lower()
    try: limit = max(1, min(int(request.args.get("limit", 6)), 12))
    except ValueError: limit = 6
    try:
        r = requests.get(feed_url(cat), timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        entries = ET.fromstring(r.content).findall(".//item")
    except Exception as e:
        print("News fetch failed:", e)
        return jsonify(error="news fetch failed", detail=str(e)[:140], items=[]), 502
    items, seen = [], set()
    for it in entries:
        raw = it.findtext("title", "")
        try:
            dt = parsedate_to_datetime(it.findtext("pubDate"))
            if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
            published = dt.astimezone(timezone.utc)
        except Exception:
            continue
        src = it.find("source")
        outlet = (src.text if src is not None and src.text else raw.rsplit(" - ", 1)[-1]).strip()
        title = raw.rsplit(" - ", 1)[0] if " - " in raw else raw
        if title in seen: continue
        seen.add(title)
        items.append({"title": title, "url": it.findtext("link", ""), "outlet": outlet,
                      "type": classify(outlet, title), "published": published.isoformat()})
    items.sort(key=lambda x: x["published"], reverse=True)   # newest first
    return jsonify(category=cat, items=items[:limit])

# ---------- Routes ----------
@app.get("/")
def home():
    return send_from_directory(BASE, "index.html")

@app.get("/health")
def health():
    return jsonify(status="ok", model_loaded=model is not None, llm=bool(os.getenv("ANTHROPIC_API_KEY")))

@app.post("/analyze")
def analyze():
    text = (request.get_json(silent=True) or {}).get("text", "").strip()
    if not text:
        return jsonify(error="No text provided"), 400
    style = style_score(text)
    sources = search_news(text)
    trusted = sum(s["type"] == "trusted" for s in sources)
    checked = any(s["type"] == "factcheck" for s in sources)

    if checked:          verdict, src = "DISPUTED", 30
    elif trusted >= 2:   verdict, src = "REAL", 95
    elif trusted == 1:   verdict, src = "LIKELY REAL", 75
    elif not sources and style < 40: verdict, src = "LIKELY FAKE", 10
    else:                verdict, src = "UNVERIFIED", 40
    return jsonify(style_score=round(style, 1), source_score=src,
                   credibility=round(0.5 * style + 0.5 * src), verdict=verdict, sources=sources)

SYSTEM_PROMPT = """You are the assistant on the "AI Fake News Detector" website. Explain how the tool works and help users judge news credibility. Keep answers short, simple, and friendly (2-4 sentences).
HOW IT WORKS: user types a claim; an ML model (TF-IDF + Logistic Regression, ~44,000 Kaggle articles, 98% test accuracy) scores how much the writing style looks like real news; the tool searches Google News; it checks for trusted outlets (The Hindu, Reuters, BBC, NDTV) and fact-checkers (Alt News, BOOM, Snopes); verdicts: REAL (2+ trusted sources), LIKELY REAL (1), DISPUTED (fact-checker covered it), LIKELY FAKE (no sources and fake-style wording), UNVERIFIED (no trusted source found).
PATTERNS: real news has named sources, dates, numbers, neutral tone; fake news has ALL CAPS, '!!!', 'SHOCKING', 'share this', vague claims, emotional language, no named source.
LIMITATIONS (be honest): checks style and source presence, not factual truth; model trained on US political news 2016-17 so less reliable elsewhere; Unverified does not mean fake; not a replacement for human fact-checking.
RULES: never claim a specific story is definitely true or false; tell users to use the Check button and read source links. Tips: check original source, search trusted outlets, check date, reverse-search images, look for fact-checkers. If asked about unrelated topics, politely say you only help with news credibility and this tool."""

def context_block(ctx):
    if not ctx:
        return "\n\nCURRENT ANALYSIS: the user has not analyzed any article yet. If they ask about a result, tell them to paste an article in the analyzer and press Check credibility."
    r = ctx.get("result", {})
    srcs = "; ".join("%s (%s): %s" % (s.get("outlet"), s.get("type"), s.get("title")) for s in r.get("sources", [])) or "none found"
    return ("\n\nCURRENT ANALYSIS (the article the user just checked):\nArticle excerpt: \"%s\"\nVerdict: %s | Overall credibility: %s%% | Style score: %s%% | Source score: %s%%\nSources: %s\n"
            "Use this to explain the result in plain words, quote the specific suspicious phrases from the excerpt (ALL CAPS, '!!!', clickbait words, missing attribution, no numbers), and give tailored steps to verify THIS claim. Use **bold** for key points and short bullet lines starting with '•'."
            % (ctx.get("text", ""), r.get("verdict"), r.get("credibility"), r.get("style_score"), r.get("source_score"), srcs))

@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return jsonify(error="LLM not configured"), 503   # frontend uses its built-in context-aware answers
    try:
        import anthropic
        msgs = [m for m in data.get("history", []) if m.get("role") in ("user", "assistant")][-8:]
        if not msgs or msgs[-1]["role"] != "user":
            msgs.append({"role": "user", "content": data.get("message", "")})
        resp = anthropic.Anthropic(api_key=key).messages.create(
            model="claude-sonnet-5-5", max_tokens=450,
            system=SYSTEM_PROMPT + context_block(data.get("context")), messages=msgs)
        return jsonify(reply=resp.content[0].text)
    except Exception as e:
        print("Chat failed:", e)
        return jsonify(error="chat failed"), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
