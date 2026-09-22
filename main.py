"""
公众号深度改写工具 · 云端后端 (FastAPI, 轻量版)
部署目标: Render 免费实例 (512MB) · 全开源免费 · 任意 Python 版本可构建

设计取舍 (v3.2 轻量):
  - 抓取改用 jina.ai Reader (https://r.jina.ai/<url>): 免费、免密钥、适配绝大多数
    文章页与视频页描述, 无需 trafilatura/yt-dlp 等重型依赖, 构建稳定。
  - 语音转文字 (ASR) 由前端浏览器 Whisper 完成 (已内置), 不占服务器算力/内存,
    且视频不上传服务器, 更隐私。服务端不跑 sherpa-onnx, 避免免费实例构建失败。
  - 改写走 OpenRouter (主) + 可选 Groq/Gemini 环境变量备援。
  - 配图走 Pollinations 代理, 规避浏览器 CORS。

端点:
  GET  /health                健康检查
  POST /api/grab   {url, cookie?}            抓取正文 (jina reader)
  POST /api/rewrite {text, model?, apiKey?}  多模型深度改写
  POST /api/image  {prompt}                  AI 配图 (Pollinations 代理)
"""
import os
import json
import asyncio
import urllib.parse
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, FileResponse

app = FastAPI(title="wechat-rewriter-backend", version="3.2-light")

ALLOWED_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = Path(__file__).parent
STATIC_INDEX = BASE_DIR / "static" / "index.html"


@app.get("/")
async def index():
    if STATIC_INDEX.exists():
        # 运行时把占位符替换为环境变量里的 OpenRouter Key（避免把密钥提交到公开仓库）
        html = STATIC_INDEX.read_text(encoding="utf-8")
        key = os.environ.get("OPENROUTER_KEY", "")
        html = html.replace("__OPENROUTER_KEY__", key)
        return HTMLResponse(html, media_type="text/html")
    return HTMLResponse("<h1>前端文件缺失</h1><p>请将 dist/index.html 放到 backend/static/。</p>")


@app.get("/health")
async def health():
    return {"service": "ok", "grab": "jina", "rewrite": "openrouter",
            "asr": "browser_whisper"}


# ---------------------------------------------------------------------------
# /api/grab —— 抓取正文 (jina.ai Reader, 免费免密钥)
# ---------------------------------------------------------------------------
_VIDEO_HOSTS = ("youtube.com", "youtu.be", "bilibili.com", "douyin.com",
                "kuaishou.com", "xiaohongshu.com", "v.", "tiktok.com",
                "x.com", "twitter.com", "instagram.com")


@app.post("/api/grab")
async def grab(payload: dict):
    url = (payload.get("url") or "").strip()
    if not url:
        raise HTTPException(400, "url required")

    text, title = await _jina_read(url)
    if not text or len(text) < 30:
        return {"ok": False, "text": "", "error": "抓取内容过短或目标反爬",
                "fallback": "upload_or_paste"}

    is_video = any(h in url for h in _VIDEO_HOSTS)
    return {"ok": True, "type": "video" if is_video else "article",
            "source": "jina", "title": title, "text": text,
            "images": [], "needs_asr": is_video}


async def _jina_read(url: str):
    import httpx
    target = "https://r.jina.ai/" + url
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as c:
            r = await c.get(target,
                            headers={"Accept": "text/plain",
                                     "X-With-Images-Summary": "true"})
        if r.status_code != 200:
            return "", ""
        body = r.text
        # jina 返回格式: 第一行通常是标题(以 # 开头) 或元信息, 之后是正文
        lines = body.splitlines()
        title = ""
        start = 0
        for i, ln in enumerate(lines):
            if ln.startswith("Title:"):
                title = ln[len("Title:"):].strip()
                start = i + 1
                break
            if ln.startswith("# "):
                title = ln[2:].strip()
                start = i + 1
                break
        text = "\n".join(lines[start:]).strip()
        return text, title
    except Exception:
        return "", ""


# ---------------------------------------------------------------------------
# /api/rewrite —— 多模型深度改写
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "你是一位资深公众号爆款主编。请对给定内容进行【深度改写】而非同义词替换：\n"
    "1) 重构叙事结构与论点顺序, 改变切入角度与逻辑脉络;\n"
    "2) 用自己的话重述, 保留核心事实与数据, 但表达方式完全不同, 避免与原文连续 5 字以上雷同 (防搬运/原创度>80);\n"
    "3) 补足可读性: 小标题、金句、互动结尾; 口语化但不低质;\n"
    "4) 不要编造未提及的事实或数据。\n"
    "直接输出改写后的完整图文内容 (含小标题), 不要解释你的改写策略。"
)


def _provider_list(api_key: Optional[str]):
    providers = []
    or_key = api_key or os.environ.get("OPENROUTER_KEY")
    if or_key:
        providers.append(("https://openrouter.ai/api/v1", or_key))
    if os.environ.get("GROQ_KEY"):
        providers.append(("https://api.groq.com/openai/v1", os.environ["GROQ_KEY"]))
    if os.environ.get("GEMINI_KEY"):
        providers.append(("https://generativelanguage.googleapis.com/v1beta/openai/",
                          os.environ["GEMINI_KEY"]))
    if not providers:
        raise RuntimeError("no LLM provider configured")
    return providers


@app.post("/api/rewrite")
async def rewrite(payload: dict):
    text = (payload.get("text") or "").strip()
    if not text:
        raise HTTPException(400, "text required")
    model = payload.get("model") or os.environ.get(
        "DEFAULT_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
    api_key = payload.get("apiKey") or None

    last_err = None
    for base_url, key in _provider_list(api_key):
        try:
            out = await _call_llm(base_url, key, model, text)
            return {"ok": True, "model": model, "provider": base_url, "text": out}
        except Exception as e:
            last_err = str(e)
            continue
    return {"ok": False, "error": last_err or "all providers failed",
            "fallback": "try_another_model_or_key"}


async def _call_llm(base_url: str, api_key: str, model: str, text: str) -> str:
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        temperature=0.8,
        timeout=120,
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# /api/image —— AI 配图 (Pollinations 代理, 规避 CORS)
# ---------------------------------------------------------------------------
@app.post("/api/image")
async def image(payload: dict):
    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(400, "prompt required")
    import httpx
    url = ("https://image.pollinations.ai/prompt/"
           + urllib.parse.quote(prompt, safe="")
           + "?width=1024&height=576&nologo=true")
    r = httpx.get(url, follow_redirects=True, timeout=60)
    if r.status_code != 200:
        raise HTTPException(502, "image gen failed")
    return Response(content=r.content, media_type="image/jpeg")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
