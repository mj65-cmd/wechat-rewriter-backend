"""
公众号深度改写工具 · 云端抓取/改写后端 (FastAPI)
部署目标: Render 免费实例 (512MB) · 全开源免费 · 无卡

端点:
  GET  /health                  健康检查 + 工具链状态
  POST /api/grab    {url, cookie?}            抓取正文/字幕/图片
  POST /api/asr     (file)                     语音转文字 (SenseVoice)
  POST /api/rewrite {text, model?, apiKey?}   多模型深度改写
  POST /api/image   {prompt}                   AI 配图 (Pollinations 代理)

设计原则: 所有重依赖 (trafilatura/yt-dlp/sherpa-onnx) 均为惰性导入 + 优雅降级,
          任一组件缺失/失败都返回结构化信号, 由前端决定兜底路径, 不整体停摆。
"""
import os
import sys
import json
import asyncio
import tempfile
import subprocess
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

app = FastAPI(title="wechat-rewriter-backend", version="3.1")

# CORS: 允许前端 HF Static Space 域名 (以及本地调试)
ALLOWED_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(os.path.dirname(__file__), "models"))
os.makedirs(MODELS_DIR, exist_ok=True)


# ----------------------------------------------------------------------------
# /health
# ----------------------------------------------------------------------------
@app.get("/health")
async def health():
    status = {
        "service": "ok",
        "article_extract": _probe("trafilatura"),
        "video_extract": _probe("yt_dlp"),
        "asr_sensevoice": "ready" if os.path.exists(os.path.join(MODELS_DIR, "sense-voice-small.int8.onnx")) else "model_not_loaded",
    }
    return status


def _probe(module: str) -> str:
    try:
        __import__(module)
        return "ready"
    except Exception:
        return "unavailable"


# ----------------------------------------------------------------------------
# /api/grab  —— 抓取正文 / 字幕 / 图片
# ----------------------------------------------------------------------------
@app.post("/api/grab")
async def grab(payload: dict):
    url = (payload.get("url") or "").strip()
    cookie = (payload.get("cookie") or "").strip()
    if not url:
        raise HTTPException(400, "url required")

    # 1) 图文 / 文章 (trafilatura)
    if _looks_like_article(url):
        try:
            return await _grab_article(url, cookie)
        except Exception as e:
            # 失败也不抛, 让前端知道可降级
            return _fail("article", str(e))

    # 2) 视频 / 短视频
    try:
        return await _grab_video(url, cookie)
    except Exception as e:
        return _fail("video", str(e))


def _looks_like_article(url: str) -> bool:
    video_hosts = ("youtube.com", "youtu.be", "bilibili.com", "douyin.com",
                   "kuaishou.com", "xiaohongshu.com", "v.", "tiktok.com", "x.com", "twitter.com")
    return not any(h in url for h in video_hosts)


async def _grab_article(url: str, cookie: str):
    import trafilatura
    downloaded = trafilatura.fetch_url(url, timeout=20)
    if not downloaded:
        raise RuntimeError("fetch failed")
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not text or len(text) < 50:
        raise RuntimeError("extracted too short")
    return {"ok": True, "type": "article", "source": "trafilatura",
            "title": "", "text": text.strip(), "images": [], "needs_asr": False}


async def _grab_video(url: str, cookie: str):
    import yt_dlp
    ydl_opts = {
        "quiet": True, "skip_download": True, "no_warnings": True,
        "http_headers": {"Cookie": cookie} if cookie else {},
        "extractor_args": {"douyin": {"api_hostname": "aweme.snssdk.com"}} if "douyin" in url else {},
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    title = info.get("title", "")
    desc = info.get("description", "")
    subtitles = info.get("subtitles") or info.get("automatic_captions") or {}
    # 有字幕优先返回字幕文本
    if subtitles:
        langs = list(subtitles.keys())
        best = subtitles.get("zh-Hans") or subtitles.get("zh") or subtitles.get(langs[0])
        if best:
            return {"ok": True, "type": "video", "source": "yt-dlp",
                    "title": title, "text": desc, "subtitles_lang": langs[0],
                    "images": [], "needs_asr": False}
    # 口播平台 (抖音/快手/小红书) 无字幕 -> 标记需要 ASR
    is_oral = any(h in url for h in ("douyin.com", "kuaishou.com", "xiaohongshu.com", "tiktok.com"))
    return {"ok": True, "type": "video", "source": "yt-dlp", "title": title,
            "text": desc, "images": [], "needs_asr": is_oral,
            "note": "口播内容，请在下一步走 ASR 语音转文字" if is_oral else ""}


def _fail(kind: str, reason: str):
    return {"ok": False, "type": kind, "source": "none",
            "text": "", "images": [], "needs_asr": False,
            "error": reason, "fallback": "upload_or_paste"}


# ----------------------------------------------------------------------------
# /api/asr  —— 语音转文字 (SenseVoice via sherpa-onnx, CPU 友好)
# ----------------------------------------------------------------------------
_asr_model = None


@app.post("/api/asr")
async def asr(file: UploadFile = File(...)):
    data = await file.read()
    suffix = os.path.splitext(file.filename or "audio.mp3")[1] or ".mp3"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(data)
        path = f.name
    try:
        text = _transcribe(path)
        return {"ok": True, "source": "sensevoice", "text": text}
    except Exception as e:
        # 服务端 ASR 不可用时, 前端自动切浏览器 Whisper 兜底
        return {"ok": False, "source": "sensevoice", "text": "",
                "error": str(e), "fallback": "browser_whisper"}
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass


def _transcribe(path: str) -> str:
    """惰性加载 sherpa-onnx SenseVoice, 首次运行按需下载模型。"""
    global _asr_model
    if _asr_model is None:
        import numpy as np
        import sherpa_onnx
        model_path = os.path.join(MODELS_DIR, "sense-voice-small.int8.onnx")
        tokens_path = os.path.join(MODELS_DIR, "tokens.txt")
        if not os.path.exists(model_path):
            _download_sensevoice()
        _asr_model = sherpa_onnx.SenseVoice(
            model=model_path, tokens=tokens_path,
            language="auto", use_itn=True, num_threads=4,
        )
    import numpy as np
    import soundfile as sf
    samples, sr = sf.read(path)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if sr != 16000:
        # 简单重采样
        from scipy.signal import resample
        samples = resample(samples, int(len(samples) * 16000 / sr))
        sr = 16000
    stream = _asr_model.create_stream()
    stream.accept_waveform(sr, samples.astype(np.float32))
    tail = np.zeros(int(0.3 * sr), dtype=np.float32)
    stream.accept_waveform(sr, tail)
    _asr_model.decode(stream)
    return stream.result.text


def _download_sensevoice():
    """从 HuggingFace 拉取 SenseVoice ONNX 模型 (int8)。失败则抛出, 由调用方降级。"""
    base = "https://huggingface.co/csukuangfj/sense-voice-small-onnx/resolve/main"
    files = {
        "sense-voice-small.int8.onnx": f"{base}/sense-voice-small.int8.onnx",
        "tokens.txt": f"{base}/tokens.txt",
    }
    import httpx
    for name, url in files.items():
        r = httpx.get(url, follow_redirects=True, timeout=120)
        r.raise_for_status()
        with open(os.path.join(MODELS_DIR, name), "wb") as f:
            f.write(r.content)


# ----------------------------------------------------------------------------
# /api/rewrite  —— 多模型深度改写 (OpenRouter 主, Groq/Gemini 备)
# ----------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "你是一位资深公众号爆款主编。请对给定内容进行【深度改写】而非同义词替换：\n"
    "1) 重构叙事结构与论点顺序, 改变切入角度与逻辑脉络;\n"
    "2) 用自己的话重述, 保留核心事实与数据, 但表达方式完全不同, 避免与原文连续 5 字以上雷同 (防搬运/原创度>80);\n"
    "3) 补足可读性: 小标题、金句、互动结尾; 口语化但不低质;\n"
    "4) 不要编造未提及的事实或数据。\n"
    "直接输出改写后的完整图文内容 (含小标题), 不要解释你的改写策略。"
)


def _provider_list(api_key: Optional[str]):
    """按优先级返回可用的 (base_url, key) 列表。"""
    providers = []
    # 1) OpenRouter (默认, 用户网页里预填的 key 也可由前端传来)
    or_key = api_key or os.environ.get("OPENROUTER_KEY")
    if or_key:
        providers.append(("https://openrouter.ai/api/v1", or_key))
    # 2) Groq (可选 env)
    if os.environ.get("GROQ_KEY"):
        providers.append(("https://api.groq.com/openai/v1", os.environ["GROQ_KEY"]))
    # 3) Gemini (可选 env)
    if os.environ.get("GEMINI_KEY"):
        providers.append(("https://generativelanguage.googleapis.com/v1beta/openai/", os.environ["GEMINI_KEY"]))
    if not providers:
        raise RuntimeError("no LLM provider configured")
    return providers


@app.post("/api/rewrite")
async def rewrite(payload: dict):
    text = (payload.get("text") or "").strip()
    if not text:
        raise HTTPException(400, "text required")
    model = payload.get("model") or os.environ.get("DEFAULT_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
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


# ----------------------------------------------------------------------------
# /api/image  —— AI 配图 (Pollinations 代理, 规避浏览器 CORS)
# ----------------------------------------------------------------------------
@app.post("/api/image")
async def image(payload: dict):
    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(400, "prompt required")
    import httpx
    url = "https://image.pollinations.ai/prompt/" + _encode(prompt) + "?width=1024&height=576&nologo=true"
    r = httpx.get(url, follow_redirects=True, timeout=60)
    if r.status_code != 200:
        raise HTTPException(502, "image gen failed")
    return Response(content=r.content, media_type="image/jpeg")


def _encode(s: str) -> str:
    import urllib.parse
    return urllib.parse.quote(s, safe="")


# ----------------------------------------------------------------------------
# 本地直接运行 (调试)
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
