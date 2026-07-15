"""共享 embedding 工具：本机 litellm 的 bge-m3（deepseek 不出本机）。

两处复用（DRY，别各写一份）：
  · 缝合护栏换轴（task_stitch._should_strong_stitch_by_reply 的内容地板）；
  · 候选召回（llm_segment 给 LLM 分组喂「疑似同任务」候选对）。

刻意做成「可不依赖网络」：embed 失败/未配时返回 None，调用方回退到原有确定性行为，
绝不让核心切分因为 embedding 服务挂了而崩。
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

_EMB_BASE = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:4000/v1")
_EMB_KEY = os.environ.get("LLM_API_KEY", "sk-litellm-master-key")
_EMB_MODEL = os.environ.get("EMB_MODEL", "bge-m3")
_MAX_CHARS = 2000


def embed_texts(texts: list[str]):
    """批量嵌入 → L2 归一化的 (n, d) ndarray；失败返回 None（调用方回退）。"""
    if not texts:
        return None
    try:
        from openai import OpenAI
        cli = OpenAI(base_url=_EMB_BASE, api_key=_EMB_KEY)
        out = []
        for i in range(0, len(texts), 32):
            r = cli.embeddings.create(model=_EMB_MODEL,
                                      input=[t[:_MAX_CHARS] or "(空)" for t in texts[i:i + 32]])
            out.extend(x.embedding for x in r.data)
        m = np.asarray(out, dtype=np.float32)
        m /= (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
        return m
    except Exception:
        return None


@lru_cache(maxsize=4096)
def _embed_one_cached(text: str):
    m = embed_texts([text])
    return None if m is None else m[0]


def cosine(a, b) -> float | None:
    if a is None or b is None:
        return None
    return float(np.dot(a, b))


def text_sim(t1: str, t2: str) -> float | None:
    """两段文本的余弦相似度（带缓存）；任一嵌入失败返回 None。"""
    a, b = _embed_one_cached(t1), _embed_one_cached(t2)
    return cosine(a, b)
