"""语料扫描与增量 diff（manifest 指纹模式）。

业界对齐：Airbyte/LlamaIndex 目录连接器的 path→fingerprint 账本。
指纹用 mtime_ns+size（轻量）；内容级身份由文件名解析出的 doc_id 保证。
"""

from __future__ import annotations

from pathlib import Path

ALLOWED_SUFFIXES = {".pdf", ".md", ".txt"}
DEFAULT_EXCLUDE = {"0 分析结果", "_index", ".git", "venv", "__pycache__", "node_modules"}


def scan(root: str | Path, exclude: set[str] = DEFAULT_EXCLUDE) -> dict[str, str]:
    """递归扫描 → {path: fingerprint}。跳过排除目录与数据工件（csv/png/html）。"""
    out: dict[str, str] = {}
    for p in Path(root).rglob("*"):
        if not p.is_file():
            continue
        rel_parts = p.relative_to(root).parts[:-1]  # 目录部分
        if any(seg in exclude for seg in rel_parts):
            continue
        if p.suffix.lower() not in ALLOWED_SUFFIXES or p.name == ".DS_Store":
            continue
        st = p.stat()
        out[str(p)] = f"{st.st_mtime_ns}-{st.st_size}"
    return out


def diff(manifest: dict[str, str], scanned: dict[str, str]):
    """→ (to_add, to_update, to_delete)。"""
    to_add = [p for p in scanned if p not in manifest]
    to_update = [p for p, fp in scanned.items()
                 if p in manifest and manifest[p] != fp]
    to_delete = [p for p in manifest if p not in scanned]
    return to_add, to_update, to_delete
