"""语料扫描与增量 diff（manifest 指纹模式）。

业界对齐：Airbyte/LlamaIndex 目录连接器的 path→fingerprint 账本。
指纹 = SHA-256 内容哈希（v0.7 正确性批；修前 mtime_ns+size：cp -p /
rsync -a 保 mtime 改内容会漏检——语料静默过期，检索答旧数；touch 会
假更新）。成本：双语料 36MB 全量哈希 ~0.1s 级；老 manifest 指纹格式
不同 → 升级后首次 ingest 全量重摄取一次（幂等，之后回到增量）。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

ALLOWED_SUFFIXES = {".pdf", ".md", ".txt"}
DEFAULT_EXCLUDE = {"0 分析结果", "_index", ".git", "venv", "__pycache__", "node_modules"}


def _fingerprint(p: Path) -> str:
    """内容哈希指纹（流式读，1MB 块）——只看内容，与 mtime/ctime 无关。"""
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return f"sha256:{h.hexdigest()[:24]}"


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
        out[str(p)] = _fingerprint(p)
    return out


def diff(manifest: dict[str, str], scanned: dict[str, str]):
    """→ (to_add, to_update, to_delete)。"""
    to_add = [p for p in scanned if p not in manifest]
    to_update = [p for p, fp in scanned.items()
                 if p in manifest and manifest[p] != fp]
    to_delete = [p for p in manifest if p not in scanned]
    return to_add, to_update, to_delete
