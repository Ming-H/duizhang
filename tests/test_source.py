"""source.py 与增量摄取的测试：diff 三态 / 排除 / 幂等 / 删除落库。"""

import sqlite3

import pytest

from duizhang.source import DEFAULT_EXCLUDE, diff, scan


def _touch(path, content="x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_scan_excludes_analysis_dir(tmp_path):
    _touch(tmp_path / "0 分析结果" / "a.md")
    _touch(tmp_path / "电子" / "公司X" / "公司X_2025年年度报告.md", "正文内容足够长")
    scanned = scan(tmp_path)
    assert len(scanned) == 1
    assert "分析结果" not in list(scanned)[0]


def test_scan_suffix_whitelist(tmp_path):
    _touch(tmp_path / "ok.txt", "文本")
    _touch(tmp_path / "data.csv", "a,b")
    _touch(tmp_path / "pic.png")
    scanned = scan(tmp_path)
    assert list(scanned) == [str(tmp_path / "ok.txt")]


def test_diff_three_way():
    manifest = {"keep": "fp1", "gone": "fpX", "changed": "fp2"}
    scanned = {"keep": "fp1", "changed": "fpNEW", "new": "fp3"}
    add, upd, dele = diff(manifest, scanned)
    assert add == ["new"] and upd == ["changed"] and dele == ["gone"]


def _ingest_once(repo_root, work, corpus_dir):
    import os, subprocess, sys
    env = {**os.environ,
           "DZ_DB": str(work / "test.db"), "DZ_IDX": str(work / "test.idx")}
    r = subprocess.run(
        [sys.executable, "-m", "duizhang.cli", "ingest", str(corpus_dir)],
        cwd=repo_root, capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-400:]
    return r.stdout


def test_incremental_add_modify_delete(tmp_path):
    """端到端：新增→幂等→修改→删除，全走 manifest。"""
    import shutil
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    work = tmp_path / "work"
    work.mkdir()
    # 用仓库真样例做种子，避免依赖大语料
    seed = work / "corpus"
    seed.mkdir()
    shutil.copy(repo / "samples" / "欧科亿_2026年半年度业绩预告.pdf", seed)
    _touch(seed / "notes.md", "欧科亿 业绩预告 预计净利润 36,000 万元")

    out1 = _ingest_once(repo, work, seed)
    assert "+1 新增" in out1 or "+2 新增" in out1

    # 幂等：指纹未变 → 无变化
    out2 = _ingest_once(repo, work, seed)
    assert "无变化" in out2

    # 修改：指纹变 → 更新
    _touch(seed / "notes.md", "欧科亿 业绩预告 预计净利润 42,000 万元 变更")
    out3 = _ingest_once(repo, work, seed)
    assert "~1 更新" in out3

    # 删除：文件没了 → 出库
    (seed / "notes.md").unlink()
    out4 = _ingest_once(repo, work, seed)
    assert "-1 删除" in out4
