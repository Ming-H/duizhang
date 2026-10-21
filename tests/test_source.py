"""source.py 与增量摄取的测试：diff 三态 / 排除 / 幂等 / 删除落库。"""

import sqlite3

from tally.source import diff, scan


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
    import os
    import subprocess
    import sys
    env = {**os.environ, "TALLY_DB": str(work / "test.db")}
    r = subprocess.run(
        [sys.executable, "-m", "tally.cli", "ingest", str(corpus_dir)],
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
    _touch(seed / "欧科亿_备注.md", "欧科亿 业绩预告 预计净利润 36,000 万元")

    out1 = _ingest_once(repo, work, seed)
    assert "+1 新增" in out1 or "+2 新增" in out1

    # 幂等：指纹未变 → 无变化
    out2 = _ingest_once(repo, work, seed)
    assert "无变化" in out2

    # 修改：指纹变 → 更新
    _touch(seed / "欧科亿_备注.md", "欧科亿 业绩预告 预计净利润 42,000 万元 变更")
    out3 = _ingest_once(repo, work, seed)
    assert "~1 更新" in out3

    # 删除：文件没了 → 出库
    (seed / "欧科亿_备注.md").unlink()
    out4 = _ingest_once(repo, work, seed)
    assert "-1 删除" in out4


def test_ingest_second_root_keeps_first_root_docs(tmp_path):
    """多根摄取：ingest 目录 B 不能把目录 A 摄取的文档当孤儿清掉。
    （2026-10-14 实测：ingest data/corpus 静默清光了 samples/ 的文档。）"""
    import shutil
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    work = tmp_path / "work"
    work.mkdir()
    root_a, root_b = work / "a", work / "b"
    root_a.mkdir()
    root_b.mkdir()
    shutil.copy(repo / "samples" / "欧科亿_2026年半年度业绩预告.pdf", root_a)
    shutil.copy(repo / "samples" / "海伦哲_2026年第一季度报告.pdf", root_b)

    _ingest_once(repo, work, root_a)
    _ingest_once(repo, work, root_b)      # 现状：这一步会把 A 的文档当孤儿删掉
    _ingest_once(repo, work, root_a)      # 回到 A：幂等，B 的文档必须仍在

    con = sqlite3.connect(work / "test.db")
    docs = {r[0] for r in con.execute("SELECT doc_id FROM docs")}
    con.close()
    assert docs == {"欧科亿_2026H1E", "海伦哲_2026Q1"}


def test_scan_fingerprint_is_content_hash(tmp_path):
    """指纹=内容哈希而非 mtime+size（v0.7 正确性批）：保 mtime 改内容
    （cp -p / rsync -a 场景）必须检出；touch（内容未变）不得假更新。
    修前 mtime_ns+size 对前者漏检——语料静默过期，检索答旧数。"""
    import os

    from tally import source
    f = tmp_path / "a公司_2025年年度报告.md"
    f.write_text("营收 100 万")
    fp1 = source.scan(tmp_path)[str(f)]
    # 内容不变 + touch → 指纹不变（哈希只看内容，不假更新）
    os.utime(f, ns=(f.stat().st_atime_ns + 10**9, f.stat().st_mtime_ns + 10**9))
    assert source.scan(tmp_path)[str(f)] == fp1
    # 内容变 + mtime 恢复旧值 → 指纹必须变（rsync -a 漏检场景）
    old_mt = f.stat().st_mtime_ns
    f.write_text("营收 200 万")
    os.utime(f, ns=(old_mt, old_mt))
    assert source.scan(tmp_path)[str(f)] != fp1
