"""向量嵌入层（P1.2 可选第三路）：bge-small-zh-v1.5 @ fastembed/ONNX。

三路 RRF 的向量路：词法双轨对「口语 vs 报表术语」零词面重叠（归母
净资产 vs 归属于母公司所有者权益合计）无能为力，语义嵌入补位——
但消融实测（2026-10-19，31 题真值集）它在数字/口径精确任务上净负
贡献，故默认关闭：DZ_VEC=on 显式 opt-in（见 enabled 注释）。
可选依赖纪律：未装 fastembed / 未 opt-in / 库未嵌入（vecmeta 未
绑定或模型不符）→ ranking 返回 None，fts.search 退回双轨行为，
CI 无此依赖照常全绿。安装：pip install -r requirements-vec.txt。

容量注记（DESIGN-scale 同源哲学）：向量存 SQLite vecs 表（float32
blob，512 维 ≈ 2KB/块），查询时按 doc_scope SQL 过滤后内存 cosine
——万级块 ~16MB 数十 ms；百万级块需换 sqlite-vec/faiss（届时只动
本模块的 ranking 实现，融合协议不变）。
"""

from __future__ import annotations

import os

MODEL = "BAAI/bge-small-zh-v1.5"
DIM = 512
_model = None          # 模型单例（加载 ~18s，懒加载一次）


def installed() -> bool:
    """可选依赖是否在（CLI embed 建索引只看这个）。"""
    try:
        import fastembed  # noqa: F401
        return True
    except ImportError:
        return False


def enabled() -> bool:
    """查询路是否启用：显式 DZ_VEC=on（opt-in）且依赖在。

    默认关闭是消融定的（2026-10-19，31 题真值集）：财报数字/口径
    精确任务上词法双轨已饱和（answer_hit 1.00×3 数据集），向量第三路
    把语义相近但口径不同的块拉近、挤掉精确命中——0.875/0.833/0.926。
    语义问法场景（为什么/怎么样）再显式开启实验。
    """
    return os.environ.get("DZ_VEC") == "on" and installed()


def _get():
    global _model
    if _model is None:
        from fastembed import TextEmbedding
        _model = TextEmbedding(MODEL)
    return _model


def embed_blobs(texts: list[str]) -> list[bytes]:
    """批量嵌入 → float32 blob 列表（Store.add_vecs 直接可写）。"""
    import numpy as np
    vecs = list(_get().embed(texts))
    return [np.asarray(v, dtype=np.float32).tobytes() for v in vecs]


def _query_vec(query: str):
    import numpy as np
    v = next(iter(_get().embed([query])))
    v = np.asarray(v, dtype=np.float32)
    n = float((v * v).sum() ** 0.5)
    return v / (n or 1.0)


def ranking(con, docs: dict, query: str, doc_filter: str,
            doc_scope: set[str] | None, limit: int = 24) -> list[str] | None:
    """查询侧：embed query → doc 过滤后取库内向量 → cosine top。

    返回 chunk_id 相似度降序列表（三路 RRF 第三排名）；任何前置条件
    不满足返回 None（调用方退回双轨）。与 fts.search 同一套
    doc_filter/doc_scope 语义（复用 fts.allow_ids）。
    """
    if not enabled():
        return None
    row = con.execute("SELECT model FROM vecmeta LIMIT 1").fetchone()
    if not row or row[0] != MODEL:
        return None
    from duizhang import fts
    allow = fts.allow_ids(docs, doc_filter, doc_scope)
    if allow == []:
        return None
    sql = ("SELECT chunk_id, vec FROM vecs "
           + (f"WHERE doc_id IN ({','.join('?' * len(allow))})" if allow else ""))
    rows = con.execute(sql, allow or []).fetchall()
    if not rows:
        return None
    import numpy as np
    mat = np.frombuffer(b"".join(r[1] for r in rows),
                        dtype=np.float32).reshape(len(rows), DIM)
    norms = (mat * mat).sum(axis=1) ** 0.5
    norms[norms == 0] = 1.0
    sims = (mat / norms[:, None]) @ _query_vec(query)
    order = sorted(range(len(rows)), key=lambda i: -sims[i])[:limit]
    return [rows[i][0] for i in order]
