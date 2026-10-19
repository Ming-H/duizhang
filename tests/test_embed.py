"""embed 模块（P1.2 向量第三路，可选依赖）：CI 无 fastembed 自动跳过。"""

import pytest

pytest.importorskip("fastembed", reason="可选依赖未装（requirements-vec.txt）")


def test_embed_blobs_and_query_ranking(tmp_path, monkeypatch):
    monkeypatch.setenv("DZ_VEC", "on")      # 查询路 opt-in（默认关）
    from duizhang import embed
    from duizhang.schemas import Chunk, Doc
    from duizhang.store import Store
    assert embed.enabled()
    # 嵌入产出 float32 blob（512 维 × 4 字节）
    blobs = embed.embed_blobs(["营业收入 38,239,935,640.67 元",
                               "归属于母公司所有者权益合计"])
    assert len(blobs) == 2 and len(blobs[0]) == 512 * 4

    doc_id = "测试司_2025Q1"
    s = Store(tmp_path / "v.db")
    s.add_doc(Doc(doc_id, "测试司", "2025Q1", "一季报", "x", 1),
              [Chunk(f"{doc_id}#p1#0", doc_id, 1, 0,
                     "营业收入（元） 6,674,176,487.23"),
               Chunk(f"{doc_id}#p2#0", doc_id, 2, 0,
                     "经营活动产生的现金流量净额（元） 1,234,567.89")])
    # 未绑定模型/无向量 → ranking 为 None（双轨退化，不炸）
    assert embed.ranking(s.con, {doc_id: Doc(doc_id, "测试司", "2025Q1",
                                            "一季报", "x", 1)},
                         "营收多少", "", None) is None
    # 嵌入后：语义最近的块排第一
    s.set_vec_model(embed.MODEL)
    s.add_vecs([(f"{doc_id}#p1#0", doc_id, blobs[0]),
                (f"{doc_id}#p2#0", doc_id, blobs[1])])
    r = embed.ranking(s.con, {doc_id: Doc(doc_id, "测试司", "2025Q1",
                                          "一季报", "x", 1)},
                      "营收多少", "", None)
    assert r and r[0] == f"{doc_id}#p1#0"
    # doc_scope 限定生效：scope 排除后返回空
    r2 = embed.ranking(s.con, {doc_id: Doc(doc_id, "测试司", "2025Q1",
                                           "一季报", "x", 1)},
                       "营收多少", "", set())
    assert r2 is None
    s.close()


def test_vec_default_off_opt_in_on(tmp_path):
    """消融定默认（P1.2）：财报数字/口径题上词法双轨饱和 1.00，向量
    第三路 answer_hit 掉到 0.875/0.833/0.926——语义泛化把口径不同
    的块拉近挤掉精确命中。故默认关闭，DZ_VEC=on 显式 opt-in；
    installed()（依赖在否）与 enabled()（查询路开关）分离，
    CLI embed 只看前者（建索引不要求运行时开关）。"""
    import os

    from duizhang import embed
    from duizhang.store import Store
    s = Store(tmp_path / "v.db")
    s.set_vec_model(embed.MODEL)
    old = os.environ.get("DZ_VEC")
    try:
        os.environ.pop("DZ_VEC", None)
        assert embed.installed() is True      # 依赖在（本机装了 fastembed）
        assert embed.enabled() is False       # 默认关
        assert embed.ranking(s.con, {}, "x", "", None) is None
        os.environ["DZ_VEC"] = "on"
        assert embed.enabled() is True
    finally:
        if old is None:
            os.environ.pop("DZ_VEC", None)
        else:
            os.environ["DZ_VEC"] = old
    s.close()
