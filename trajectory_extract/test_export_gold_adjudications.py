import export_gold_adjudications as E
import pytest


def _ev(msg_id, text, role="user", name="张耀明"):
    return {"msg_id": msg_id, "text": text, "role": role, "name": name}


def test_export_gold_from_review_maps_adjudicated_l_labels_to_msg_ids():
    evs = [
        _ev("t0", "!parallel 业界有没有做这种轨迹拆分", name="廉莲"),
        _ev(
            "b",
            "这正好对应我们在做的事（task_stitch / task_resegment / 回复链重切）。我并行开几路调研。",
            role="bot",
            name="claude(lian-server)",
        ),
        _ev(
            "c",
            "你说的子需求边界是什么；以及在训练的时候，并不是以子需求为单位作为训练数据啊，而是以任务为单位作为训练数据啊",
        ),
    ]
    review = """# review
## #11　来源:run3_survive
- **L8** 〔真人·廉莲〕　!parallel 业界有没有做这种轨迹拆分
- L9 〔bot·claude(lian-server)〕　这正好对应我们在做的事（task_stitch / task_resegment / 回复链重切）。我并行开几路调研。
- **L11** 〔真人·张耀明〕　你说的子需求边界是什么；以及在训练的时候，并不是以子需求为单位作为训练数据啊，而是以任务为单位作为训练数据啊
"""
    adjudication = """### #11

- 人判：合格候选。
- 真 C：L11
- 真 B：L9
- 真 t0：L8
"""

    rows = E.export_gold_from_review(review, adjudication, evs)

    assert rows == [{
        "card_no": 11,
        "status": "ok",
        "missing": [],
        "labels": {"c": "L11", "b": "L9", "t0": "L8"},
        "gold": {
            "t0_msg_id": "t0",
            "bot_error_msg_id": "b",
            "corrector_msg_id": "c",
        },
        "note": "- 人判：合格候选。 - 真 C：L11 - 真 B：L9 - 真 t0：L8",
        "adjudication_source_msg_id": "",
    }]


def test_export_gold_resolves_duplicate_lines_by_card_context():
    evs = [
        _ev("old_t0", "重复的长问题"),
        _ev("old_b", "旧回答", role="bot", name="claude(lian-server)"),
        _ev("t0", "重复的长问题"),
        _ev("b", "这句是错的", role="bot", name="claude(lian-server)"),
        _ev("c", "这里不对"),
    ]
    review = """# review
## #2　来源:redflag_137
- **L4** 〔真人·张耀明〕　重复的长问题
- L41 〔bot·claude(lian-server)〕　这句是错的
- **L42** 〔真人·张耀明〕　这里不对
"""
    adjudication = """### #2

- 人判：合格候选。
- 真 t0：L4
- 真 B：L41
- 真 C：L42
"""

    rows = E.export_gold_from_review(review, adjudication, evs)

    assert rows[0]["status"] == "ok"
    assert rows[0]["gold"] == {
        "t0_msg_id": "t0",
        "bot_error_msg_id": "b",
        "corrector_msg_id": "c",
    }


def test_adjudication_source_msg_id_is_required_for_cli_exports():
    text = """# 人工裁决

人工裁决来源：群消息 msg_id=om_x100b6bd79a1af8b0c425ae09348be64，发送者张耀明

### #4

- 人判：合格候选。
- 真 t0：L4
- 真 B：L6
- 真 C：L7
"""

    assert E.require_adjudication_source_msg_id(text) == "om_x100b6bd79a1af8b0c425ae09348be64"
    with pytest.raises(ValueError, match="lacks a human source msg_id"):
        E.require_adjudication_source_msg_id("### #4\n- 人判：合格候选。")
