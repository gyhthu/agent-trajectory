"""控制命令(!parallel qK)必须继承 **它 target instruction(qK) 的路由**，
而不是盲目继承当前活跃 session（廉莲 2026-07-03 纠偏）。

核心回归：enqueue 与 !parallel 之间插入了别的指令、last_sid 已漂走时，
`!parallel q1` 仍要落到 q1 那条指令被路由到的 session。旧的「继承 last_sid」
版本在这里会错挂——本测试就是钉死这个漂移场景。
"""
import route_day as R


class _FakeMsg:
    def __init__(self, c): self.content = c


class _FakeChoice:
    def __init__(self, c): self.message = _FakeMsg(c)


class _FakeResp:
    def __init__(self, c): self.choices = [_FakeChoice(c)]


class _FakeClient:
    """把带『新提案』的那条判成 new(新开)，其余不该走到模型。"""
    class chat:
        class completions:
            @staticmethod
            def create(**kw):
                um = kw["messages"][-1]["content"]
                sess = "new" if "新提案" in um else "1"
                return _FakeResp('{"session":"%s","reason":"x"}' % sess)


def _base_evs():
    return [
        {"msg_id": "m0", "role": "user", "name": "张耀明", "text": "S1 主线：先搞A"},
        {"msg_id": "m1", "role": "user", "name": "廉莲",
         "text": "新提案：原子轨迹切分不合理，换个切法", "parent_id": None},
        {"msg_id": "m2", "role": "assistant", "name": "bot",
         "text": "🕒 已入队 q1（原子轨迹切分方案）", "parent_id": "m1"},
        # 插一条主线续问：回 S1，把 last_sid 从 S2 漂回 S1
        {"msg_id": "m3", "role": "user", "name": "张耀明", "text": "A那个再改下", "parent_id": "m0"},
        {"msg_id": "m4", "role": "user", "name": "廉莲", "text": "!parallel q1", "parent_id": None},
    ]


def test_control_inherits_target_q_not_active_session():
    res = R.route_day(_base_evs(), respect_reply=True, client=_FakeClient())
    assert res["n_sessions"] == 2
    ctrl = [d for d in res["decisions"] if d["mode"] == "control"]
    assert len(ctrl) == 1
    d = ctrl[0]
    # q1 = m1 = S2；即便活跃 session 已漂到 S1，也必须落 S2
    assert d["session"] == 2, d
    assert d["target_q"] == 1
    assert d["target_src"] == "q-map"
    assert res["assign"]["4"] == 2


def test_control_falls_back_to_active_when_no_qmap():
    """无显式 qK 记录（如 !compact）→ 退回当前活跃 session，账目标 active-fallback。"""
    evs = _base_evs()[:2] + [
        {"msg_id": "m9", "role": "user", "name": "廉莲", "text": "!compact", "parent_id": None},
    ]
    res = R.route_day(evs, respect_reply=True, client=_FakeClient())
    ctrl = [d for d in res["decisions"] if d["mode"] == "control"][0]
    assert ctrl["target_q"] is None
    assert ctrl["target_src"] == "active-fallback"
    assert ctrl["session"] == 2  # 活跃就是刚开的 S2


if __name__ == "__main__":
    test_control_inherits_target_q_not_active_session()
    test_control_falls_back_to_active_when_no_qmap()
    print("ok")
