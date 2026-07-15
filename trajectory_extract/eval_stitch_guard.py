"""缝合护栏消融：旧『时间轴(24h+异话题)』 vs 新『内容轴(bge-m3 地板)』。

张耀明 2026-06-29 第4条硬要求：一定要给修改前后的消融对照。这也是给「该不该按回复边强缝合」
这个决策**补一个以前没有的人工锚定小测试集**（#4：现状没有好测试集评判切分好坏，先从这块补起）。

每条样例由人工定死正确决策（该缝合 True / 该降级 False），bge-m3 又是确定性的，所以
「旧 guard / 新 guard 各判对几条」是无歧义、可复现的。4 类覆盖时间轴的两个洞 + 两个对照：
  ① 同天挂错(异话题、内容对不上)         期望降级  ← 旧时间轴漏网(gap<24h 强缝)，新轴要拦住
  ② 同天真续接(异话题标签、内容接得上)   期望缝合  ← 旧新都该缝（对照，别误伤）
  ③ 跨天真续接(标签漂移、内容接得上)     期望缝合  ← 旧时间轴误切(gap≥24h+异话题降级)，新轴要救回
  ④ 跨天挂错(异话题、内容对不上)         期望降级  ← 旧新都该降级（对照）

样例为合成（脱敏后的真实群词汇，已标 synthetic），不含任何真实 secret/open_id。
跑：python3 eval_stitch_guard.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import task_stitch as ts

DAY = 86400
T0 = 1_750_000_000


def seg(topic, *texts, roles=None):
    rows = []
    for i, t in enumerate(texts):
        role = (roles[i] if roles else ("bot" if i % 2 == 0 else "user"))
        rows.append({"ts": T0 + i, "role": role, "text": t, "msg_id": f"{topic}{i}",
                     "parent_id": "", "name": "x", "kind": None, "who": "x"})
    return {"topic": topic, "rows": rows}


# 每条: (类型, 期望强缝合?, 同天?, 子段(cur_seg=子消息真正所属), 父段(parent_id误指/真指), 子消息正文)
# 不变量(严格)：挂错→子消息内容属 cur_seg、与 parent_seg **不同主题**(期望低相似→降级)；
#               续接→子消息内容**延续 parent_seg 主题**(期望高相似→缝合)。
CASES = [
    # ① 同天挂错：子消息属 cur_seg(litellm/网络/算法)，parent_seg 是无关的轨迹切分/训练 → 应降级
    ("①同天挂错", False, True,
     seg("服务器运维", "litellm 容器重启了一下", "等会看 model_info"),
     seg("数据处理清洗", "轨迹切分的 terminal 口径改成最近被碰那条", "原子段宁过切勿过并"),
     "litellm 重启完 model 列表恢复了吗 SALT_KEY 那个 db 还报错没"),
    ("①同天挂错", False, True,
     seg("每日一题", "背包 dp 二维降一维", "倒序遍历"),
     seg("训练评测", "sft 的 loss 曲线还在抖动得厉害", "学习率 warmup 再调一版重跑"),
     "背包那题一维优化 dp 数组要倒序遍历才不会重复选对吧"),
    ("①同天挂错", False, True,
     seg("bot基建", "飞书卡片标签提到顶部高亮", "改 send.sh 渲染那段"),
     seg("数据处理清洗", "增量切分水位线按消息 id 存", "封存老任务只重切活动尾巴"),
     "飞书那个卡片标签高亮的 send.sh 改完了吗 mention 还正常不"),

    # ② 同天真续接：子消息延续 parent_seg 主题 → 应缝合（对照，别误伤）
    ("②同天真续接", True, True,
     seg("数据处理清洗", "原子段缝合那块先放一放", "等下回头看"),
     seg("服务器运维", "先把 litellm 的 SALT_KEY 固定到 env", "compose 补传后清死行重注册"),
     "接着 litellm 那个 SALT_KEY，compose 里我加上重注册了 model_info 稳定到 2 了"),
    ("②同天真续接", True, True,
     seg("bot基建", "换个事先说卡片渲染", "稍后再聊"),
     seg("数据处理清洗", "挂错回复会把两个原子段焊死", "缝合换轴改用内容相似度地板"),
     "接着缝合换轴，内容地板定多少能把挂错那条和真续接分开"),

    # ③ 跨天真续接(标签漂移)：子消息延续 parent_seg → 应缝合；旧时间轴会误切
    ("③跨天真续接", True, False,
     seg("服务器运维", "今天先聊部署", "明天继续"),
     seg("数据处理清洗", "bge-m3 候选召回先做离线验证", "在两份金标上算多召回哪些跨断档同任务对"),
     "昨天那个 bge-m3 候选召回的离线验证，跨断档同任务对召回 57% 噪声才 1 个"),
    ("③跨天真续接", True, False,
     seg("训练评测", "先看别的", "回头说"),
     seg("数据处理清洗", "未确认这个状态专指 bot 失约、请求悬空", "delivery 仅占位也归未确认"),
     "接着未确认那个状态，我把 delivery 仅占位的也并进未确认一起判了"),

    # ④ 跨天挂错：子消息属 cur_seg、与 parent_seg 无关 → 应降级（对照，旧新都该降级）
    ("④跨天挂错", False, False,
     seg("服务器运维", "ssh 跳板 hk 配 ProxyJump", "德国机出口干净免 vpn"),
     seg("数据处理清洗", "轨迹切分原子段口径", "话题持续切换才断单条 blip 不切"),
     "ssh 那个 ProxyJump 到德国机配好了吗 LocalForward 6080 通不通"),
    ("④跨天挂错", False, False,
     seg("每日一题", "动态规划状态转移方程", "区间 dp 怎么枚举断点"),
     seg("bot基建", "飞书事件门控 mentions 才算真 @", "补 include_bot 权限再发版"),
     "区间 dp 那道题枚举断点的复杂度是 O(n^3) 对吧记忆化怎么写"),
]


def run():
    sim_fn = ts.build_reply_sim_fn()
    if sim_fn is None:
        raise SystemExit("bge-m3 不可用，消融要真 embedding")

    print(f"内容地板 _REPLY_CONTENT_FLOOR = {ts._REPLY_CONTENT_FLOOR}\n")
    print(f"{'类型':<12} {'期望':<6} {'sim':<6} {'旧时间轴':<10} {'新内容轴':<10}")
    print("-" * 56)
    old_ok = new_ok = 0
    rows = []
    for typ, expect, same_day, child_seg, parent_seg, child_text in CASES:
        gap = 1800 if same_day else 2 * DAY
        child_ev = {"ts": T0 + (parent_seg["rows"][-1]["ts"] - T0) + gap,
                    "text": child_text, "role": "user", "msg_id": "child"}
        parent_ev = {"ts": parent_seg["rows"][-1]["ts"], "text": parent_seg["rows"][-1]["text"]}
        # 旧轴：sim_fn=None → 纯时间
        old = ts._should_strong_stitch_by_reply(child_seg, parent_seg, child_ev, parent_ev, None)
        # 新轴：真 bge-m3
        new = ts._should_strong_stitch_by_reply(child_seg, parent_seg, child_ev, parent_ev, sim_fn)
        sim = sim_fn(child_ev, parent_seg)
        old_correct = (old == expect)
        new_correct = (new == expect)
        old_ok += old_correct
        new_ok += new_correct
        rows.append((typ, sim))
        ev = "缝合" if expect else "降级"
        os_ = ("缝合" if old else "降级") + ("✓" if old_correct else "✗")
        ns_ = ("缝合" if new else "降级") + ("✓" if new_correct else "✗")
        print(f"{typ:<12} {ev:<6} {sim:<6.3f} {os_:<11} {ns_:<11}")

    n = len(CASES)
    print("-" * 56)
    print(f"{'判对':<12} {'':<6} {'':<6} {old_ok}/{n}={old_ok/n:.0%}    {new_ok}/{n}={new_ok/n:.0%}")

    # 标定参考：真续接 sim 分布 vs 挂错 sim 分布，看地板卡在哪
    cont = sorted(s for (t, s) in rows if t.startswith(("②", "③")))
    miss = sorted(s for (t, s) in rows if t.startswith(("①", "④")))
    print(f"\n标定参考：真续接 sim {[f'{x:.2f}' for x in cont]}（应在地板之上）")
    print(f"          挂错  sim {[f'{x:.2f}' for x in miss]}（应在地板之下）")
    if cont and miss:
        print(f"          可分间隔：挂错最高 {max(miss):.2f} ↔ 真续接最低 {min(cont):.2f}"
              f" → 地板 {ts._REPLY_CONTENT_FLOOR} 落在区间内 = {'可分' if max(miss) < ts._REPLY_CONTENT_FLOOR <= min(cont) else '需重标'}")


if __name__ == "__main__":
    run()
