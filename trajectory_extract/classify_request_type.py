#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用户要求二分类器（张耀明 2026-07-07 定：只分前两类，弃「追问核对」类）。

分类轴 = 用户要求(instruction)本身要 bot 干什么，决定 replay 走哪条腿：
  ① 问答/说明类 : 要信息/解释/概念/调研/判断——纯文本回路可直接 replay。
  ② 执行/操作类 : 要 bot 去做/去跑/去产出（跑测试、切分、读盘核对、改文件等）——需沙箱才复现得出错。

弃用的「③ 追问核对(为什么错/你确定吗)」不是 replay 类：replay 还原的是 t0 前场景，
那一刻错还没发生，'为什么错' 属 t0 之后的纠正(anchor)，不是可 replay 的题面。
故 instruction 里若夹了回头追问，只按其**前瞻要求**(要不要 bot 去做)判 ①/②。

复用基座(不重造)：replay_three_leg._client / _chat（litellm 127.0.0.1:4000 + deepseek）。
铁律：只输出真跑出来的标签；解析不了就标 parse_error，不硬凑。
"""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replay_three_leg as R

DATA = "/opt/shared/data/task-trajectory"
SNAP = f"{DATA}/pre_instruction_snapshots.jsonl"
MODEL = os.environ.get("REPLAY_MODEL", "deepseek-v3.2")
N = int(os.environ.get("CLS_N", "0"))  # 0=全跑；>0=只跑前 N 条验证

_SYS = """你是任务分类器。判断【用户这条要求】要 bot 干什么，只输出三类之一：
- exec : 要求 bot 去【做/执行/操作】——跑脚本、跑测试、切分数据、改代码、产出具体结果或产物。
         ★读盘/读历史产物核对也算 exec(张耀明 2026-07-07 拍)：凡"完成得对"必须真动手(跑命令、读文件/git 核对、写文件)才谈得上的，都算 exec。哪怕最终只是把读到的内容讲出来，只要答对的前提是先去读盘核对，就归 exec。
         ★特别地——要 bot【调出/翻出它之前产出的具体产物(prompt、脚本、文档、数据)并评估其内容/覆盖/遗漏】=exec：答对必须先把那份旧产物从盘上/历史里读回来核，不是凭记忆能答准的。例："你之前构造的X是什么样的、有没有覆盖到Y、有没有遗漏"→exec。
- qa   : 要求 bot 给【信息/解释/概念/判断/调研】——回答一个问题、解释一个词、讨论设计、问业界做法。不需要真动手执行、凭已有知识/上下文就能答对的，算 qa。
- drop : 【非 replay 题面】——弃出 replay 集。两种：
         (a) 纯追问 bot **自己上一轮的行为/输出**、无任何前瞻交付：如"你刚为什么给错""你确定吗""你思考时为什么显示那段文本""为什么你动不动造出『20条』这种词"。这类是对已发生错误的元层质疑，replay 还原的是 t0 前场景、那刻错还没发生，故不是可复现题面。
         (b) 纯【中止/取消】信号：stop/停/别发了/够了/取消——只是叫 bot 停手，不产出任何东西。
             ★注意：触发动作、会产出结果的命令(如 !ctx 显示上下文路由、!parallel 并行跑、/run)**不是** drop，它们要 bot 去执行→exec。drop(b) 仅限"喊停"。

判据只看这条要求的【前瞻诉求】：它让 bot 接下来去做什么。
规则(按序判)：
1) 明示延迟执行——用户明说"先回答我/别急着做/你只用回答/先别动手"→ 一律 qa(哪怕后面提到要做什么)。
2) 通篇**只有**追问 bot 自身上一轮行为/为什么错、或纯控制信号、且无任何前瞻交付(不要 bot 去读/去跑/去产出) → drop。
3) 夹了回头追问但**同时**提出前瞻交付(要梳理/要跑/要给文本/要调出旧产物核对) → 忽略追问部分，按前瞻交付判 exec/qa。
辨析 drop(a) vs exec：区别在**有没有要 bot 交付一份需读盘/核对才做得出的东西**。"为什么你会显示X/为什么你造词"=只要一个对自身行为的解释→drop；"你之前构造的X是什么样、有没有覆盖/遗漏"=要交付一份对旧产物的核对评估→exec。

只输出一行 JSON：{"class":"exec"|"qa"|"drop","confidence":0.0-1.0,"reason":"≤25字"}"""


def classify(client, instruction):
    user = f"【用户要求】\n{instruction}\n\n输出 JSON。"
    raw = R._chat(client, MODEL, _SYS, user).strip()
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return {"class": "parse_error", "confidence": 0.0, "reason": raw[:40]}
    try:
        d = json.loads(m.group(0))
        c = d.get("class")
        if c not in ("exec", "qa", "drop"):
            return {"class": "parse_error", "confidence": 0.0, "reason": f"bad class {c}"}
        return {"class": c, "confidence": float(d.get("confidence", 0)), "reason": str(d.get("reason", ""))[:40]}
    except Exception as e:
        return {"class": "parse_error", "confidence": 0.0, "reason": f"{e}"[:40]}


def main():
    rows = [json.loads(l) for l in open(SNAP, encoding="utf-8") if l.strip()]
    if N:
        rows = rows[:N]
    client = R._client()
    out = []
    for i, r in enumerate(rows):
        instr = r.get("original_instruction", "")
        t0 = (r.get("t0") or {}).get("msg_id", "")
        res = classify(client, instr)
        res.update({"idx": i, "t0_msg_id": t0, "instruction": instr[:70]})
        out.append(res)
        print(f'{i:3d} [{res["class"]:5s} {res["confidence"]:.2f}] {instr[:42]!r} :: {res["reason"]}', flush=True)
    tag = f"_n{N}" if N else ""
    dst = f"{DATA}/request_type_2class{tag}.jsonl"
    with open(dst, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    import collections
    dist = dict(collections.Counter(o["class"] for o in out))
    print(f"\n== dist: {dist} == wrote {len(out)} -> {dst}")


if __name__ == "__main__":
    main()
