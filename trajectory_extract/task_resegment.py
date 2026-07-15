#!/usr/bin/env python3
"""离线任务重切器 —— 把 group-memory 里被实时启发式「粘连」成一坨的大任务，
按 话题标签 + 时间间隔 + 应答 bot 重新拆成若干子任务。

背景：实时看板对裸群聊（无回复边）只能都塞进主线 lane，导致一个 task_id 下
混了多个不相关子话题（如本群 #1）。本脚本不改线上数据，只读 group-memory
流，离线重判子任务边界，供轨迹串接做更细的分组。

切分信号（强→弱）：
  1) 时间大间隙：相邻消息 gap ≥ GAP_SEC（默认 30min）→ 硬边界（换摊了）。
  2) 话题漂移：topic_label 经窗口平滑后发生持续切换 → 边界。
     （单条 blip、〔闲聊〕、空标签不触发，避免抖动。）
用法：
  python task_resegment.py --group <chat_id> --task <task_id> [--gap 1800]
"""
import argparse, json, os, urllib.request
from collections import Counter

GM_URL = os.environ.get("GROUP_MEMORY_URL",
                        os.environ.get("TASKBOARD_URL", "http://127.0.0.1:8765")).rstrip("/")
NOISE_TOPICS = {"闲聊", "", "-"}


def fetch(group, n=600):
    url = f"{GM_URL}/message/recent?group_id={group}&n={n}"
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.load(r).get("results", [])


def rows_of_task(results, task_id):
    rows = []
    for r in results:
        md = r.get("metadata", {})
        if (md.get("task_id") or "") != task_id:
            continue
        rows.append({
            "ts": int(md.get("timestamp") or 0),
            "topic": md.get("topic_label") or "-",
            "bot": md.get("bot_id", "?"),
            "msg_id": md.get("msg_id", ""),
            "text": (r.get("text") or "").replace("\n", " ").strip(),
        })
    rows.sort(key=lambda x: x["ts"])
    return rows


def smooth_topic(rows):
    """把 noise 话题（闲聊/空）继承上一条的有效话题，得到每条的『有效话题』。"""
    last = "-"
    out = []
    for r in rows:
        t = r["topic"]
        if t in NOISE_TOPICS:
            t = last
        else:
            last = t
        out.append(t)
    return out


def verify_partition(rows, segs):
    """CHIEF(arXiv 2602.23701) §1.1 不变量：重切出的子任务段必须
    『连续 + 全覆盖 + 不重叠 + 保序』地复原原始流。

    规则切分（逐条恰好分配到一个段）理论上天然满足，但写成显式断言做护栏——
    将来若把切分换成 LLM 草拟 / 重构本函数，第一时间 fail-loud，绝不让
    『粘连(漏切)/丢段/重复』的脏分段静默流进下游训练集。
    （呼应团队原则·报错必须及时显示、禁止静默失败）

    校验项：
      1) 全覆盖+不重叠+保序：所有段的 rows 按段序拼回，必须与原始 rows 逐一同序同对象。
      2) 无空段。
      3) 段内时间单调不减；段间起始时间不倒退。
    断言失败抛 AssertionError（带可定位的差异信息）。
    """
    flat = [r for s in segs for r in s["rows"]]
    orig_ids = [id(r) for r in rows]
    flat_ids = [id(r) for r in flat]
    if flat_ids != orig_ids:
        miss = len(rows) - len(flat)
        dup = len(flat) - len(set(flat_ids))
        raise AssertionError(
            "[重切校验失败·连续/全覆盖/不重叠] 子任务段未能复原原始流："
            f"原始 {len(rows)} 条 / 切出 {len(flat)} 条（缺 {miss}、重 {dup}），"
            f"段数={len(segs)}。绝不放行到下游训练集。")
    for k, s in enumerate(segs):
        if not s["rows"]:
            raise AssertionError(f"[重切校验失败·空段] 第 {k} 段无任何消息。")
    last_start = None
    for k, s in enumerate(segs):
        ts = [r["ts"] for r in s["rows"]]
        for i in range(1, len(ts)):
            if ts[i] < ts[i - 1]:
                raise AssertionError(
                    f"[重切校验失败·段内时间倒序] 第 {k} 段第 {i} 条 ts={ts[i]} < 前条 {ts[i-1]}。")
        if last_start is not None and ts[0] < last_start:
            raise AssertionError(
                f"[重切校验失败·段序错乱] 第 {k} 段起始 ts={ts[0]} 早于上一段起始 {last_start}。")
        last_start = ts[0]
    return True


def segment(rows, gap_sec):
    eff = smooth_topic(rows)
    segs, cur = [], None
    for i, r in enumerate(rows):
        gap = (r["ts"] - rows[i - 1]["ts"]) if i else 0
        boundary = False
        if cur is None:
            boundary = True
        elif gap >= gap_sec:
            boundary = True  # 大间隙：硬边界
        elif eff[i] != cur["topic"] and eff[i] not in NOISE_TOPICS:
            # 话题持续切换才算（下一条仍是新话题，或本条非末条且新话题已坐实）
            nxt = eff[i + 1] if i + 1 < len(eff) else eff[i]
            if nxt == eff[i]:
                boundary = True
        if boundary:
            cur = {"topic": eff[i], "rows": [], "start": r["ts"]}
            segs.append(cur)
        else:
            cur["topic"] = eff[i]  # 跟随有效话题
        cur["rows"].append(r)
    verify_partition(rows, segs)  # CHIEF §1.1：切完即校验回原始流，fail-loud
    return segs


def label(seg):
    topics = Counter(r["topic"] for r in seg["rows"] if r["topic"] not in NOISE_TOPICS)
    bots = sorted({r["bot"][:8] for r in seg["rows"]})
    dom = topics.most_common(1)[0][0] if topics else "(无)"
    first = next((r["text"] for r in seg["rows"] if r["topic"] not in NOISE_TOPICS),
                 seg["rows"][0]["text"])
    return dom, bots, first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--gap", type=int, default=1800)
    args = ap.parse_args()
    rows = rows_of_task(fetch(args.group), args.task)
    segs = segment(rows, args.gap)
    import datetime as dt
    print(f"原任务 {args.task}：{len(rows)} 条 → 离线重切出 {len(segs)} 个子任务"
          f"（gap≥{args.gap//60}min 或话题持续切换）\n")
    for k, s in enumerate(segs, 1):
        dom, bots, first = label(s)
        t0 = dt.datetime.fromtimestamp(s["rows"][0]["ts"]).strftime("%m-%d %H:%M")
        t1 = dt.datetime.fromtimestamp(s["rows"][-1]["ts"]).strftime("%m-%d %H:%M")
        tdist = Counter(r["topic"] for r in s["rows"] if r["topic"] not in NOISE_TOPICS)
        print(f"【子任务 {k}】{dom}  | {len(s['rows'])}条 | {t0}→{t1} | bot×{len(bots)}")
        print(f"    话题分布: {dict(tdist)}")
        print(f"    首条: {first[:64]}")


if __name__ == "__main__":
    main()
