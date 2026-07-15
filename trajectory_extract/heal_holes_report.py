#!/usr/bin/env python3
"""自愈跑完后打印愈后 state 的洞情况（供 bgtask 卡回填看成效）。"""
import json
import incremental_segment as inc

STATE = "/opt/shared/data/task-trajectory/state/oc_53b8b620867a189d8dfe502865dfccc5.json"


def main():
    d = json.load(open(STATE, encoding="utf-8"))
    ft = d.get("frozen_tasks") or []
    holes = [t for t in ft if inc._decompose_failed(t)]
    print(f"愈后：冻结任务 {len(ft)} 个，剩余洞 {len(holes)} 个（愈前 9）")
    for t in ft:
        subs = t.get("subreqs") or []
        print(f"  {t.get('title', '')[:20]:<20} 子需求={len(subs):<3} decompose_ok={t.get('decompose_ok')}")
    tail_subs = len(d.get("active_tail_subreqs") or [])
    print(f"  [活动尾巴] 子需求={tail_subs}")


if __name__ == "__main__":
    main()
