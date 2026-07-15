#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把飞书 doc-viewer 会撑框的行内 markdown 构造转成安全形式。

飞书那个 viewer 把每个行内反引号代码 `x` 撑成整行灰底块、把裸尖括号
<x> 当 HTML 标签吞掉。修法（已在 07-06 总账等 10 篇验证 0/0/0 安全）：
  - 保护 ``` 多行代码块围栏，原样不动（那本就该是块）；
  - 其余行内反引号 span `x` → 去反引号变纯文本（标识符里的下划线按
    CommonMark 是词内下划线、不触发斜体，安全）；
  - span 内的尖括号一并转义 &lt; &gt;（防被当标签吞）。

只对显式传入的、已验证安全的交付散文档跑；含下划线开头 Python 方法名或
块内嵌反引号的文档会被这个转换帮倒忙，不要盲目铺开。
"""
import re
import shutil
import sys
from pathlib import Path


def convert(text: str) -> str:
    parts = re.split(r"(```.*?```)", text, flags=re.S)
    out = []
    for i, p in enumerate(parts):
        if i % 2 == 1:  # ``` 围栏块，原样保留
            out.append(p)
            continue

        def _repl(m: re.Match) -> str:
            inner = m.group(1)
            return inner.replace("<", "&lt;").replace(">", "&gt;")

        out.append(re.sub(r"`([^`\n]+)`", _repl, p))
    return "".join(out)


def audit(text: str) -> dict:
    # 复核：正文(剔围栏)里应无行内反引号、无裸尖括号
    body = re.sub(r"```.*?```", "", text, flags=re.S)
    return {
        "inline_backtick": len(re.findall(r"`[^`\n]+`", body)),
        "bare_angle": len(re.findall(r"<[A-Za-z/][^>]*>", body)),
    }


def main(paths: list[str]) -> int:
    bak_dir = None
    for path in paths:
        p = Path(path)
        src = p.read_text(encoding="utf-8")
        if bak_dir is None:
            bak_dir = p.parent / ".render_bak"
            bak_dir.mkdir(exist_ok=True)
        shutil.copy2(p, bak_dir / (p.name + ".bak"))
        conv = convert(src)
        p.write_text(conv, encoding="utf-8")
        a = audit(conv)
        flag = "" if a["inline_backtick"] == 0 and a["bare_angle"] == 0 else "  ⚠️FLAG"
        print(f"backtick={a['inline_backtick']:3d} angle={a['bare_angle']:3d}{flag}  {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
