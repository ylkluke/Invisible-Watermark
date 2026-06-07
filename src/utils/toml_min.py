from __future__ import annotations

import re
from pathlib import Path
from typing import Any


_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?(?:\d+\.\d*|\d*\.\d+)(?:[eE][+-]?\d+)?$")


def _strip_comment(line: str) -> str:
    in_squote = False
    in_dquote = False
    esc = False
    out: list[str] = []
    for ch in line:
        if esc:
            out.append(ch)
            esc = False
            continue
        if ch == "\\" and in_dquote:
            out.append(ch)
            esc = True
            continue
        if ch == "'" and not in_dquote:
            in_squote = not in_squote
            out.append(ch)
            continue
        if ch == '"' and not in_squote:
            in_dquote = not in_dquote
            out.append(ch)
            continue
        if ch == "#" and not in_squote and not in_dquote:
            break
        out.append(ch)
    return "".join(out).strip()


def _parse_value(raw: str) -> Any:
    s = raw.strip()
    if not s:
        raise ValueError("empty value")

    if s == "true":
        return True
    if s == "false":
        return False

    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        body = s[1:-1]
        if s.startswith('"'):
            body = (
                body.replace("\\n", "\n")
                .replace("\\t", "\t")
                .replace('\\"', '"')
                .replace("\\\\", "\\")
            )
        return body

    if _INT_RE.match(s):
        return int(s, 10)
    if _FLOAT_RE.match(s):
        return float(s)

    raise ValueError(f"unsupported TOML value: {raw!r}")


def load_toml(path: str | Path) -> dict[str, Any]:
    """
    Minimal TOML loader for this repo.

    Supported:
      - Tables: [section] and [a.b]
      - Scalars: strings, int, float, bool
      - Comments with '#'

    Not supported:
      - Arrays, inline tables, multi-line strings, dates, etc.
    """
    p = Path(path)
    data: dict[str, Any] = {}
    cur: dict[str, Any] = data

    for lineno, raw_line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = _strip_comment(raw_line)
        if not line:
            continue

        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            if not section:
                raise ValueError(f"{p}:{lineno}: empty section header")
            cur = data
            for part in section.split("."):
                part = part.strip()
                if not part:
                    raise ValueError(f"{p}:{lineno}: invalid section header: {section!r}")
                nxt = cur.get(part)
                if nxt is None:
                    nxt = {}
                    cur[part] = nxt
                if not isinstance(nxt, dict):
                    raise ValueError(f"{p}:{lineno}: section conflicts with value: {part!r}")
                cur = nxt
            continue

        if "=" not in line:
            raise ValueError(f"{p}:{lineno}: expected key = value")
        k, v = line.split("=", 1)
        key = k.strip()
        if not key:
            raise ValueError(f"{p}:{lineno}: empty key")
        cur[key] = _parse_value(v)

    return data

