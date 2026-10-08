#!/usr/bin/env python3
"""Put one covenant memory whose body is read from a FILE.

Covenant's CLI takes the body as a command-line argument (main.py put --body). A JLens capture or an L-lens artifact
can be larger than one argument may be (32 KB on Windows, 128 KB per argument on Linux). This helper therefore makes
the same call main.py makes, MemoryStore(root).put(name, description, type, body, agent), using the same covenant
code, with the body read from a file.

Usage: put_memory_file.py --root ROOT --name NAME --description TEXT --type TYPE --agent AGENT --body-file PATH
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--description", required=True)
    ap.add_argument("--type", default="project")
    ap.add_argument("--agent", default="cli")
    ap.add_argument("--body-file", required=True)
    ap.add_argument("--covenant-src", default=os.environ.get("COVENANT_SRC", ""))
    a = ap.parse_args(argv)
    amd = os.path.join(a.covenant_src, "ai_memory_system")
    if not os.path.isfile(os.path.join(amd, "memory_store.py")):
        print("covenant memory_store.py not found under %r (set COVENANT_SRC)" % amd, file=sys.stderr)
        return 1
    sys.path.insert(0, amd)
    from memory_store import MemoryStore  # noqa: E402 -- covenant's own store, not a copy
    with open(a.body_file, encoding="utf-8") as f:
        body = f.read()
    out = MemoryStore(a.root).put(a.name, a.description, a.type, body, a.agent)
    print(json.dumps(out, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
