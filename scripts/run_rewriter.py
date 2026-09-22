#!/usr/bin/env python3
"""
run_rewriter.py — Interactive Query Rewriter Tester
====================================================
এই script টা run করলে তুমি নিজে query দিতে পারবে
এবং expand_query() এর real output দেখতে পারবে।

চালানোর নিয়ম:
  cd shibir-chat-back-end-dev
  python scripts/run_rewriter.py

.env ফাইলে OPENAI_API_KEY থাকতে হবে।
"""

from __future__ import annotations

import sys
import os
import json
import time

# ── project root টা sys.path এ add করো ──────────────────────────────────────
# এই script shibir-chat-back-end-dev/ এর ভেতর থেকে run হবে
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ── .env load করো ────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

# ── Import ────────────────────────────────────────────────────────────────────
from app.rag.query_rewriter import expand_query, detect_language, _reset_fallback_count, get_fallback_count

# ── Color codes (terminal) ────────────────────────────────────────────────────
class C:
    HEADER  = "\033[95m"
    BLUE    = "\033[94m"
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    BOLD    = "\033[1m"
    RESET   = "\033[0m"

ROUTE_COLOR = {
    "BENGALI":        C.GREEN,
    "BANGLISH":       C.YELLOW,
    "ENGLISH_ARABIC": C.CYAN,
}

ROUTE_LABEL = {
    "BENGALI":        "🟢 BENGALI        → Direct use + synonym expansion",
    "BANGLISH":       "🟡 BANGLISH       → Transliterate to Bengali script",
    "ENGLISH_ARABIC": "🔵 ENGLISH/ARABIC → Translate to Bengali + keep original",
}


def print_banner():
    print(f"""
{C.BOLD}{C.HEADER}╔══════════════════════════════════════════════════════╗
║         Query Rewriter — Interactive Tester          ║
║   Bengali / Banglish / English / Arabic supported    ║
╚══════════════════════════════════════════════════════╝{C.RESET}

Commands:
  {C.BOLD}query টাইপ করো{C.RESET}  → output দেখো
  {C.BOLD}q{C.RESET} বা {C.BOLD}exit{C.RESET}      → বের হও
  {C.BOLD}clear{C.RESET}          → screen clear
""")


def print_result(query: str, route: str, result: tuple[str, ...], elapsed_ms: float):
    color = ROUTE_COLOR.get(route, C.RESET)
    label = ROUTE_LABEL.get(route, route)

    print(f"\n{C.BOLD}{'─' * 54}{C.RESET}")
    print(f"  Input   : {C.BOLD}{query}{C.RESET}")
    print(f"  Route   : {color}{label}{C.RESET}")
    print(f"  Time    : {elapsed_ms:.0f}ms")
    print(f"{C.BOLD}{'─' * 54}{C.RESET}")

    if not result:
        print(f"  {C.RED}⚠ কোনো result আসেনি (fallback হয়েছে){C.RESET}")
        return

    print(f"  {C.BOLD}Search strings ({len(result)}টি):{C.RESET}")
    for i, q in enumerate(result):
        prefix = f"  [{i}]"
        if i == 0:
            print(f"{C.BOLD}{prefix} {C.GREEN}{q}{C.RESET}  ← canonical Bengali (bn)")
        elif i == len(result) - 1 and route != "BENGALI":
            print(f"{prefix} {C.CYAN}{q}{C.RESET}  ← original query")
        else:
            print(f"{prefix} {q}")
    print()


def run():
    print_banner()

    session_count = 0

    while True:
        try:
            query = input(f"{C.BOLD}Query > {C.RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{C.YELLOW}বের হচ্ছি...{C.RESET}")
            break

        if not query:
            continue

        if query.lower() in ("q", "exit", "quit", "বের", "বাহির"):
            print(f"{C.YELLOW}বের হচ্ছি...{C.RESET}")
            break

        if query.lower() == "clear":
            os.system("clear" if os.name != "nt" else "cls")
            print_banner()
            continue

        # ── Detect route ──────────────────────────────────────────────────────
        route = detect_language(query)

        # ── expand_query() call ───────────────────────────────────────────────
        _reset_fallback_count()
        expand_query.cache_clear()   # fresh call হর বার, cache skip

        print(f"\n  {C.YELLOW}Processing...{C.RESET}", end="\r")

        start = time.perf_counter()
        try:
            result = expand_query(query)
        except Exception as e:
            print(f"  {C.RED}Error: {e}{C.RESET}")
            continue
        elapsed_ms = (time.perf_counter() - start) * 1000

        # ── fallback হয়েছে? ──────────────────────────────────────────────────
        if get_fallback_count() > 0:
            print(f"  {C.RED}⚠ LLM call fail করেছে — raw query দিয়ে fallback হয়েছে{C.RESET}")

        print_result(query, route, result, elapsed_ms)
        session_count += 1


if __name__ == "__main__":
    run()
