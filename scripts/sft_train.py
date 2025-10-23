#!/usr/bin/env python3
"""
Thin shim so that legacy tooling can execute `python -m scripts.sft_train`
while the canonical implementation lives in `scripts.chat_sft`.
"""

from runpy import run_module


if __name__ == "__main__":
    run_module("scripts.chat_sft", run_name="__main__")
