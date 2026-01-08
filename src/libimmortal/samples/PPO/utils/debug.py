import os, glob
from typing import Optional, Tuple, Dict, Any

import libimmortal.samples.PPO.utils.ddp as ddp

# -------------------------
# Debug helpers (Player.log)
# -------------------------

def _find_latest_player_log() -> Optional[str]:
    # common default location on Linux
    pats = [
        "/root/.config/unity3d/DefaultCompany/**/Player.log",
        os.path.expanduser("~/.config/unity3d/DefaultCompany/**/Player.log"),
    ]
    cands = []
    for p in pats:
        cands.extend(glob.glob(p, recursive=True))
    if not cands:
        return None
    cands.sort(key=os.path.getmtime, reverse=True)
    return cands[0]


def _tail_file(path: str, n: int = 200) -> str:
    try:
        with open(path, "rb") as f:
            data = f.read()
        lines = data.splitlines()[-n:]
        return "\n".join([ln.decode("utf-8", errors="replace") for ln in lines])
    except Exception as e:
        return f"[tail failed] {repr(e)}"


def _print_player_log_tail(prefix: str, n: int = 200):
    if not ddp.is_main_process():
        return
    pl = _find_latest_player_log()
    if pl is None:
        print(f"{prefix} no Player.log found under DefaultCompany/**/Player.log")
        return
    print(f"{prefix} Player.log: {pl}")
    print(f"{prefix} --- tail -n {n} ---")
    print(_tail_file(pl, n=n))
    print(f"{prefix} --- end tail ---")