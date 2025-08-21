# D:\Dev\kha\api\state\memory_trace_store.py
from __future__ import annotations
from collections import defaultdict, deque
from typing import Deque, Dict, List, Tuple
import time

class TraceStore:
    def __init__(self, maxlen: int = 1024):
        self._buf: Dict[str, Deque[Tuple[float,float,float,int]]] = defaultdict(lambda: deque(maxlen=maxlen))
        # tuple: (t, E, C, Lver)

    def record(self, user_id: str, E: float, C: float, Lver: int) -> None:
        self._buf[user_id].append((time.time(), float(E), float(C), int(Lver)))

    def last_n(self, user_id: str, n: int = 200) -> List[Tuple[float,float,float,int]]:
        b = self._buf.get(user_id)
        if not b: return []
        return list(b)[-n:]

    def stats(self, user_id: str, window: int = 50) -> dict:
        arr = self.last_n(user_id, window)
        if not arr: return {}
        ts, Es, Cs, Ls = zip(*arr)
        dE = [Es[i+1] - Es[i] for i in range(len(Es)-1)]
        dC = [abs(Cs[i+1] - Cs[i]) for i in range(len(Cs)-1)]
        inc_streak = 0
        for x in reversed(dE):
            if x > 0: inc_streak += 1
            else: break
        return {
            "n": len(arr),
            "E_last": Es[-1],
            "C_last": Cs[-1],
            "Lver_last": Ls[-1],
            "max_abs_dC": max(dC) if dC else 0.0,
            "inc_streak": inc_streak,
        }

TRACE = TraceStore(maxlen=2048)
