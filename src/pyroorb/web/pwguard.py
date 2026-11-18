"""Wrong-password lockout for the server's password-gated routes.

A caller gets ``per_client`` wrong passwords in ``window_sec``; past that it is refused
without the password being compared, until its oldest failure leaves the window. That
bounds one caller's guessing, but a caller who can present many addresses gets
``per_client`` guesses at each, so every caller TOGETHER is also held to ``everyone``
failures per window. That backstop is the one place a distributed guesser can lock the
real password holder out for a window; it is set high enough that a single caller,
held to ``per_client``, cannot reach it alone.

The table is bounded: a failure older than the window is forgotten, and past
``max_clients`` callers the least recently failing are dropped, so a stream of new
addresses costs a fixed amount of memory, never more.
"""
import threading
import time
from collections import deque

#: wrong passwords one caller may make per window
PER_CLIENT = 5
#: wrong passwords every caller together may make per window
EVERYONE = 100
#: the window, seconds
WINDOW_SEC = 600.0
#: callers tracked at most
MAX_CLIENTS = 4096


class FailureGuard:
    """Counts wrong passwords per caller and in total; ``refused`` says when to stop
    comparing. Thread-safe."""

    def __init__(self, per_client=PER_CLIENT, everyone=EVERYONE, window_sec=WINDOW_SEC,
                 max_clients=MAX_CLIENTS, clock=time.time):
        self.per_client = int(per_client)
        self.everyone = int(everyone)
        self.window_sec = float(window_sec)
        self.max_clients = int(max_clients)
        self._clock = clock
        self._lock = threading.Lock()
        self._fails = {}            # client -> deque of failure times, oldest first
        self._all = deque()         # every failure time, oldest first

    def _prune(self, now):
        cut = now - self.window_sec
        while self._all and self._all[0] <= cut:
            self._all.popleft()
        for k in [k for k, q in self._fails.items() if not q or q[-1] <= cut]:
            del self._fails[k]

    def refused(self, client):
        """None when ``client`` may try a password now, else the reason it may not."""
        now = self._clock()
        with self._lock:
            self._prune(now)
            q = self._fails.get(client)
            if q is not None:
                while q and q[0] <= now - self.window_sec:
                    q.popleft()
                if len(q) >= self.per_client:
                    return "too many wrong passwords from this address; try again in a few minutes"
            if len(self._all) >= self.everyone:
                return "too many wrong passwords on this server; try again in a few minutes"
        return None

    def failed(self, client):
        """Record one wrong password from ``client``."""
        now = self._clock()
        with self._lock:
            self._prune(now)
            self._fails.setdefault(client, deque()).append(now)
            self._all.append(now)
            if len(self._fails) > self.max_clients:
                stale = sorted(self._fails, key=lambda k: self._fails[k][-1])
                for k in stale[: len(self._fails) - self.max_clients]:
                    if k != client:
                        del self._fails[k]

    def succeeded(self, client):
        """Forget ``client``'s failures (the server-wide count keeps them)."""
        with self._lock:
            self._fails.pop(client, None)

    def reset(self):
        """Tests only."""
        with self._lock:
            self._fails.clear()
            self._all.clear()
