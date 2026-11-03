"""Peak resident memory of a search, sampled -- the instrument needed before
anyone argues about the size of the sweep floor.

``ItfConfig.sweep_min_free_mb`` is 1000 because ONE object (2006 SJ252) died at
~550 MB free under a 500 MB floor, so the floor was doubled.  ``collect.py``
carries 1500 for the same question.  Neither number has ever been measured
against what a search actually costs, because nothing anywhere records that:
no ``ru_maxrss``, no sampler.  This is the sampler.

Why a THREAD and not ``resource.getrusage``:

  * ``ru_maxrss`` is a high-water mark for the whole PROCESS and never falls.
    In the web server every search shares one process, so after the first big
    search the figure is that search's, forever, attributed to whoever asks
    next.  Sampling gives a per-search window instead: the peak BETWEEN start
    and stop, and the delta over the level the process was already holding.
  * the number that decides anything is not RSS but AVAILABLE memory (the
    sweep floor's own currency), and only a sampler catches how close a search
    came to the floor WITHOUT tripping it -- which is the difference between
    "the floor is generous" and "the floor was one window away all along".

Everything degrades to None rather than raising: on a kernel without
``/proc/self/status`` the search runs exactly as it did before.
"""

import collections
import os
import threading
import time

from .memory import available_mb, breakdown


def rss_mb():
    """This process's resident set in MB, or None where it cannot be read."""
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


class RssSampler:
    """Sample RSS and available memory on a background thread; report the peak
    RSS and the LOW-WATER available figure over the sampled window.

    Cheap by construction: two small /proc reads every ``interval`` seconds
    (the default 0.5 s costs microseconds against a search measured in
    minutes), and the thread is a daemon so it can never hold a process open.
    """

    def __init__(self, interval=0.5, max_seconds=7200.0,
                 burst_window=10.0):
        self.interval = float(interval)
        # A hard self-stop.  The search has ONE return path, so a sampler is
        # normally stopped exactly once -- but an exception escaping the search
        # would leave this thread with no one to stop it, and in the web server
        # that is a per-search leak rather than a process that is about to
        # exit.  The deadline bounds a leaked sampler instead of demanding a
        # try/finally around 270 lines of search body.
        self.max_seconds = float(max_seconds)
        self._deadline = None
        self.start_rss = None
        self.peak_rss = None
        self.min_avail = None
        self.samples = 0
        # The BURST window.  This is the number the sweep floor actually needs,
        # and it is not the same as what a search costs in total: the floor is
        # re-checked every window, so what it must survive is however much the
        # process can allocate BETWEEN two checks -- one chunk's rows, or the
        # LOV profile pass that runs in whatever the sweep left behind (the exact
        # way 2006 SJ252 died: sweep ends ~550 MB free, churn eats the rest in
        # seconds).  A floor below the burst cannot save the process however
        # generous it looks; a floor far above it truncates for nothing.
        self.burst_window = float(burst_window)
        self.max_burst = None
        self._recent = collections.deque()      # (monotonic, rss) within window
        self._stop = threading.Event()
        self._thread = None
        self._stopped = False

    def _sample(self):
        r = rss_mb()
        if r is not None:
            self.peak_rss = r if self.peak_rss is None else max(self.peak_rss, r)
            now = time.monotonic()
            self._recent.append((now, r))
            cutoff = now - self.burst_window
            while len(self._recent) > 1 and self._recent[0][0] < cutoff:
                self._recent.popleft()
            rise = r - min(v for _t, v in self._recent)
            self.max_burst = rise if self.max_burst is None \
                else max(self.max_burst, rise)
        a = available_mb()
        if a is not None:
            self.min_avail = a if self.min_avail is None else min(self.min_avail, a)
        self.samples += 1

    def start(self):
        self.start_rss = rss_mb()
        self._sample()
        if self.start_rss is None:
            return self                 # nothing to sample: stay inert
        self._deadline = time.monotonic() + self.max_seconds
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="itf-rss-sampler")
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.wait(self.interval):
            if time.monotonic() > self._deadline:
                return                  # abandoned: stop rather than sample forever
            try:
                self._sample()
            except Exception:           # never let the instrument kill a search
                return

    def stop(self):
        """Stop sampling and return the report dict.  Idempotent: a second call
        returns the SAME report rather than taking one more sample, so a caller
        that stops defensively cannot move the numbers it is about to log."""
        if self._stopped:
            return self.report()
        self._stopped = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._sample()
        return self.report()

    def report(self):
        """``{}`` when nothing could be measured, else the search's memory
        profile.  ``peak_rss_mb`` is the process high-water DURING the window;
        ``rss_growth_mb`` is that minus what the process was already holding,
        which is the figure a per-search admission decision would need."""
        if self.peak_rss is None:
            return {}
        out = {"peak_rss_mb": self.peak_rss, "samples": self.samples}
        if self.start_rss is not None:
            out["start_rss_mb"] = self.start_rss
            out["rss_growth_mb"] = max(0, self.peak_rss - self.start_rss)
        if self.min_avail is not None:
            out["min_available_mb"] = self.min_avail
        if self.max_burst is not None:
            out["max_burst_mb"] = self.max_burst
            out["burst_window_s"] = self.burst_window
        cg = breakdown()
        if cg:
            # the active_file term is the open question: the cgroup side of
            # available_mb() charges it as spent, the host's MemAvailable would
            # count about half of it as free
            out["cgroup"] = cg
        return out

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False
