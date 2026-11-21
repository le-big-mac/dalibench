"""How a stage waits for its worker threads, and how it is asked to stop.

A stage (`attempt`, `judge`) hands work to thread pools and waits in the main
thread. Two things go wrong when it waits the plain way, with
`concurrent.futures.wait` and a signal handler that raises KeyboardInterrupt.

1. The main thread can be deaf to a signal. Python runs a handler in the main
   thread, between two instructions. The kernel may hand a signal to any thread
   of the process. When it picks a worker, the main thread is not woken, and
   one that sits in a wait with no time limit runs the handler only when that
   wait ends by itself: when a model call is answered, which can be an hour
   later. Seen as a stop signal answered a minute late on a loaded machine.

2. The main thread can be cut short in the middle of something. An interrupt
   raised by a handler arrives between any two instructions:
   - inside `ThreadPoolExecutor.submit`, after the work is queued and before
     the new thread is registered. Nothing then knows of that thread: the pool
     does not join it and nothing waits for it, so the command can say
     "stopped" and give up its lock on the run before that worker starts
     calling its model and writing files;
   - inside `concurrent.futures.wait`, while it holds the lock of a future.
     The lock is never given back, the worker can never record its result,
     and the stage waits for ever for a call that has long been answered;
   - between `submit` and the line that records the future, so that the count
     the stage prints is wrong and the future is not among those it waits for.
   A wait that polls with `concurrent.futures.wait` to cure 1 goes through
   that second window ten times a second.

So a stage does its waiting through a `Watch`:

- The wait holds nothing while it waits. Each future reports itself through a
  callback that adds it to a set, which is one indivisible step, and the main
  thread sleeps for a short time, again and again, and looks at the set. An
  interrupt during that sleep leaves nothing held, and a handler runs within
  `POLL_SECONDS` of its signal whichever thread the kernel gave it to. One
  lock is still touched, once for each future: the future's own, for the
  moment it takes to register the callback when the wait first sees it.
- A stop can be asked for instead of raised. A signal handler calls
  `ask_stop()`; while a `Watch` is entered that only sets a flag, and the
  stage raises KeyboardInterrupt itself at its next `checkpoint()`: at the top
  of its loop and in its wait, never inside `submit` and never with its
  bookkeeping half done. When no `Watch` is entered `ask_stop()` returns
  False and the handler raises as it always did.

A stop that is asked for is also seen by the workers (`Watch.stop_seen`): work
that was handed out in the same instant as the stop came, and has not begun,
is not begun.

What is not promised: safety under a handler that raises. Python's default
SIGINT handler does, and code that calls the stages as a library may have one.
The interrupt then arrives where it arrives. The sleep of the wait holds
nothing, but a few instructions do take a lock for a moment (a future's own,
when the wait first sees it; the pool's, inside `submit`), and an interrupt
raised exactly there can leave it held or leave a worker unknown, as in 2
above. An interrupt raised as a `Watch` is being left, while something still
refers to it, can also leave it entered. Those windows are in the standard
library's own code or in the `with` statement itself and are not closed here;
`ask_stop()` from the handler is the way that has none of them.
"""

from __future__ import annotations

import time
import weakref
from concurrent.futures import Future
from typing import Iterable

POLL_SECONDS = 0.1  # the longest the main thread sleeps between two looks: how late a signal handed to a worker is taken
FIRST_POLL_SECONDS = 0.001  # the first sleep of a wait; each next one is twice as long, up to POLL_SECONDS

# The watches entered and not yet left, innermost last, each held weakly: one whose `with` was cut short before it
# began (an interrupt raised between entering and the block) is referred to by nothing else, so it is gone, and a
# stop is never asked of a stage that is not there to take it.
_entered: list["weakref.ref[Watch]"] = []


def ask_stop() -> bool:
    """Ask the stage at work to stop at its next checkpoint. False if no stage is at work, and then nothing was asked.

    For a signal handler: it stores two flags and takes no lock.
    """
    while _entered:
        watch = _entered[-1]()
        if watch is None:
            _entered.pop()
            continue
        watch.stop_asked = watch.stop_seen = True
        return True
    return False


class Watch:
    """The futures a stage waits for, and the stop that was asked of it.

    Entered with `with` around the stage's loop. While it is entered, `ask_stop()` reaches it.
    Left with a stop asked for and never acted on, it raises KeyboardInterrupt (unless something
    else is already being raised): the stop is not lost because the work happened to end.
    """

    def __init__(self) -> None:
        self.stop_asked = False  # a stop was asked for and no checkpoint has raised it yet
        self.stop_seen = False  # a stop was asked for, at any time: never cleared, and read by the workers
        self._watched: set[Future] = set()  # the main thread's own
        # Added to by whichever thread finishes a future, in one indivisible step and under no lock of ours.
        # Nothing is moved from one container to another: a future is in here or it is not.
        self._finished: set[Future] = set()

    def __enter__(self) -> "Watch":
        _entered.append(weakref.ref(self))
        return self

    def __exit__(self, kind, value, traceback) -> None:
        _entered[:] = [ref for ref in _entered if ref() is not None and ref() is not self]
        if kind is None and self.stop_asked:
            self.stop_asked = False
            raise KeyboardInterrupt

    def checkpoint(self) -> None:
        """Raise KeyboardInterrupt here if a stop was asked for. Call it where nothing is half done."""
        if self.stop_asked:
            self.stop_asked = False
            raise KeyboardInterrupt

    def _look(self, running: Iterable[Future]) -> set[Future]:
        """Those of `running` that have finished, however they ended."""
        running = list(running)
        for future in running:
            if future not in self._watched:
                # The callback first: cut short between the two lines, the future is asked to report twice, which
                # does no harm; the other way round it would never report and be waited for for ever.
                future.add_done_callback(self._finished.add)  # called at once if it has finished already
                self._watched.add(future)
        return {future for future in running if future in self._finished}

    def first_finished(self, running: Iterable[Future]) -> set[Future]:
        """The futures of `running` that have finished, once at least one has. With none running, none are waited for.

        A checkpoint: a stop that was asked for is raised here, before anything is returned.
        """
        running, pause = list(running), FIRST_POLL_SECONDS
        while True:
            self.checkpoint()
            finished = self._look(running)
            if finished or not running:
                return finished
            time.sleep(pause)
            pause = min(POLL_SECONDS, pause * 2)

    def all_finished(self, running: Iterable[Future]) -> None:
        """Returns when every future of `running` has finished, however it ended. Not a checkpoint: this is the
        wait for work in flight after a stop, and a further stop asked for meanwhile does not end it."""
        running, pause = list(running), FIRST_POLL_SECONDS
        while len(self._look(running)) < len(set(running)):
            time.sleep(pause)
            pause = min(POLL_SECONDS, pause * 2)
