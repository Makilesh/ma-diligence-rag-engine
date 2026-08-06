"""
Sliding window rate limiter per model.

Thread-safe for concurrent async requests. Uses asyncio.Lock()
which MUST be created inside a running event loop — NOT at class
definition / import time.

IMPORTANT: The lock is released BEFORE sleeping so other coroutines
can proceed. After waking, the lock is re-acquired and timestamps
are re-purged to handle changes made while sleeping.
"""

import asyncio
import time
from collections import deque

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class RateLimiter:
    """
    Sliding window rate limiter per model.
    Thread-safe for concurrent async requests.

    IMPORTANT: The lock is released BEFORE sleeping so other coroutines
    can proceed. After waking, the lock is re-acquired and timestamps
    are re-purged to handle changes made while sleeping.

    Args:
        rpm_limit: Maximum requests per minute for this model.
    """

    def __init__(self, rpm_limit: int) -> None:
        """
        Initialize rate limiter.

        Args:
            rpm_limit: Maximum requests per minute.

        Raises:
            ValueError: If rpm_limit is not positive.
        """
        if rpm_limit <= 0:
            raise ValueError(f"rpm_limit must be positive, got {rpm_limit}")
        self.rpm_limit = rpm_limit
        self._call_timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """
        Wait until a call slot is available within the current 60-second window.

        This method blocks the calling coroutine until a rate limit slot opens.
        The lock is released during sleep to allow other coroutines to proceed.

        Raises:
            RuntimeError: If called outside an async context.
        """
        while True:
            async with self._lock:
                now = time.monotonic()
                # Purge timestamps older than 60 seconds
                while self._call_timestamps and self._call_timestamps[0] < now - 60:
                    self._call_timestamps.popleft()

                if len(self._call_timestamps) < self.rpm_limit:
                    # Slot available — record and return
                    self._call_timestamps.append(time.monotonic())
                    return

                # No slot available — calculate wait time
                wait_time = 60.0 - (now - self._call_timestamps[0])

            # CRITICAL: Lock is RELEASED here before sleeping.
            # Without this, asyncio.sleep() holds the lock for up to 60s,
            # serializing ALL concurrent requests behind the sleeping coroutine.
            if wait_time > 0:
                logger.info(
                    "Rate limiter waiting for slot",
                    extra={
                        "rpm_limit": self.rpm_limit,
                        "wait_seconds": round(wait_time, 2),
                    },
                )
                await asyncio.sleep(wait_time)
            # Loop back: re-acquire lock, re-purge timestamps, re-check slot

    async def try_acquire(self) -> bool:
        """
        Take a slot if one is free right now; never sleeps.

        This is the primitive multi-key rotation needs. `acquire()` waits for the
        window to open, which is correct when there is only one place to send the
        request — but with several API keys, blocking on a saturated key while
        another key sits idle wastes the very capacity the extra keys were added
        for. The selector calls this across every (key, model) pair first and
        only falls back to the blocking path when none can serve immediately.

        Returns:
            True if a slot was taken, False if the window is currently full.
        """
        async with self._lock:
            now = time.monotonic()
            while self._call_timestamps and self._call_timestamps[0] < now - 60:
                self._call_timestamps.popleft()

            if len(self._call_timestamps) < self.rpm_limit:
                self._call_timestamps.append(time.monotonic())
                return True
            return False

    @property
    def current_usage(self) -> int:
        """
        Returns the number of calls made in the current 60-second window.
        Non-blocking, approximate count (no lock acquired).

        Returns:
            Number of active timestamps in the sliding window.
        """
        now = time.monotonic()
        return sum(1 for ts in self._call_timestamps if ts >= now - 60)
