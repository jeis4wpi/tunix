from __future__ import annotations
import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Hashable, List, Optional, Tuple

from trajectory_collect_engine import Trajectory

@dataclass
class TrajectoryItem:
    pair_index: int
    group_id: Hashable
    episode_id: int
    start_step: int
    traj: Trajectory
    meta: Dict[str, Any] = field(default_factory=dict)

class GroupQueueManager:
    def __init__(
        self,
        *,
        group_size: int,
        max_open_groups: Optional[int] = None,
    ):
        self.group_size = group_size
        self.max_open_groups = max_open_groups

        self._buckets: Dict[Hashable, List[TrajectoryItem]] = defaultdict(list)
        self._ready_groups: Deque[List[TrajectoryItem]] = deque()

        self._group_events: Dict[Hashable, asyncio.Event] = defaultdict(asyncio.Event)
        self._have_ready = asyncio.Event()

        self._open_groups: int = 0
        self._open_group_keys: set = set()
        self._drain_waiters: Deque[asyncio.Future] = deque()

        self._clearing = False
        self._exc: Optional[BaseException] = None
        self._lock = asyncio.Lock()

        self._batch_buf: List[TrajectoryItem] = []

    def put_exception(self, exc: BaseException):
        self._exc = exc
        self._have_ready.set()
        for ev in self._group_events.values():
            ev.set()
        while self._drain_waiters:
            fut = self._drain_waiters.popleft()
            if not fut.done(): fut.set_result(None)

    async def prepare_clear(self):
        self._clearing = True
        self._have_ready.set()
        for ev in self._group_events.values():
            ev.set()
        while self._drain_waiters:
            fut = self._drain_waiters.popleft()
            if not fut.done(): fut.set_result(None)

    async def clear(self):
        async with self._lock:
            self._buckets.clear()
            self._ready_groups.clear()
            self._group_events.clear()
            self._open_groups = 0
            self._open_group_keys.clear()
            self._batch_buf.clear()
            self._exc = None
            self._clearing = False
            self._have_ready = asyncio.Event()

    async def put(self, item: TrajectoryItem):
        if self._clearing:
            return
        if self._exc:
            raise self._exc

        async with self._lock:
            key = item.group_id
            if key not in self._open_group_keys:
                if self.max_open_groups is not None and self._open_groups >= self.max_open_groups:
                    fut = asyncio.get_event_loop().create_future()
                    self._drain_waiters.append(fut)
                    self._lock.release()
                    try:
                        await fut
                    finally:
                        await self._lock.acquire()
                self._open_group_keys.add(key)
                self._open_groups += 1

            bucket = self._buckets[key]
            bucket.append(item)
            if len(bucket) == self.group_size:
                self._ready_groups.append(bucket.copy())
                self._buckets.pop(key, None)
                self._open_group_keys.discard(key)
                self._open_groups = max(0, self._open_groups - 1)
                self._have_ready.set()
                if key in self._group_events:
                    self._group_events[key].set()

    async def _get_one_ready_group(self) -> List[TrajectoryItem]:
        while not self._ready_groups:
            if self._exc:
                raise self._exc
            if self._clearing:
                return []
            await self._have_ready.wait()
            self._have_ready.clear()
        group = self._ready_groups.popleft()
        if self._drain_waiters:
            fut = self._drain_waiters.popleft()
            if not fut.done(): fut.set_result(None)
        return group

    async def get_batch(self, batch_size: int) -> List[TrajectoryItem]:
        while len(self._batch_buf) < batch_size:
            group = await self._get_one_ready_group()
            if not group:
                break
            self._batch_buf.extend(group)

        out = self._batch_buf[:batch_size]
        self._batch_buf = self._batch_buf[batch_size:]
        return out