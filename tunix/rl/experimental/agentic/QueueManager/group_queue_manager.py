from __future__ import annotations
import asyncio
from collections import deque
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
        max_open_buckets: Optional[int] = None,
    ):
        self.group_size = group_size
        self.max_open_buckets = max_open_buckets or 0
        self._buckets: Dict[Tuple[Hashable, int], List[TrajectoryItem]] = {}
        self._ready_groups: Deque[List[TrajectoryItem]] = deque()
        self._clearing = False
        self._exc: Optional[BaseException] = None
        self._lock = asyncio.Lock()
        self._capacity = asyncio.Condition(self._lock)
        self._have_ready = asyncio.Event()
        self._batch_buf: List[TrajectoryItem] = []

    def put_exception(self, exc: BaseException):
        self._exc = exc
        self._have_ready.set()
        async def _notify_all():
            async with self._capacity:
                self._capacity.notify_all()
        asyncio.create_task(_notify_all())

    async def prepare_clear(self):
        self._clearing = True
        self._have_ready.set()
        async with self._capacity:
            self._capacity.notify_all()

    async def clear(self):
        async with self._lock:
            self._buckets.clear()
            self._ready_groups.clear()
            self._batch_buf.clear()
            self._exc = None
            self._clearing = False
            self._have_ready = asyncio.Event()

    async def put(self, item: TrajectoryItem):
        if self._clearing:
            return
        if self._exc:
            raise self._exc
        key = (item.group_id, item.episode_id)
        async with self._capacity:
            new_bucket = key not in self._buckets
            while (not self._clearing) and (self.max_open_buckets > 0) and new_bucket and (self._open_bucket_count() >= self.max_open_buckets):
                await self._capacity.wait()
            if self._clearing:
                return
            if self._exc:
                raise self._exc
            bucket = self._buckets.setdefault(key, [])
            bucket.append(item)
            if len(bucket) == self.group_size:
                self._ready_groups.append(bucket.copy())
                del self._buckets[key]
                self._capacity.notify_all()
                self._have_ready.set()

    async def _get_one_ready_group(self) -> List[TrajectoryItem]:
        while True:
            if self._exc:
                raise self._exc
            if self._clearing:
                return []
            if self._ready_groups:
                return self._ready_groups.popleft()
            await self._have_ready.wait()
            self._have_ready.clear()

    async def get_batch(self, batch_size: int) -> List[TrajectoryItem]:
        out = []
        if self._batch_buf:
            take = min(batch_size, len(self._batch_buf))
            out.extend(self._batch_buf[:take])
            self._batch_buf = self._batch_buf[take:]
            if len(out) == batch_size:
                return out
        while len(out) < batch_size:
            group = await self._get_one_ready_group()
            if not group:
                break
            room = batch_size - len(out)
            if len(group) <= room:
                out.extend(group)
            else:
                out.extend(group[:room])
                self._batch_buf.extend(group[room:])
        return out

    def _open_bucket_count(self) -> int:
        return len(self._buckets)
