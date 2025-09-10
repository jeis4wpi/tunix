from __future__ import annotations
import asyncio
from typing import Any, Callable, Dict, Hashable, List, Optional, Tuple, Type

from trajectory_collect_engine import TrajectoryCollectEngine, LLMBaseAgent, BaseEnv, Trajectory
from group_queue_manager import GroupQueueManager, TrajectoryItem

class RolloutOrchestrator:
   def __init__(
       self,
       *,
       engine_cls: Type[TrajectoryCollectEngine] = TrajectoryCollectEngine,
       engine_defaults: Optional[Dict[str, Any]] = None,
       max_concurrency: Optional[int] = None,
   ):
       self.engine_cls = engine_cls
       self.engine_defaults = engine_defaults or {}
       self.max_concurrency = max_concurrency

       self._tasks: List[asyncio.Task] = []
       self._stop = asyncio.Event()

   async def _runner(
       self,
       i: int,
       agent: LLMBaseAgent,
       env: BaseEnv,
       manager: GroupQueueManager,
       group_key: Callable[[int, BaseEnv, Trajectory], Hashable],
       episodes_per_pair: Optional[int],
       start_step_fn: Optional[Callable[[], int]] = None,
   ):
       sem = asyncio.Semaphore(self.max_concurrency or len([0]))
       episode_id = 0
       while not self._stop.is_set() and (episodes_per_pair is None or episode_id < episodes_per_pair):
           async with sem:
               eng = self.engine_cls(agent, env, **self.engine_defaults)
               traj = await eng.collect()
           gid = group_key(i, env, traj)
           start_step = start_step_fn() if start_step_fn else 0
           item = TrajectoryItem(
               pair_index=i,
               group_id=gid,
               episode_id=episode_id,
               start_step=start_step,
               traj=traj,
               meta={}
           )
           await manager.put(item)
           episode_id += 1

   async def run_and_yield_batches(
       self,
       pairs: List[Tuple[LLMBaseAgent, BaseEnv]],
       *,
       group_size: int,
       batch_size: int,
       group_key: Callable[[int, BaseEnv, Trajectory], Hashable],
       episodes_per_pair: Optional[int] = None,
       max_open_groups: Optional[int] = None,
       start_step_fn: Optional[Callable[[], int]] = None,
   ):
       manager = GroupQueueManager(group_size=group_size, max_open_groups=max_open_groups)

       try:
           for i, (a, e) in enumerate(pairs):
               self._tasks.append(asyncio.create_task(
                   self._runner(i, a, e, manager, group_key, episodes_per_pair, start_step_fn)
               ))

           while not self._stop.is_set():
               batch = await manager.get_batch(batch_size)
               if batch:
                   yield batch
               all_done = all(t.done() for t in self._tasks)
               if all_done:
                   tail = await manager.get_batch(1_000_000)
                   if tail:
                       yield tail
                   break
       finally:
           self._stop.set()
           await manager.prepare_clear()
           await manager.clear()
           for t in self._tasks:
               if not t.done():
                   t.cancel()
           self._tasks.clear()