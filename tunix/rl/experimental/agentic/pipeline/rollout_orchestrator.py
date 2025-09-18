from __future__ import annotations
import asyncio
import logging
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
       self._semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

       self._tasks: List[asyncio.Task] = []
       self._stop = asyncio.Event()
       self._logger = logging.getLogger(self.__class__.__name__)

   async def _collect_trajectory(self, agent: LLMBaseAgent, env: BaseEnv) -> Trajectory:
       """Helper method to collect a single trajectory."""
       engine = self.engine_cls(agent, env, **self.engine_defaults)
       return await engine.collect()

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
       episode_id = 0
       self._logger.debug(f"Starting runner for pair {i}")

       try:
           while not self._stop.is_set() and (episodes_per_pair is None or episode_id < episodes_per_pair):
               try:
                   if self._semaphore:
                       async with self._semaphore:
                           traj = await self._collect_trajectory(agent, env)
                   else:
                       traj = await self._collect_trajectory(agent, env)

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

               except Exception as e:
                   self._logger.error(f"Error collecting trajectory for pair {i}, episode {episode_id}: {e}")
                   # Continue with next episode instead of crashing the entire runner
                   episode_id += 1
                   continue

       except Exception as e:
           self._logger.error(f"Fatal error in runner for pair {i}: {e}")
           raise
       finally:
           self._logger.debug(f"Runner for pair {i} completed with {episode_id} episodes")

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
       manager = GroupQueueManager(group_size=group_size, max_open_buckets=max_open_groups)

       try:
           for i, (a, e) in enumerate(pairs):
               self._tasks.append(asyncio.create_task(
                   self._runner(i, a, e, manager, group_key, episodes_per_pair, start_step_fn)
               ))

           while not self._stop.is_set():
               batch = await manager.get_batch(batch_size)
               if batch:
                   yield batch

               # Check if all tasks are done with small delay to handle race conditions
               all_done = all(t.done() for t in self._tasks)
               if all_done:
                   # Small delay to ensure any final items are processed
                   await asyncio.sleep(0.01)

                   # Collect remaining items in smaller chunks to avoid memory issues
                   while True:
                       remaining = await manager.get_batch(batch_size)
                       if not remaining:
                           break
                       yield remaining
                   break
       finally:
           self._stop.set()
           self._logger.debug("Stopping orchestrator and cleaning up resources")

           # Cancel all running tasks
           for t in self._tasks:
               if not t.done():
                   t.cancel()

           # Wait for all tasks to complete or be cancelled
           if self._tasks:
               await asyncio.gather(*self._tasks, return_exceptions=True)

           # Clean up manager
           await manager.prepare_clear()
           await manager.clear()
           self._tasks.clear()

           self._logger.debug("Cleanup completed")