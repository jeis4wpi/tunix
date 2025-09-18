import asyncio
import logging
import time
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple
from tunix.rl.experimental.agentic.agents import base_agent
from tunix.rl.experimental.agentic.environments import base_environment
from tunix.rl.experimental.agentic.utils import convert_messages_to_tokens_and_masks

BaseEnv = base_environment.BaseEnv
Trajectory = base_agent.Trajectory
LLMBaseAgent = base_agent.LLMBaseAgent
logger = logging.getLogger(__name__)


class TrajectoryCollectEngine:
  """Asynchronous trajectory collection engine for agent-environment interactions.

  This engine orchestrates complete rollout episodes by managing the interaction
  loop between LLM-based agents and environments. It handles model inference,
  environment stepping, reward computation, and trajectory storage with support
  for concurrent multi-pair execution and streaming results.

  The engine implements the standard RL rollout pattern: reset → step* → final
  reward computation → return calculation, while providing flexible callback
  integration for custom model calls and reward functions.
  """

  def __init__(
      self,
      agent: LLMBaseAgent,
      env=None,
      *,
      model_call: Callable[[list[Dict[str, str]]], str],
      final_reward_fn: Optional[Callable[[Dict[str, Any], str], float]] = None,
      max_steps: int = 10,
      gamma: float = 1.0,
      timeout: float = 30.0,
      tokenizer=None,
      chat_parser=None,
  ):
    """Initialize the trajectory collection engine.

    Args:
        agent (LLMBaseAgent): The agent that will interact with the environment
        env (BaseEnv): The environment providing tasks and feedback
        model_call (Callable): Function that takes chat completions and returns
          model response string. Handles the actual LLM inference.
        final_reward_fn (Optional[Callable]): Optional function to compute
          additional reward at episode end. Takes (task, response) and returns
          float. Defaults to zero if not provided.
        max_steps (int): Maximum number of interaction steps before forced
          termination
        gamma (float): Discount factor for return calculation (1.0 = no
          discounting)
        timeout (float): Maximum episode duration in seconds before timeout
          termination
        tokenizer: Optional tokenizer for converting messages to token IDs
        chat_parser: Optional chat parser for formatting messages
    """
    self.agent = agent
    self.env = env
    self.model_call = model_call
    self.final_reward_fn = final_reward_fn or (lambda *_: 0.0)
    self.max_steps = max_steps
    self.gamma = gamma
    self.timeout = timeout

    # Tokenizer utilities for stepwise tokenization
    self.tokenizer = tokenizer
    self.chat_parser = chat_parser


  async def collect(self, mode: str = "Text") -> Any:
    """Execute a complete rollout episode and return the resulting trajectory.

    Orchestrates the full interaction sequence: environment reset, iterative
    agent-environment steps, final reward computation, Monte Carlo return
    calculation, and resource cleanup.

    Args:
        mode (str): Output format. Options:
            - "Text": return full Trajectory object (default).
            - "Token": return flattened tokenized dict for training.
            - "Step": return stepwise tokenized data only.
            - "Conversation": return raw conversation messages.

    Returns:
        Trajectory | dict | list: Depending on mode.
    """
    await self._reset()
    for _ in range(self.max_steps):
        done = await self._one_step()
        if done:
            break
    await self._append_final_reward()
    self._fill_returns()
    await self._close()

    if mode == "Text":
        return self.agent.trajectory
    elif mode == "Step":
        # return stepwise info only
        return [
            {
                "context_tokens": getattr(step, "context_tokens", []),
                "prompt_tokens": getattr(step, "prompt_tokens", []),
                "response_tokens": getattr(step, "response_tokens", []),
                "response_masks": getattr(step, "response_masks", []),
                "reward": step.reward,
                "mc_return": step.mc_return,
            }
            for step in self.agent.trajectory.steps
        ]
    elif mode == "Token":
        # flatten all steps into single batch dict
        prompt_tokens, response_tokens, response_masks = [], [], []
        sep_token_id = getattr(self.tokenizer, "eos_token_id", None)

        for i, step in enumerate(self.agent.trajectory.steps):
            if hasattr(step, "prompt_tokens"):
                prompt_tokens.extend(step.prompt_tokens)
            if hasattr(step, "response_tokens"):
                response_tokens.extend(step.response_tokens)
            if hasattr(step, "response_masks"):
                response_masks.extend(step.response_masks)

            # add separator between steps
            if sep_token_id is not None and i < len(self.agent.trajectory.steps) - 1:
                response_tokens.append(sep_token_id)
                response_masks.append(1)

        return {
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "response_masks": response_masks,
            "trajectory_reward": self.agent.trajectory.reward,
        }
    elif mode == "Conversation":
        # return raw conversation history
        return self.agent.chat_completions
    else:
        raise ValueError(f"Unsupported mode: {mode}")



  @staticmethod
  async def collect_multiple(
      pairs: List[Tuple[LLMBaseAgent, BaseEnv]],
      *,
      model_call: Callable[[list[Dict[str, str]]], str],
      final_reward_fn: Optional[Callable[[Dict[str, Any], str], float]] = None,
      max_steps: int = 10,
      gamma: float = 1.0,
      timeout: float = 30.0,
  ) -> AsyncGenerator[Tuple[int, Trajectory], None]:
    """Execute multiple agent-environment pairs concurrently.

    Runs multiple rollouts in parallel and yields completed trajectories
    as they finish, enabling efficient batch processing with streaming
    results. Useful for distributed training or large-scale evaluation.

    Args:
        pairs (List[Tuple[LLMBaseAgent, BaseEnv]]): List of (agent, environment)
          pairs
        model_call (Callable): Shared model inference function for all pairs
        final_reward_fn (Optional[Callable]): Shared final reward function
        max_steps (int): Maximum steps per episode
        gamma (float): Discount factor for return calculation
        timeout (float): Per-episode timeout in seconds

    Yields:
        Tuple[int, Trajectory]: (pair_index, completed_trajectory) as episodes
        finish
    """

    async def _run_one(i: int, agent: LLMBaseAgent, env: BaseEnv):
      """Execute a single agent-environment pair with the given configuration."""
      engine = TrajectoryCollectEngine(
          agent,
          env,
          model_call=model_call,
          final_reward_fn=final_reward_fn,
          max_steps=max_steps,
          gamma=gamma,
          timeout=timeout,
      )
      traj = await engine.collect()
      print('test')
      return i, traj

    # Launch all pairs concurrently and yield results as they complete
    tasks = [_run_one(i, a, e) for i, (a, e) in enumerate(pairs)]
    for coro in asyncio.as_completed(tasks):
      yield await coro

  async def _reset(self):
    """Initialize the episode by resetting environment and agent state.

    Resets the environment to get initial observation, clears agent state,
    and provides the initial observation to the agent. Also starts the
    episode timer for timeout tracking.
    """
    obs, _ = await asyncio.get_event_loop().run_in_executor(
        None, self.env.reset
    )
    self.agent.reset()
    self.agent.update_from_env(observation=obs, reward=0.0, done=False, info={})
    self._start_ts = time.time()

  async def _one_step(self) -> bool:
    """Execute one complete agent-environment interaction step.

    Performs the core interaction cycle: get agent's chat completions,
    call the model to generate response, parse response into action,
    execute action in environment, and update agent with results.
    Also checks for timeout conditions.

    Returns:
        bool: True if episode should terminate (done or timeout), False to
        continue
    """
    # 1) Generate model response from current conversation context
    resp = await asyncio.get_event_loop().run_in_executor(
        None, self.model_call, self.agent.chat_completions
    )
    action = self.agent.update_from_model(resp).action

    if action is None:
      logger.warning(
          "Agent returned None action, using empty action list as fallback"
      )
      action = []

    # 2) Execute action in environment and get feedback
    obs, rew, done, info = await asyncio.get_event_loop().run_in_executor(
        None, self.env.step, action
    )
    self.agent.update_from_env(obs, rew, done, info)

    # 3) Convert messages to stepwise tokens if tokenizer is available
    if self.tokenizer is not None and self.chat_parser is not None:
      cur_step = self.agent.get_current_state()
      if cur_step is not None:
        try:
          # (a) context tokens (all history)
          context_tokens, _ = convert_messages_to_tokens_and_masks(
              self.agent.chat_completions,
              tokenizer=self.tokenizer,
              parser=self.chat_parser,
              contains_first_msg=True,
              contains_generation_msg=True,
          )
          cur_step.context_tokens = context_tokens

          # (b) last user prompt tokens
          user_messages = [m for m in self.agent.chat_completions if m["role"] == "user"]
          if user_messages:
            last_user_msg = [user_messages[-1]]
            prompt_tokens, _ = convert_messages_to_tokens_and_masks(
                last_user_msg,
                tokenizer=self.tokenizer,
                parser=self.chat_parser,
                contains_first_msg=False,
                contains_generation_msg=True,
            )
            cur_step.prompt_tokens = prompt_tokens

          # (c) assistant response tokens
          response_tokens, response_masks = convert_messages_to_tokens_and_masks(
              [{"role": "assistant", "content": resp}],
              tokenizer=self.tokenizer,
              parser=self.chat_parser,
              contains_first_msg=False,
              contains_generation_msg=False,
          )
          cur_step.response_tokens = response_tokens
          cur_step.response_masks = response_masks

        except Exception as e:
          logger.error(f"Tokenization failed at step: {e}")

    # 4) Check for timeout termination
    if time.time() - self._start_ts > self.timeout:
      self.agent.get_current_state().done = True
      return True
    return done

  async def _append_final_reward(self):
    """Compute and add final reward to the last step of the episode.

    Applies the final reward function (if provided) to the episode's
    final response and adds it to the last step's reward. This enables
    additional reward signals based on overall episode performance.
    """
    last_step = self.agent.get_current_state()
    if last_step is None:
      return
    add_r = await asyncio.get_event_loop().run_in_executor(
        None, self.final_reward_fn, self.env.task, last_step.model_response
    )
    last_step.reward += add_r

  def _fill_returns(self):
    """Compute Monte Carlo returns for all steps in the trajectory.

    Calculates discounted returns working backwards from the final step,
    where each step's return is its immediate reward plus the discounted
    return of subsequent steps. Sets the trajectory's total reward to
    the first step's return.
    """
    traj = self.agent.trajectory
    g = 0.0
    for step in reversed(traj.steps):
      g = step.reward + self.gamma * g
      step.mc_return = g
    traj.reward = traj.steps[0].mc_return if traj.steps else 0.0

  async def _close(self):
    """Clean up resources by closing the environment.

    Ensures proper cleanup of environment resources such as network
    connections, file handles, or external processes.
    """
    await asyncio.get_event_loop().run_in_executor(None, self.env.close)
