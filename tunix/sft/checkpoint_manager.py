# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Checkpoint manager for PEFT."""

import time
from absl import logging
from flax import nnx
import jax
import orbax.checkpoint as ocp
from flax.traverse_util import flatten_dict, unflatten_dict

_DEFAULT_CHECKPOINTING_OPTIONS = ocp.CheckpointManagerOptions(
    save_decision_policy=ocp.checkpoint_managers.ContinuousCheckpointingPolicy(
        minimum_interval_secs=180,
    ),
    max_to_keep=3,
)


class CheckpointManager:
  """Checkpoint manager for PEFT."""

  def __init__(
      self,
      root_directory: str | None = None,
      options: ocp.CheckpointManagerOptions | None = None,
      checkpoint_storage_use_ocdbt: bool = True,
      checkpoint_storage_use_zarr3: bool = True,
  ):
    """Initializes the checkpoint manager.

    Args:
      root_directory: The root directory for the checkpoint manager. If None,
        the checkpoint manager will be disabled.
      options: The options for the checkpoint manager.
    """
    self._checkpoint_manager: ocp.CheckpointManager | None = None
    if root_directory is not None:
      item_handlers = {
          "items": ocp.PyTreeCheckpointHandler(use_ocdbt=checkpoint_storage_use_ocdbt, use_zarr3=checkpoint_storage_use_zarr3)
      }
      item_names = ("items",)
      self._checkpoint_manager = ocp.CheckpointManager(
          root_directory,
          item_names=item_names,
          item_handlers=item_handlers,
          options=options or _DEFAULT_CHECKPOINTING_OPTIONS,
      )

  def latest_step(self) -> int | None:
    """Returns the latest step."""
    if self._checkpoint_manager is None:
      return None
    return self._checkpoint_manager.latest_step()

  def save(
    self,
    step: int,
    model: nnx.Module,
    save_only_lora_params: bool = False,
    force: bool = False,
) -> bool:
    """Saves the params for the given step."""
    if self._checkpoint_manager is None:
      return False
    if not force and not self._checkpoint_manager.should_save(step):
      return False

    if save_only_lora_params:
      params = nnx.state(model, nnx.LoRAParam)
    else:
      params = nnx.state(model)

    # **THE FIX**: Cast the nnx.State object to a dict before flattening.
    # This solves the AssertionError and allows the sanitization to work.
    flat_state = flatten_dict(dict(params))

    # Create a new flat dictionary, unwrapping the nnx.Variable objects.
    plain_flat_dict = {
        key: leaf.value
        for key, leaf in flat_state.items()
        if isinstance(leaf, nnx.Variable)
    }

    # Rebuild the nested structure from the clean, flat dictionary.
    pytree_params = unflatten_dict(plain_flat_dict)

    # Gracefully handle the case where the parameter tree is empty.
    if not pytree_params:
        logging.warning(
            "Skipping checkpoint for step %d. The parameter tree to save is empty.", step
        )
        return False

    # Block until all computations on the concrete arrays are finished.
    jax.block_until_ready(pytree_params)

    logging.info("Saving checkpoint for step %d", step)

    return self._checkpoint_manager.save(
        step,
        args=ocp.args.Composite(
            items=ocp.args.PyTreeSave(pytree_params),
        ),
        force=force,
    )

  def maybe_restore(
      self,
      model: nnx.Module,
      step: int | None = None,
      restore_only_lora_params: bool = False,
  ) -> int:
    """Restores the params from the latest checkpoint if available and updates the model provided.

    Args:
      model: The model to restore the params for.
      step: The step to restore the params from. If None, the latest step will
        be used.
      restore_only_lora_params: Whether to restore only the LoRA params.

    Returns:
      The step of the restored checkpoint or 0 if no checkpoint is available.
    """
    restore_start = time.time()
    if self._checkpoint_manager is None:
      return 0
    if step is None:
      step = self._checkpoint_manager.latest_step()
      # If no checkpoint is available, return 0.
      if step is None:
        return 0
    # Load the params from the checkpoint.
    if restore_only_lora_params:
      abstract_params = nnx.state(model, nnx.LoRAParam)
    else:
      abstract_params = nnx.state(model)
    abstract_pytree = jax.tree.map(
        lambda x: x.value if isinstance(x, nnx.Variable) else x,
        abstract_params,
        is_leaf=lambda n: isinstance(n, nnx.Variable),
    )
    ckpt = self._checkpoint_manager.restore(
        step,
        args=ocp.args.Composite(
            items=ocp.args.PyTreeRestore(abstract_pytree),
        ),
    )
    # Update the model state with params from the restored checkpoint.
    # Create a new State object from the restored pytree of arrays,
    # using the abstract_params as a structural template.
    restored_state = jax.tree.map(
        lambda var, val: var.replace(value=val), abstract_params, ckpt.items
    )
    nnx.update(model, restored_state)
    logging.info(
        "Restored params from step: %d in %.3f seconds",
        step,
        time.time() - restore_start,
    )
    return step

  def close(self):
    """Closes the checkpoint manager."""
    if self._checkpoint_manager is None:
      return
    self._checkpoint_manager.close()
