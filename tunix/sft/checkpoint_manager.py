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

  # def save(
  #       self,
  #       step: int,
  #       model: nnx.Module,
  #       save_only_lora_params: bool = False,
  #       force: bool = False,
  #   ) -> bool:
  #       """Saves the params for the given step."""
  #       if self._checkpoint_manager is None:
  #           return False
  #       if not force and not self._checkpoint_manager.should_save(step):
  #           return False
        
  #       if save_only_lora_params:
  #           params = nnx.state(model, nnx.LoRAParam)
  #       else:
  #           params = nnx.state(model)

  #       # ---- START: New Diagnostic Step ----
  #       # This code will inspect the contents of `params` to see why it appears empty.
  #       logging.info("--- Running Diagnostics: Inspecting model state ---")

  #       all_leaves = jax.tree_util.tree_leaves(params)
  #       variable_leaves = [leaf for leaf in all_leaves if isinstance(leaf, nnx.Variable)]
  #       other_leaves = [leaf for leaf in all_leaves if not isinstance(leaf, nnx.Variable)]

  #       logging.info("Total leaves found in nnx.state(model): %d", len(all_leaves))
  #       logging.info("Found %d leaves that are `nnx.Variable` instances.", len(variable_leaves))
  #       logging.info("Found %d leaves that are NOT `nnx.Variable` instances.", len(other_leaves))

  #       if other_leaves:
  #           # Log the types of the first few "other" leaves to see what they are.
  #           other_types = list(set(type(leaf) for leaf in other_leaves))
  #           logging.info("Unique types of non-Variable leaves found: %s", other_types)

  #       logging.info("--- End Diagnostics ---")
  #       # ---- END: New Diagnostic Step ----

  #       # The rest of the sanitization logic remains the same.
  #       flat_state = flatten_dict(dict(params))
  #       plain_flat_dict = {
  #           key: leaf.value
  #           for key, leaf in flat_state.items()
  #           if isinstance(leaf, nnx.Variable)
  #       }
  #       pytree_params = unflatten_dict(plain_flat_dict)

  #       # The check that produces your warning message.
  #       if not pytree_params:
  #           logging.warning(
  #               "Skipping checkpoint for step %d. The parameter tree to save is empty.", step
  #           )
  #           return False

  #       jax.block_until_ready(pytree_params)
  #       logging.info("Saving checkpoint for step %d", step)
  #       return self._checkpoint_manager.save(
  #           step,
  #           args=ocp.args.Composite(
  #               items=ocp.args.PyTreeSave(pytree_params),
  #           ),
  #           force=force,
  #       )

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

    # ---- START: New Recursive Flattening Logic ----
    flat_params = {}
    def _recursive_flatten(pytree, prefix=()):
        # If it's a State object, recurse into its items.
        if isinstance(pytree, nnx.State):
            for key, value in dict(pytree).items():
                _recursive_flatten(value, prefix=prefix + (key,))
        # If it's a Variable, extract its value (the array).
        elif isinstance(pytree, nnx.Variable):
            flat_params[prefix] = pytree.value
        # If it's already a raw JAX array (like a PRNGKey), save it.
        elif isinstance(pytree, (jax.Array, jax.ShapeDtypeStruct)):
            flat_params[prefix] = pytree

    _recursive_flatten(params)
    pytree_params = unflatten_dict(flat_params)
    # ---- END: New Recursive Flattening Logic ----

    if not pytree_params:
        logging.warning("Skipping checkpoint: No JAX arrays found in the model state to save.")
        return False

    jax.block_until_ready(pytree_params)
    logging.info("Saving checkpoint for step %d", step)
    return self._checkpoint_manager.save(
        step,
        args=ocp.args.Composite(items=ocp.args.PyTreeSave(pytree_params)),
        force=force,
    )

  def maybe_restore(
    self,
    model: nnx.Module,
    step: int | None = None,
    restore_only_lora_params: bool = False,
) -> int:
    """Restores the params from the latest checkpoint if available and updates the model provided."""
    restore_start = time.time()
    if self._checkpoint_manager is None:
        return 0
    if step is None:
        step = self._checkpoint_manager.latest_step()
    if step is None:
        return 0

    if restore_only_lora_params:
        abstract_state = nnx.state(model, nnx.LoRAParam)
    else:
        abstract_state = nnx.state(model)

    # **THE FIX**: Cast the nnx.State object to a dict before flattening.
    flat_abstract_state = flatten_dict(dict(abstract_state))

    abstract_pytree_flat = {
        key: jax.ShapeDtypeStruct(leaf.value.shape, leaf.value.dtype)
        for key, leaf in flat_abstract_state.items()
        if isinstance(leaf, nnx.Variable)
    }
    abstract_pytree = unflatten_dict(abstract_pytree_flat)

    if not abstract_pytree:
        logging.warning("Skipping restore: The abstract parameter tree is empty.")
        return 0
        
    ckpt = self._checkpoint_manager.restore(
        step,
        args=ocp.args.Composite(
            items=ocp.args.PyTreeRestore(abstract_pytree),
        ),
    )
    
    flat_restored_values = flatten_dict(ckpt.items)

    updates_flat = {}
    for path, restored_value in flat_restored_values.items():
        original_variable = flat_abstract_state[path]
        updates_flat[path] = original_variable.replace(value=restored_value)
    
    update_state = unflatten_dict(updates_flat)

    nnx.update(model, update_state)

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
