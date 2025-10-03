# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
import chex
from flax import nnx
import jax
import numpy as np
import optax
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl import utils
from tunix.rl.rollout import base_rollout
from tunix.tests import test_common as tc

os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

Mesh = jax.sharding.Mesh


class RlClusterTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

    cls.num_cpus = int(os.environ.get('DEVICE_COUNTS', 4))
    chex.set_n_cpu_devices(cls.num_cpus)
    print(f'Setting up test with {cls.num_cpus} CPU devices before JAX init')
    cls.device_count = jax.device_count()

  def test_model_loading_with_resharding(self):
    split_index = self.device_count // 2

    actor_mesh = Mesh(
        np.array(jax.devices()[:split_index]).reshape(split_index, 1),
        ('fsdp', 'tp'),
    )
    rollout_mesh = Mesh(
        np.array(jax.devices()[split_index:]).reshape(1, split_index),
        ('fsdp', 'tp'),
    )
    cluster_config = rl_cluster_lib.ClusterConfig(
        role_to_mesh={
            rl_cluster_lib.Role.ACTOR: actor_mesh,
            rl_cluster_lib.Role.REFERENCE: actor_mesh,
            rl_cluster_lib.Role.ROLLOUT: rollout_mesh,
        },
        rollout_engine='vanilla',
        offload_to_cpu=False,
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optax.sgd(1e-3),
            eval_every_n_steps=1,
            max_steps=10,
            gradient_accumulation_steps=None,
        ),
        rollout_config=base_rollout.RolloutConfig(
            max_tokens_to_generate=10,
            max_prompt_length=256,
            kv_cache_size=1024,
        ),
    )

    vocab = tc.MockVocab()
    model = tc.ToyTransformer(rngs=nnx.Rngs(0), vocab_size=vocab.GetPieceSize())
    ref_model = tc.ToyTransformer(
        rngs=nnx.Rngs(0), vocab_size=vocab.GetPieceSize()
    )

    original_actor_mesh = utils.get_pytree_mesh_info(nnx.state(model))
    self.assertIsNone(original_actor_mesh)

    rl_cluster = rl_cluster_lib.RLCluster(
        actor=model,
        reference=ref_model,
        tokenizer=vocab,
        cluster_config=cluster_config,
    )
    trainer_actor_mesh = utils.get_pytree_mesh_info(
        nnx.state(rl_cluster.actor_trainer.model)
    )
    self.assertEqual(trainer_actor_mesh, actor_mesh)

    rollout_actor_mesh = utils.get_pytree_mesh_info(
        nnx.state(rl_cluster.rollout.model())
    )
    self.assertEqual(rollout_actor_mesh, rollout_mesh)

    ref_model_mesh = utils.get_pytree_mesh_info(
        nnx.state(rl_cluster.inference_worker._models['reference'])
    )
    self.assertEqual(ref_model_mesh, actor_mesh)

  @parameterized.named_parameters(
      ('1', None, None, None, None, [None, None, None, None, 1]),
      ('2', 8, None, None, None, [8, 8, 8, 8, 1]),
      ('3', 8, 2, None, None, [8, 2, 2, 2, 4]),
      ('4', 8, 4, 8, None, [8, 4, 8, 4, 2]),
      ('5', 8, 4, None, 8, [8, 4, 4, 8, 2]),
      ('6', 16, 8, 8, 16, [16, 8, 8, 16, 2]),
  )
  def test_batch_sizes(
      self,
      mini_batch_size,
      training_micro_batch_size,
      rollout_micro_batch_size,
      compute_logps_micro_batch_size,
      expected_values,
  ):
    cfg = rl_cluster_lib.RLTrainingConfig(
        actor_optimizer=optax.sgd(1e-3),
        critic_optimizer=None,
        mini_batch_size=mini_batch_size,
        training_micro_batch_size=training_micro_batch_size,
        rollout_micro_batch_size=rollout_micro_batch_size,
        compute_logps_micro_batch_size=compute_logps_micro_batch_size,
        eval_every_n_steps=1,
    )

    self.assertEqual(
        expected_values,
        [
            cfg.mini_batch_size,
            cfg.training_micro_batch_size,
            cfg.rollout_micro_batch_size,
            cfg.compute_logps_micro_batch_size,
            cfg.gradient_accumulation_steps,
        ],
    )

  @parameterized.named_parameters(
      ('1', 2, 4, None, None),
      ('2', 8, 3, None, None),
      ('3', 8, 4, 3, None),
      ('4', 8, 4, None, 3),
      ('5', None, 2, None, None),
      ('6', None, None, 2, None),
  )
  def test_batch_sizes_errors(
      self,
      mini_batch_size,
      training_micro_batch_size,
      rollout_micro_batch_size,
      compute_logps_micro_batch_size,
  ):
    with self.assertRaises(ValueError):
      rl_cluster_lib.RLTrainingConfig(
          actor_optimizer=optax.sgd(1e-3),
          critic_optimizer=None,
          mini_batch_size=mini_batch_size,
          training_micro_batch_size=training_micro_batch_size,
          rollout_micro_batch_size=rollout_micro_batch_size,
          compute_logps_micro_batch_size=compute_logps_micro_batch_size,
          eval_every_n_steps=1,
      )

  def test_generate_with_chat_template(self):
    mesh = Mesh(
        np.array(jax.devices()).reshape(self.device_count, 1), ('fsdp', 'tp')
    )
    cluster_config = rl_cluster_lib.ClusterConfig(
        role_to_mesh={
            rl_cluster_lib.Role.ACTOR: mesh,
            rl_cluster_lib.Role.ROLLOUT: mesh,
        },
        rollout_engine='vanilla',
        offload_to_cpu=False,
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optax.sgd(1e-3),
            critic_optimizer=None,
            eval_every_n_steps=1,
            max_steps=10,
            mini_batch_size=1,
            rollout_micro_batch_size=1,
        ),
        rollout_config=base_rollout.RolloutConfig(
            max_tokens_to_generate=10,
            max_prompt_length=256,
            kv_cache_size=1024,
        ),
    )

    mock_tokenizer = mock.MagicMock(spec=PreTrainedTokenizerBase)
    mock_tokenizer.apply_chat_template.return_value = 'formatted prompt'
    mock_tokenizer.bos_id = 0
    mock_tokenizer.eos_id = 1
    mock_tokenizer.pad_id = 0

    vocab = tc.MockVocab()
    model = tc.ToyTransformer(rngs=nnx.Rngs(0), vocab_size=vocab.GetPieceSize())

    rl_cluster = rl_cluster_lib.RLCluster(
        actor=model,
        tokenizer=mock_tokenizer,
        cluster_config=cluster_config,
    )

    expected_text = 'generated text'
    rl_cluster.generate = mock.MagicMock(
        return_value=base_rollout.RolloutOutput(
            text=[expected_text],
            logits=np.zeros((1, 1, 1)),
            tokens=np.zeros((1, 1)),
            left_padded_prompt_tokens=np.zeros((1, 1)),
            logprobs=None,
        )
    )

    messages = [{'role': 'user', 'content': 'Hello'}]
    result = rl_cluster.generate_with_chat_template(messages, mode='eval')

    self.assertEqual(result, expected_text)
    mock_tokenizer.apply_chat_template.assert_called_once_with(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    rl_cluster.generate.assert_called_once_with(
        prompts=['formatted prompt'],
        mode=rl_cluster_lib.Mode.EVAL,
        micro_batch_size=1,
    )

if __name__ == '__main__':
  absltest.main()
