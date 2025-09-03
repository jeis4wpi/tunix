#!pip install -q kagglehub

# !pip install -q tensorflow
# !pip install -q tensorboardX
# !pip install -q grain
# !pip install --force-reinstall "jax==0.6.2" "jaxlib==0.6.2" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
# !pip install "jax[tpu]==0.7.1.dev20250813" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
# !pip install -q git+https://github.com/google/tunix
#!pip install -e ~/tunix/
# !pip install -q git+https://github.com/google/qwix

#!pip uninstall -q -y flax
# !pip install -q git+https://github.com/google/flax.git
#!pip install -q git+https://github.com/google/flax.git@7a429f33fca2179079f163934a11658f6ddcb039
# !pip install -q tensorflow-datasets

# !pip install -q git+https://github.com/AI-Hypercomputer/pathways-utils.git

import functools
import gc
import os
from pprint import pprint
import re
import time

from flax import nnx
import grain
import humanize
import jax
import jax.numpy as jnp
import kagglehub
import optax
from orbax import checkpoint as ocp
import qwix
import tensorflow_datasets as tfds
from tqdm.auto import tqdm
from tunix.generate import sampler as sampler_lib
from tunix.models.gemma import data as data_lib
from tunix.models.gemma import gemma as gemma_lib
from tunix.models.gemma import params as params_lib
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.rollout import base_rollout
from tunix.rl.grpo.grpo_learner import GrpoConfig, GrpoLearner
from tunix.sft import metrics_logger

os.environ['TPU_LIBRARY_PATH'] = '/home/linchai_google_com/miniconda3/envs/vllm/lib/python3.12/site-packages/libtpu/libtpu.so'
os.environ['SKIP_JAX_PRECOMPILE'] = '1'

jax.devices()


# ====== Data ======
TRAIN_DATA_DIR = "./data/train"
TEST_DATA_DIR = "./data/test"
TRAIN_FRACTION = 1.0

# ====== Reproducibility ======
SEED = 42

# ====== LoRA ======
RANK = 64
ALPHA = 64.0

# ====== Sharding ======
# MESH = [(1, 4), ("fsdp", "tp")]

# ====== GRPO ======
# === Generation during GRPO training ===
MAX_PROMPT_LENGTH = 256
TOTAL_GENERATION_STEPS = 768
# Important to keep a high-ish temperature for varied, diverse responses during
# training.
TEMPERATURE = 0.9
TOP_P = 1.0
TOP_K = 50
# The number of times the policy generates multiple responses for a given prompt
# within a single training step. This corresponds to `G` in Algorithm 1 in the
# paper. The "group" in GRPO comes from here.
NUM_GENERATIONS = 2

# === other GRPO configs ===
# The number of iterations per batch (𝜇 in GRPO algo 1).
NUM_ITERATIONS = 1
# The coefficient for the KL divergence penalty (𝛽) in the GRPO loss function.
# Important to keep a high enough value for this, otherwise, the KL divergence
# can increase unchecked.
BETA = 0.08
# Epsilon value for clipping (𝜀 in GRPO loss in paper). Similar to PPO, for
# stable updates.
EPSILON = 0.2

# ====== Training ======
BATCH_SIZE = 1
# Increase `NUM_BATCHES` and `MAX_STEPS` for better results.
# NUM_BATCHES = 3738
NUM_BATCHES = 4
# Keep `NUM_TEST_BATCHES` low so that evaluation runs quickly. It can be
# increased to a max. of 330 (if batch size is 4).
NUM_TEST_BATCHES = 5 #100 #Anisha: making it small for quick eval

EVAL_EVERY_N_STEPS = 10  # this doesn't matter if `TRAIN_FRACTION = 1.0`.
NUM_EPOCHS = 1  # can potentially train for more epochs

# Number of training steps.
MAX_STEPS = int(NUM_BATCHES * NUM_ITERATIONS * TRAIN_FRACTION * NUM_EPOCHS)

# === AdamW, warmup, cosine scheduler ===
LEARNING_RATE = 3e-6
B1 = 0.9
B2 = 0.99
WEIGHT_DECAY = 0.1
# == Cosine decay with warmup scheduler ==
# Linearly increase learning rate from 0. to 5e-6 in the first 10% training
# steps, and then gradually decrease the learning rate to 0 using cosine
# scheduler.
WARMUP_STEPS = 0.1 * MAX_STEPS
# == Grad clipping ==
# Grad clipping to prevent large gradients. Found this
# important to keep KL divergence in check.
MAX_GRAD_NORM = 0.1

# Checkpoint saving
INTERMEDIATE_CKPT_DIR = "/home/linchai_google_com/content/intermediate_ckpt_llama3/"
CKPT_DIR = "/home/linchai_google_com/content/ckpts_llama3/"
SAVE_INTERVAL_STEPS = 500
MAX_TO_KEEP = 4

# ====== Inference ======
GENERATION_CONFIGS = {
    # greedy search
    "greedy": {"temperature": 1e-4, "top_k": 1, "top_p": 1.0},
    # some randomness
    "standard": {"temperature": 0.7, "top_k": 50, "top_p": 0.95},
    # liberal
    "liberal": {"temperature": 0.85, "top_k": 2000, "top_p": 1.0},
}

def show_hbm_usage():
  """Displays memory usage per device."""
  fmt_size = functools.partial(humanize.naturalsize, binary=True)

  for d in jax.local_devices():
    stats = d.memory_stats()
    used = stats["bytes_in_use"]
    limit = stats["bytes_limit"]
    print(f"Using {fmt_size(used)} / {fmt_size(limit)} ({used/limit:%}) on {d}")

from transformers import AutoTokenizer
model_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

reasoning_start = "<reasoning>"
reasoning_end = "</reasoning>"
solution_start = "<answer>"
solution_end = "</answer>"


SYSTEM_PROMPT = f"""You are given a problem. Think about the problem and \
provide your reasoning. Place it between {reasoning_start} and \
{reasoning_end}. Then, provide the final answer (i.e., just one numerical \
value) between {solution_start} and {solution_end}."""

TEMPLATE = """<start_of_turn>user
{system_prompt}

{question}<end_of_turn>
<start_of_turn>model"""

def extract_hash_answer(text: str) -> str | None:
  if "####" not in text:
    return None
  return text.split("####")[1].strip()


def get_dataset(data_dir, split="train") -> grain.MapDataset:
  # Download data
  if not os.path.exists(data_dir):
    os.makedirs(data_dir)

  data = tfds.data_source(
      "gsm8k",
      split=split,
      data_dir=data_dir,
      builder_kwargs={"file_format": tfds.core.FileFormat.ARRAY_RECORD},
      download=True,
  )

  dataset = (
      grain.MapDataset.source(data)
      .shuffle(seed=42)
      .map(
          lambda x: {
              # passed to model forward pass
              "prompts": TEMPLATE.format(
                  system_prompt=SYSTEM_PROMPT,
                  question=x["question"].decode("utf-8"),
              ),
              # passed to reward functions
              "question": x["question"].decode("utf-8"),
              # passed to reward functions
              "answer": extract_hash_answer(x["answer"].decode("utf-8")),
          }
      )
  )
  return dataset

dataset = get_dataset(TRAIN_DATA_DIR, "train").batch(BATCH_SIZE)[:NUM_BATCHES]

if TRAIN_FRACTION == 1.0:
  train_dataset = dataset.repeat(NUM_EPOCHS)
  val_dataset = None
else:
  train_dataset = dataset[: int(len(dataset) * TRAIN_FRACTION)]
  train_dataset = train_dataset.repeat(NUM_EPOCHS)

  val_dataset = dataset[int(len(dataset) * TRAIN_FRACTION) :].repeat(NUM_EPOCHS)

test_dataset = get_dataset(TEST_DATA_DIR, "test").batch(BATCH_SIZE)[
    :NUM_TEST_BATCHES
]

len(train_dataset), len(val_dataset) if val_dataset is not None else 0, len(
    test_dataset
)

from tunix.rl import utils
# show_hbm_usage = utils.show_hbm_usage

print("HBM usage before loading model:")
show_hbm_usage()

import sys
import os

# add the parent directory (one level up) to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../../maxtext')))

# ! pip install -r ../../maxtext/requirements.txt

import MaxText as mt
from MaxText import pyconfig

# from MaxText.integrations.tunix.tunix_utils import build_tunix_wrapper
from flax import linen as nn
from tunix.models.llama3 import model as llama3_lib
from functools import partial

from etils import epath


def get_ref_maxtext_model(config):

  def create_model(config):
    return mt.from_config(config, rngs=nnx.Rngs(params=0, dropout=1))

  abstract_model = nnx.eval_shape(create_model, config=config)
  graphdef, abstract_state = nnx.split(abstract_model)
  print("The abstract NNX state (all leaves are abstract arrays):")
  nnx.display(abstract_state)
  specs = nnx.get_partition_spec(abstract_state)
  mesh = abstract_model.mesh

  # JIT a function that creates the model state with proper sharding from the start.
  # By providing out_shardings, we instruct JAX to produce sharded output directly,
  # avoiding a large intermediate allocation on a single device.
  with nn.logical_axis_rules(config.logical_axis_rules):
    out_shardings = nn.logical_to_mesh_sharding(specs, mesh)

  @functools.partial(jax.jit, out_shardings=out_shardings)
  def create_sharded_state():
    # This will be JIT-compiled. JAX knows the output sharding and can
    # initialize the parameters directly on the target devices in a sharded way.
    model = create_model(config)
    return nnx.state(model)

  with mesh:
    # Create the model with sharded parameters.
    sharded_state = create_sharded_state()
    model = nnx.merge(graphdef, sharded_state)

    if config.load_parameters_path:
      target_for_restore = jax.tree.map(
          lambda v: v.value,
          sharded_state,
          is_leaf=lambda n: isinstance(n, nnx.Variable),
      )

      try:
        ckptr = ocp.Checkpointer(
            ocp.PyTreeCheckpointHandler(
                restore_concurrent_gb=None,
                save_concurrent_gb=None,
                use_ocdbt=True,
                use_zarr3=True,
            )
        )
        # This is a memory optimization. We don't want to restore the entire checkpoint - only the params.
        # Rather than pass the entire abstract state, which could unnecessarily restore opt_state and such and waste
        # memory, we instead specify here that we are just restoring the params field of the checkpoint
        # (which itself may be a dictionary containing a key named 'params').
        restore_args = ocp.checkpoint_utils.construct_restore_args(
            target_for_restore
        )
        restored = ckptr.restore(
            epath.Path(config.load_parameters_path),
            item={"params": {"params": target_for_restore}},
            transforms={},
            restore_args={"params": {"params": restore_args}},
        )
        checkpoint = restored["params"]["params"]

        if checkpoint:
          nnx.update(model, checkpoint)

      except Exception as e:
        raise ValueError(f"Checkpointing failed: {e}")

    tunix_model = TunixMaxTextLlama(
        base_model=model,
        use_attention_mask=False,  # trust Tunix loss masking
    )

    # model_config = llama3_lib.ModelConfig.llama3_1_8b()
    # tunix_model.config = model_config

  return tunix_model, mesh
# , model_config

def create_maxtext_to_vllm_mappings():
  """Create mappings for transferring MaxText scanned state to vLLM unscanned state."""
  return {
      # Token embeddings - shard vocab dimension for TP
      'base.token_embedder.embedding': (
          'model.embed.embedding',
          ('model', None),
      ),  # checked
      # Final layer norm - no sharding needed
      'base.decoder.decoder_norm.scale': (
          'model.norm.scale',
          (None,),
      ),  # checked
      # LM head (logits projection) - shard vocab dimension for TP
      'base.decoder.logits_dense.kernel': (
          'model.lm_head',
          (None, 'model'),
      ),  # checked
      # Layer-specific mappings (scanned -> unscanned)
      # MLP components - shard hidden dimensions for TP
      'base.decoder.layers.mlp.wi_0.kernel': (  # checked
          'model.layers.*.mlp.gate_proj.kernel',
          (None, 'layer', 'model'),
      ),  # gate_proj: (4096, 14336) - shard output
      'base.decoder.layers.mlp.wi_1.kernel': (  # checked
          'model.layers.*.mlp.up_proj.kernel',
          (None, 'layer', 'model'),
      ),  # up_proj: (4096, 14336) - shard output
      'base.decoder.layers.mlp.wo.kernel': (  # checked
          'model.layers.*.mlp.down_proj.kernel',
          ('model', 'layer', None),
      ),  # down_proj: (14336, 4096) - shard input
      # Layer norms - no sharding needed
      'base.decoder.layers.pre_self_attention_layer_norm.scale': (
          'model.layers.*.input_layernorm.scale',
          (None, 'layer'),  # checked
      ),
      'base.decoder.layers.post_self_attention_layer_norm.scale': (
          'model.layers.*.post_attention_layernorm.scale',
          (None, 'layer'),  # checked
      ),
      # Attention components - shard head dimensions for TP
      'base.decoder.layers.self_attention.query.kernel': (
          'model.layers.*.self_attn.q_proj.kernel',
          (None, 'layer', 'model', None),
      ),  # q_proj: shard num_heads # NOT MATCH
      'base.decoder.layers.self_attention.key.kernel': (
          'model.layers.*.self_attn.k_proj.kernel',
          (None, 'layer', 'model', None),
      ),  # k_proj: shard num_kv_heads
      'base.decoder.layers.self_attention.value.kernel': (
          'model.layers.*.self_attn.v_proj.kernel',
          (None, 'layer', 'model', None),  # match
      ),  # v_proj: shard num_kv_heads
      'base.decoder.layers.self_attention.out.kernel': (
          'model.layers.*.self_attn.o_proj.kernel',
          ('model', 'layer', None, None),
      ),  # o_proj: shard input heads #match
  }

import numpy as np

transpose_keys = {}


def reorder_rope(arr):
  evens = arr[..., ::2]
  odds = arr[..., 1::2]
  return jax.numpy.concatenate((evens, odds), axis=arr.ndim - 1)


def transform_query_kernel(arr):
  head_dim = arr.shape[-1]
  assert head_dim == 128  # hard coded for now
  depth_scale = np.dtype('float32').type(np.sqrt(head_dim))
  arr = arr * depth_scale
  return reorder_rope(arr)


def transform_key_kernel(arr):
  return reorder_rope(arr)


hook_fns = {
    'base.decoder.layers.self_attention.query.kernel': transform_query_kernel,
    'base.decoder.layers.self_attention.key.kernel': transform_key_kernel,
}

# Base model
# gemma, mesh, model_config = get_base_model(
#     ckpt_path=os.path.join(INTERMEDIATE_CKPT_DIR, "state")
# )
from MaxText.integration.tunix.tunix_adaptor import TunixMaxTextLlama

import nest_asyncio
nest_asyncio.apply()  # To fix "This event loop is already running" error in Colab
config_ref = pyconfig.initialize(
    [
        "",
        "../../maxtext/MaxText/configs/base.yml",
    ],  # TODO: @mazumdera: why decode.py?
    base_output_directory="gs://dummy_output_dir",  # This is not used in Tunix.
    run_name="test-tunix-maxtext-llama3.1-8b",
    tokenizer_type="tiktoken",
    tokenizer_path="assets/tokenizer_llama3.tiktoken",
    load_parameters_path="gs://maxtext-model-checkpoints/llama3.1-8b/2025-01-23-19-04/scanned/0/items",
    per_device_batch_size=1,
    max_prefill_predict_length=4,
    max_target_length=1024,
    steps=10,
    async_checkpointing="false",
    model_name="llama3.1-8b",
    checkpoint_period=5,
    skip_jax_distributed_system="true",
    weight_dtype="bfloat16",
    attention="dot_product",
    remat_policy="none",
    # decoder_layer_input="offload",
    # query_proj="offload",
    # key_proj="offload",
    # value_proj="offload",
    opt_type="sgd",
)

llama3_1_8b, mesh = get_ref_maxtext_model(config_ref)
# gemma_maxtext_nnx = nnx.bridge.ToNNX(gemma)
# Instead of:
nnx.display(llama3_1_8b)


# Use:
print("Model initialized successfully")
print(f"Model mesh shape: {mesh.shape}")
# print(f"Model config: {model_config}")

print("HBM usage after loading ref model:")
show_hbm_usage()

# # Policy model
# lora_gemma = get_lora_model(gemma, mesh=mesh)
# nnx.display(lora_gemma)


# Policy model
# This can remain unchanged from default Tunix's colab
# lora_gemma = get_lora_model(gemma, mesh=mesh)

# TODO: @mazumdera: change this to use lora
# lora_gemma = get_lora_model(gemma, mesh=mesh)
# nnx.display(lora_gemma)

config_policy = pyconfig.initialize(
      ["", "/home/linchai_google_com/maxtext/MaxText/configs/base.yml"], #TODO: @mazumdera: why decode.py?
      base_output_directory="gs://dummy_output_dir",  # This is not used in Tunix.
      run_name="test-tunix-maxtext-llama3.1-8b",
      # run_name="test-tunix-maxtext-llama3.1-8b",
      # dataset_path=we use Tunix's dataset
      #TODO: @mazumdera: change this to use checkpoint
      tokenizer_type="tiktoken",
      tokenizer_path="assets/tokenizer_llama3.tiktoken",
      load_parameters_path="gs://maxtext-model-checkpoints/llama3.1-8b/2025-01-23-19-04/scanned/0/items",
      # tokenizer_path="assets/tokenizer.gemma",
      per_device_batch_size=1,
      max_prefill_predict_length=4,
      max_target_length=1024,
      steps=10,
      async_checkpointing="false",
      model_name="llama3.1-8b",
      # model_name="gemma-2b",
      checkpoint_period=5,
      skip_jax_distributed_system="true",
      weight_dtype="bfloat16",
      attention="dot_product",
      remat_policy="none",
      # decoder_layer_input="offload",
      # query_proj="offload",
      # key_proj="offload",
      # value_proj="offload",
      opt_type="sgd",
  )
llama3_1_8b_policy, mesh_policy = get_ref_maxtext_model(config_policy)


llama3_1_8b_policy.to_hf_mappings = create_maxtext_to_vllm_mappings
llama3_1_8b_policy.to_hf_transpose_keys = lambda *args: transpose_keys
llama3_1_8b_policy.lora_to_hf_mappings = lambda *args: None  # No LoRA
llama3_1_8b_policy.to_hf_hook_fns = lambda *args: hook_fns

# gemma_maxtext_nnx = nnx.bridge.ToNNX(gemma)
# Instead of:
nnx.display(llama3_1_8b_policy)

# Use:
print("Model initialized successfully")
print(f"Model mesh shape: {mesh_policy.shape}")
# print(f"Model config: {model_config_policy}")

print("HBM usage after loading policy model:")
show_hbm_usage()

match_format = re.compile(
    rf"^[\s]{{0,}}"
    rf"{reasoning_start}.+?{reasoning_end}.*?"
    rf"{solution_start}(.+?){solution_end}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)

match_format.search(
    f"{reasoning_start}Let me"
    f" think!{reasoning_end}{solution_start}2{solution_end}",
)

def match_format_exactly(prompts, completions, **kargs):
  scores = []
  for completion in completions:
    score = 0
    response = completion
    # Match if format is seen exactly!
    if match_format.search(response) is not None:
      score += 3.0
    scores.append(score)
  return scores

def match_format_approximately(prompts, completions, **kargs):
  scores = []

  for completion in completions:
    score = 0
    response = completion
    # Count how many keywords are seen - we penalize if too many!
    # If we see 1, then plus some points!
    score += 0.5 if response.count(reasoning_start) == 1 else -0.5
    score += 0.5 if response.count(reasoning_end) == 1 else -0.5
    score += 0.5 if response.count(solution_start) == 1 else -0.5
    score += 0.5 if response.count(solution_end) == 1 else -0.5
    scores.append(score)
  return scores

def check_answer(prompts, completions, answer, **kargs):
  responses = completions

  extracted_responses = [
      guess.group(1) if (guess := match_format.search(r)) is not None else None
      for r in responses
  ]

  scores = []
  for guess, true_answer in zip(extracted_responses, answer):
    score = 0
    if guess is None:
      scores.append(0)
      continue
    # Correct answer gets 3 points!
    if guess == true_answer:
      score += 3.0
    # Match if spaces are seen
    elif guess.strip() == true_answer.strip():
      score += 1.5
    else:
      # We also reward it if the answer is close via ratios!
      # Ie if the answer is within some range, reward it!
      try:
        ratio = float(guess) / float(true_answer)
        if ratio >= 0.9 and ratio <= 1.1:
          score += 0.5
        elif ratio >= 0.8 and ratio <= 1.2:
          score += 0.25
        else:
          score -= 1.0  # Penalize wrong answers
      except:
        score -= 0.5  # Penalize
    scores.append(score)
  return scores

match_numbers = re.compile(
    rf"{solution_start}.*?([\d\.]{{1,}})", flags=re.MULTILINE | re.DOTALL
)
match_numbers.findall(f"{solution_start}  0.34  {solution_end}")

def check_numbers(prompts, completions, answer, **kargs):
  question = kargs["question"]
  responses = completions

  extracted_responses = [
      guess.group(1) if (guess := match_numbers.search(r)) is not None else None
      for r in responses
  ]

  scores = []
  print("START ============================")
  print(f"Question: {question[0]}")
  print(f"Answer: {answer[0]}")
  print(f"Response: {responses[0]}")
  print(f"Extracted: {extracted_responses[0]}")
  print("END ==============================")
  for guess, true_answer in zip(extracted_responses, answer):
    if guess is None:
      scores.append(0)
      continue
    # Convert to numbers
    try:
      true_answer = float(true_answer.strip())
      guess = float(guess.strip())
      scores.append(1.5 if guess == true_answer else 0.0)
    except:
      scores.append(0)
      continue

# Ckpt saving
checkpointing_options = ocp.CheckpointManagerOptions(
    save_interval_steps=SAVE_INTERVAL_STEPS, max_to_keep=MAX_TO_KEEP
)

# Metrics logger
metrics_logging_options = metrics_logger.MetricsLoggerOptions(
    log_dir="/home/linchai_google_com/content/tmp/tensorboard/grpo", flush_every_n_steps=20
)

# #TODO: @mazumdera: try optimizer offloading with adamw
# optimizer = optax.adafactor(
#     learning_rate=optax.schedules.warmup_cosine_decay_schedule(
#         init_value=0.0,
#         peak_value=LEARNING_RATE,
#         warmup_steps=WARMUP_STEPS,
#         decay_steps=MAX_STEPS,
#         end_value=0.0,
#     ),
# )
# if MAX_GRAD_NORM is not None:
#   optimizer = optax.chain(
#       optax.clip_by_global_norm(max_norm=MAX_GRAD_NORM),
#       optimizer,
#   )
from MaxText import optimizers as mt_optimizers
from MaxText import maxtext_utils
learning_rate = maxtext_utils.create_learning_rate_schedule(config_policy)
optimizer = mt_optimizers.get_optimizer(config_policy, learning_rate)

# Training config
# ROLLOUT_MESH = [(1, 8), ("fsdp", "tp")]   # simpler mesh for rollout
# rollout_mesh = jax.make_mesh(*MESH, devices=jax.devices())
cluster_config = rl_cluster_lib.ClusterConfig(
    role_to_mesh={
        rl_cluster_lib.Role.ACTOR: mesh,
        rl_cluster_lib.Role.REFERENCE: mesh,
        rl_cluster_lib.Role.ROLLOUT: mesh,
    },
    rollout_engine='vllm',
    offload_to_cpu=False,
    training_config=rl_cluster_lib.RLTrainingConfig(
        actor_optimizer=optimizer,
        eval_every_n_steps=EVAL_EVERY_N_STEPS,
        max_steps=MAX_STEPS,
        gradient_accumulation_steps=1,
        # metrics logging
        metrics_logging_options=metrics_logging_options,
        # checkpoint saving
        checkpoint_root_directory=CKPT_DIR,
        checkpointing_options=checkpointing_options,
    ),
    rollout_config=base_rollout.RolloutConfig(
        max_tokens_to_generate=TOTAL_GENERATION_STEPS,
        max_prompt_length=MAX_PROMPT_LENGTH,
        kv_cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
    ),
    rollout_vllm_model_version="meta-llama/Meta-Llama-3.1-8B",
    rollout_vllm_hbm_utilization=0.2,
    rollout_vllm_tpu_backend_type="jax",
    # rollout_vllm_init_with_random_weights=True,
)

grpo_config = GrpoConfig(
    num_generations=NUM_GENERATIONS,
    num_iterations=NUM_ITERATIONS,
    beta=BETA,
    epsilon=EPSILON,
)

# test trainer logic only to see HBM usage
show_hbm_usage()
with mesh:
  actor_trainer = rl_cluster_lib.rl_trainer.Trainer(
      model=llama3_1_8b_policy,
      optimizer=cluster_config.training_config.actor_optimizer,
      training_config=cluster_config.training_config,
      logical_axis_rules=config_policy.logical_axis_rules,)

  import numpy as np
  from tunix.sft import peft_trainer
  def dummy_datasets(batch_size: int, repeat: int = 1):
    # (num_batch, batch_size, seq_len)
    dummy_input = np.arange(1024).reshape((-1, batch_size, 1024))
    print(f"dummy_input.shape={dummy_input.shape}")
    return [
        peft_trainer.TrainingInput(
            input_tokens=x, input_mask=jnp.ones(x.shape, dtype=jnp.int32)
        )
        for x in dummy_input
    ] * repeat

  def dummy_gen_model_input_fn(x: peft_trainer.TrainingInput):
    print(f"x.input_tokens.shape={x.input_tokens.shape}")
    input = {
        'input_tokens': x.input_tokens,
        'input_mask': x.input_mask,
        'positions': jnp.arange(x.input_tokens.shape[0] * x.input_tokens.shape[1]).reshape((x.input_tokens.shape[0], x.input_tokens.shape[1])),
        'attention_mask': jnp.ones_like(x.input_tokens),
    }
    print (f" input['positions'].shape={input['positions'].shape}")
    print (f" input['attention_mask'].shape={input['attention_mask'].shape}")
    return input

  actor_trainer.with_gen_model_input_fn(dummy_gen_model_input_fn)

  def custom_shard_optimizer(optimizer, mesh, config):
    if mesh.empty:
      return
    optimizer_state_arrays = nnx.state(
        optimizer, nnx.optimizer.OptState
    )  # select only the optimizer state
    optimizer_pspecs = nnx.get_partition_spec(optimizer_state_arrays)
    # print(
    #     "Sharding optimizer state with partition specs: %s",
    #     optimizer_pspecs,
    # )

    def _adjust_pspec(pspec, state):
      print(f"##### original {pspec=}")
      if pspec is None:
        return None
      # state can be a scalar, which is not an array.
      state_ndim = getattr(state, 'ndim', 0)
      if len(pspec) > state_ndim:
        return shd.PartitionSpec(*pspec[:state_ndim])
      print(f"#####adjusted_{pspec=}")
      return pspec

    adjusted_pspecs = jax.tree.map(
        _adjust_pspec, optimizer_pspecs, optimizer_state_arrays
    )
    # print(f"#####{adjusted_pspecs=}")
    from flax import linen as nn
    from flax.linen import partitioning as nn_partitioning
    with nn_partitioning.axis_rules(config.logical_axis_rules):
      optimizer_shardings = nn.logical_to_mesh_sharding(
          adjusted_pspecs,
          mesh,
      )
    # print(f"#####{optimizer_shardings=}")
  #   optimizer_sharded_state = jax.lax.with_sharding_constraint(
  #       optimizer_state_arrays, optimizer_shardings
  #   )
  #   # optimizer_pspecs = nnx.get_partition_spec(optimizer_state_arrays)
  #   # print(
  #   #     "Sharding optimizer state with partition specs after adjust: %s",
  #   #     optimizer_pspecs,
  #   # )
  #   nnx.update(optimizer, optimizer_sharded_state)
  #   pass

  # actor_trainer.with_shard_optimizer(funtools.partial(custom_shard_optimizer, config=config))
  import jax
  jax.profiler.start_trace("gs://linchai-bucket/maxtext_tpu_vllm_xprof/grpo")
  actor_trainer.train(dummy_datasets(1, 4))
  # jax.block_until_ready(nnx.state(actor_trainer))
  jax.profiler.stop_trace()


  # import jax
  # jax.profiler.start_trace("gs://linchai-bucket/maxtext_tpu_vllm_xprof/grpo")
  # dummy_dataset = dummy_datasets(batch_size=1, repeat=4)
  # for x in dummy_dataset:
  #   input = {
  #       'input_tokens': x.input_tokens,
  #       'input_mask': x.input_mask,
  #       'positions': jnp.arange(x.input_tokens.shape[0] * x.input_tokens.shape[1]).reshape((x.input_tokens.shape[0], x.input_tokens.shape[1])),
  #       'attention_mask': jnp.ones_like(x.input_tokens),
  #   }
  #   print (f" input['positions'].shape={input['positions'].shape}")
  #   print (f" input['attention_mask'].shape={input['attention_mask'].shape}")
  #   output = actor_trainer.model(input["input_tokens"], input["positions"], None, input["attention_mask"])
  #   jax.block_until_ready(output)
  # jax.profiler.stop_trace()