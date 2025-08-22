# <a href="https://colab.research.google.com/github/google/tunix/blob/main/examples/grpo_demo.ipynb" ><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# 
# This tutorial demonstrates training the Gemma 2 2B-IT model on the GSM8K math
# reasoning benchmark using Group Relative Policy Optimization (GRPO). GRPO can
# enhance your model's problem-solving skills on mathematical word problems,
# coding problems, etc.
# 
# GRPO is an RL algorithm designed to enhance the reasoning abilities of LLMs. It
# is a variant of Proximal Policy Optimization (PPO) that reduces memory usage by
# eliminating the need for a separate value function model. GRPO works by
# generating multiple responses for a given prompt, evaluating these responses
# using a reward model, and then calculating a relative advantage based on the
# group's performance to update the policy.
# 
# In this tutorial we use Colab's `v2-8` TPU. Let's get started!


# ## Install necessary libraries





# ## Imports


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

from transformers import AutoTokenizer
import nest_asyncio


# os.environ['TPU_LIBRARY_PATH'] = '/home/mazumdera_google_com/miniconda3/envs/vllm/lib/python3.12/site-packages/libtpu/libtpu.so'
# os.environ['TPU_LIBRARY_PATH'] = '/home/mazumdera_google_com/vllm/.venv-py312/lib/python3.12/site-packages/libtpu/libtpu.so'

os.environ['SKIP_JAX_PRECOMPILE'] = '1'



nest_asyncio.apply()  # To fix "This event loop is already running" error in Colab


jax.devices()


# ## Hyperparameters
# 
# Let's define the configuration we are going to use. Note that this is by no
# means a "perfect" set of hyperparameters. To get good results, you might have
# to train the model for longer.


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
MESH = [(1, 4), ("fsdp", "tp")]

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
INTERMEDIATE_CKPT_DIR = "/home/mazumdera_google_com/content/intermediate_ckpt_llama3/"
CKPT_DIR = "/home/mazumdera_google_com/content/ckpts_llama3/"
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


# ## Utility functions


def show_hbm_usage():
  """Displays memory usage per device."""
  fmt_size = functools.partial(humanize.naturalsize, binary=True)

  for d in jax.local_devices():
    stats = d.memory_stats()
    used = stats["bytes_in_use"]
    limit = stats["bytes_limit"]
    print(f"Using {fmt_size(used)} / {fmt_size(limit)} ({used/limit:%}) on {d}")


# ## Data preprocessing
# 
# First, let's define some special tokens. We instruct the model to first reason
# between the `<reasoning>` and `</reasoning>` tokens. After
# reasoning, we expect it to provide the answer between the `<answer>` and
# `</answer>` tokens.

# model_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
model_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")


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


# We use OpenAI's GSM8K dataset. GSM8K comprises grade school math word problems.


def extract_hash_answer(text: str) -> str | None:
  print(f"Extracting answer from: {text}")
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

  loaded_dataset = (
      grain.MapDataset.source(data)
      .shuffle(seed=SEED)
      .map(
          lambda x: {
              # passed to model forward pass
              "prompts": model_tokenizer.apply_chat_template(
                  [
                      {
                          "role": "user",
                          "content": TEMPLATE.format(
                              system_prompt=SYSTEM_PROMPT,
                              question=x["question"].decode("utf-8"),
                          ),
                      },
                  ],
                  tokenize=False,
                  add_generation_prompt=True,
              ),
              # passed to reward functions
              "question": x["question"].decode("utf-8"),
              # passed to reward functions
              "answer": extract_hash_answer(x["answer"].decode("utf-8")),
          }
      )
  )
  return loaded_dataset


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


# Let's see how one batch of the dataset looks like!
# 


for ele in train_dataset[:1]:
  pprint(ele)


# ## Load the policy model and the reference model
# 
# The policy model is the model which is actually trained and whose weights are
# updated. The reference model is the model with which we compute KL divergence.
# This is to ensure that the policy updates are not huge and that it does not
# deviate too much from the reference model.
# 
# Typically, the reference model is the base model, and the policy model is the
# same base model, but with LoRA parameters. Only the LoRA parameters are updated.
# 
# Note: We perform full precision (fp32) training. You can, however, leverage
# Qwix for QAT.
# 
# To load the model, you need to be on [Kaggle](https://www.kaggle.com/) and need
# to have agreed to the Gemma license
# [here](https://www.kaggle.com/models/google/gemma/flax/).


# # Log in
# if "KAGGLE_USERNAME" not in os.environ or "KAGGLE_KEY" not in os.environ:
#   kagglehub.login()


# kaggle_ckpt_path = kagglehub.model_download("google/gemma-2/flax/gemma2-2b-it")
kaggle_ckpt_path = kagglehub.model_download("google/gemma/flax/2b")


# # This is a workaround. The checkpoints on Kaggle don't work with NNX. So, we
# # load the model, save the checkpoint locally, and then reload the model
# # (sharded).
# params = params_lib.load_and_format_params(
#     os.path.join(kaggle_ckpt_path, "gemma2-2b-it")
# )
# gemma = gemma_lib.Transformer.from_params(params, version="2-2b-it")
# checkpointer = ocp.StandardCheckpointer()
# _, state = nnx.split(gemma)
# checkpointer.save(os.path.join(INTERMEDIATE_CKPT_DIR, "state"), state)


# # Wait for the ckpt to save successfully.
# time.sleep(60)


# # Delete the intermediate model to save memory.
# del params
# del gemma
# del state
# gc.collect()


from tunix.rl import utils
# show_hbm_usage = utils.show_hbm_usage

print("HBM usage before loading model:")
show_hbm_usage()


# ### Load MaxText model


import sys
import os

# add the parent directory (one level up) to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../../maxtext')))

# ! pip install -r ../../maxtext/requirements.txt

import MaxText as mt
from MaxText import pyconfig




# #### Convert MaxText model to nnx (use a commit from MaxText repo prior to )
# 
# 


# from MaxText.integrations.tunix.tunix_utils import build_tunix_wrapper
from flax import linen as nn
from tunix.models.llama3 import model as llama3_lib
from functools import partial

from etils import epath


def get_ref_maxtext_model(config):

  def create_model(config):
    return mt.from_pretrained(config, rngs=nnx.Rngs(params=0, dropout=1))

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

    model_config = llama3_lib.ModelConfig.llama3_1_8b()
    tunix_model.config = model_config

  return tunix_model, mesh, model_config



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

config_ref = pyconfig.initialize(
    [
        "",
        "../../maxtext/MaxText/configs/base.yml",
    ],  # TODO: @mazumdera: why decode.py?
    base_output_directory="gs://dummy_output_dir",  # This is not used in Tunix.
    run_name="test-tunix-maxtext-llama3.1-8b",
    tokenizer_type="tiktoken",
    tokenizer_path="assets/tokenizer_llama3.tiktoken",
    load_parameters_path="gs://yixuannwang-maxtext-logs/llama3.1-8b-Instruct/scanned/0/items",
    # load_parameters_path="gs://maxtext-model-checkpoints/llama3.1-8b/2025-01-23-19-04/scanned/0/items",
    per_device_batch_size=1,
    max_prefill_predict_length=4,
    max_target_length=16,
    steps=10,
    async_checkpointing="false",
    model_name="llama3.1-8b",
    checkpoint_period=5,
    skip_jax_distributed_system="true",
    weight_dtype="bfloat16",
    attention="dot_product",
    remat_policy="custom",
    decoder_layer_input="offload",
    query_proj="offload",
    key_proj="offload",
    value_proj="offload",
    opt_type="sgd",
)

llama3_1_8b, mesh, model_config = get_ref_maxtext_model(config_ref)
# gemma_maxtext_nnx = nnx.bridge.ToNNX(gemma)
# Instead of:
nnx.display(llama3_1_8b)


# Use:
print("Model initialized successfully")
print(f"Model mesh shape: {mesh.shape}")
print(f"Model config: {model_config}")


_maxtext_state_flatten = nnx.state(llama3_1_8b).flat_state()
maxtext_state_flatten = {
    '.'.join(str(key) for key in keys): v for keys, v in _maxtext_state_flatten
}
print(f"maxtext_state_flatten[base.token_embedder.embedding].value={maxtext_state_flatten['base.token_embedder.embedding'].value}")



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
      ["", "/home/mazumdera_google_com/maxtext/MaxText/configs/base.yml"], #TODO: @mazumdera: why decode.py?
      base_output_directory="gs://dummy_output_dir",  # This is not used in Tunix.
      run_name="test-tunix-maxtext-llama3.1-8b",
      # run_name="test-tunix-maxtext-llama3.1-8b",
      # dataset_path=we use Tunix's dataset
      #TODO: @mazumdera: change this to use checkpoint
      tokenizer_type="tiktoken",
      tokenizer_path="assets/tokenizer_llama3.tiktoken",
      load_parameters_path="gs://yixuannwang-maxtext-logs/llama3.1-8b-Instruct/scanned/0/items",
    #   load_parameters_path="gs://maxtext-model-checkpoints/llama3.1-8b/2025-01-23-19-04/scanned/0/items",
      # tokenizer_path="assets/tokenizer.gemma",
      per_device_batch_size=1,
      max_prefill_predict_length=4,
      max_target_length=16,
      steps=10,
      async_checkpointing="false",
      model_name="llama3.1-8b",
      # model_name="gemma-2b",
      checkpoint_period=5,
      skip_jax_distributed_system="true",
      weight_dtype="bfloat16",
      attention="dot_product",
      remat_policy="custom",
      decoder_layer_input="offload",
      query_proj="offload",
      key_proj="offload",
      value_proj="offload",
      opt_type="sgd",
  )
llama3_1_8b_policy, mesh_policy, model_config_policy = get_ref_maxtext_model(config_policy)


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
print(f"Model config: {model_config_policy}")





print("HBM usage after loading policy model:")
show_hbm_usage()


_maxtext_state_flatten = nnx.state(llama3_1_8b_policy).flat_state()
maxtext_state_flatten = {
    '.'.join(str(key) for key in keys): v for keys, v in _maxtext_state_flatten
}
print(f"maxtext_state_flatten[base.token_embedder.embedding].value={maxtext_state_flatten['base.token_embedder.embedding'].value}")



# ## Define reward functions
# 
# We define four reward functions:
# 
# - reward if the format of the output exactly matches the instruction given in
# `TEMPLATE`;
# - reward if the format of the output approximately matches the instruction given
# in `TEMPLATE`;
# - reward if the answer is correct/partially correct;
# - Sometimes, the text between `<answer>`, `</answer>` might not be one
#   number. So, extract the number, and reward the model if the answer is correct.
# 
# The reward functions are inspired from
# [here](https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb).
# 
# First off, let's define a RegEx for checking whether the format matches.


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


# Give the model a reward of 3 points if the format matches exactly.


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


# We also reward the model if the format of the output matches partially.


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


# Reward the model if the answer is correct. A reward is also given if the answer
# does not match exactly, i.e., based on how close the answer is to the correct
# value.


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


# Sometimes, the text between `<answer>` and `</answer>` might not be one
# number; it can be a sentence. So, we extract the number and compare the answer.


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
  return scores


# ## Evaluate
# 
# 
# Before we train the model, let's evaluate the model on the test set so we can
# see the improvement post training.
# 
# We evaluate it in two ways:
# 
# **Quantitative**
# 
# * **Answer Accuracy**: percentage of samples for which the model predicts the
# correct final numerical answer  
# * **Answer (Partial) Accuracy**: percentage of samples for which the model
# predicts a final numerical answer such that the \`model answer / answer\`
# ratio lies between 0.9 and 1.1.  
# * **Format Accuracy**: percentage of samples for which the model outputs the
# correct format, i.e., reasoning between the reasoning special tokens, and the
# final answer between the \`\<start\_answer\>\`, \`\<end\_answer\>\` tokens.
# 
# **Qualitative**
# 
# We'll also print outputs for a few given questions so that we can compare the generated output later.
# 


def generate(
    question, sampler, temperature=0.7, top_k=50, top_p=0.95, seed=None
):
  """Given prompt, generates text."""

  if isinstance(question, str):
    input_batch = [
        TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            question=question,
        ),
    ]
  else:
    input_batch = [
        TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            question=q,
        )
        for q in question
    ]

  out_data = sampler(
      input_strings=input_batch,
      total_generation_steps=768,
      temperature=temperature,
      top_k=top_k,
      top_p=top_p,
      echo=False,
      seed=seed if seed is not None else None,
  )

  output = out_data.text
  if isinstance(question, str):
    return output[0]
  return output


def evaluate(
    dataset,
    sampler,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    num_passes=1,
    corr_lst=False,
    make_lst=False,
):
  """Computes accuracy and percentage of outputs matching the format."""

  response_lst = []
  corr = 0
  partially_corr = 0
  corr_format = 0
  total = 0

  for batch in tqdm(dataset):
    answers = batch["answer"]
    questions = batch["question"]

    multiple_call_responses = [[] for _ in range(len(questions))]
    for p in range(num_passes):
      responses = generate(
          questions, sampler, temperature, top_k, top_p, seed=p
      )
      for idx, response in enumerate(responses):
        multiple_call_responses[idx].append(response)

    for question, multiple_call_response, answer in zip(
        questions, multiple_call_responses, answers
    ):
      # check answer
      corr_ctr_per_question = 0
      partially_corr_per_question = 0
      corr_format_per_question = 0
      for response in multiple_call_response:
        extracted_response = (
            guess.group(1)
            if (guess := match_numbers.search(response)) is not None
            else "-1000000"
        )
        try:
          if float(extracted_response.strip()) == float(answer.strip()):
            corr_ctr_per_question += 1

          ratio = float(extracted_response.strip()) / float(answer.strip())
          if ratio >= 0.9 and ratio <= 1.1:
            partially_corr_per_question += 1
        except:
          print("SKIPPED")

        # check format
        if match_format.search(response) is not None:
          corr_format_per_question += 1

        if (
            corr_ctr_per_question > 0
            and partially_corr_per_question > 0
            and corr_format_per_question > 0
        ):
          break

      if corr_ctr_per_question > 0:
        corr += 1
        if corr_lst and make_lst:
          response_lst.append((question, answer, multiple_call_response))
      else:
        if not corr_lst and make_lst:
          response_lst.append((question, answer, multiple_call_response))
      if partially_corr_per_question > 0:
        partially_corr += 1
      if corr_format_per_question > 0:
        corr_format += 1

      total += 1
      if total % 10 == 0:
        print(
            f"===> {corr=}, {total=}, {corr / total * 100=}, "
            f"{partially_corr / total * 100=}, {corr_format / total * 100=}"
        )

  to_return = (
      corr,
      total,
      corr / total * 100,
      partially_corr / total * 100,
      corr_format / total * 100,
  )
  if make_lst:
    return to_return, response_lst
  return to_return


gemma_tokenizer = data_lib.GemmaTokenizer()
sampler = sampler_lib.Sampler(
    # transformer=lora_gemma,
    transformer=llama3_1_8b_policy,
    tokenizer=gemma_tokenizer,
    cache_config=sampler_lib.CacheConfig(
        cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
        num_layers=model_config.num_layers,
        num_kv_heads=model_config.num_kv_heads,
        head_dim=model_config.head_dim,
    ),
)


(corr, total, accuracy, partial_accuracy, format_accuracy) = evaluate(
    test_dataset,
    sampler,
    **GENERATION_CONFIGS["greedy"],
)
print(
    f"{corr=}, {total=}, {accuracy=}%, {partial_accuracy=}%,"
    f" {format_accuracy=}%"
)

# # TODO: @mazumdera: why is this 0?
# # corr=0, total=5, accuracy=0.0%, partial_accuracy=0.0%, format_accuracy=0.0%



# for eval_example in QUALITATIVE_EVAL_EXAMPLES:
#   question = eval_example["question"]
#   answer = eval_example["answer"]
#   response = generate(
#       question,
#       sampler,
#       temperature=INFERENCE_TEMPERATURE,
#       top_k=INFERENCE_TOP_K,
#       top_p=INFERENCE_TOP_P,
#   )

#   print(f"Question:\n{question}")
#   print(f"Answer:\n{answer}")
#   print(f"Response:\n{response}")
#   print("===============")


# ## Train
# 
# Let's set up all the configs first - checkpointing, metric logging and training.
# We then train the model.


# Ckpt saving
checkpointing_options = ocp.CheckpointManagerOptions(
    save_interval_steps=SAVE_INTERVAL_STEPS, max_to_keep=MAX_TO_KEEP
)

# Metrics logger
metrics_logging_options = metrics_logger.MetricsLoggerOptions(
    log_dir="/home/mazumdera_google_com/content/tmp/tensorboard/grpo", flush_every_n_steps=20
)


# Logs
# %load_ext tensorboard
# %tensorboard --logdir /home/mazumdera_google_com/content/tmp/tensorboard/grpo --port=0
# %reload_ext tensorboard


# # Training config
# training_config = GrpoTrainingConfig(
#     max_prompt_length=MAX_PROMPT_LENGTH,
#     total_generation_steps=TOTAL_GENERATION_STEPS,
#     num_generations=NUM_GENERATIONS,
#     num_iterations=NUM_ITERATIONS,
#     beta=BETA,
#     epsilon=EPSILON,
#     temperature=TEMPERATURE,
#     top_p=TOP_P,
#     top_k=TOP_K,
#     eval_every_n_steps=EVAL_EVERY_N_STEPS,
#     max_steps=MAX_STEPS,
#     # metrics logging
#     metrics_logging_options=metrics_logging_options,
#     # checkpoint saving
#     checkpoint_root_directory=CKPT_DIR,
#     checkpointing_options=checkpointing_options,
# )


# Optimizer, learning rate scheduler, gradient clipping
# optimizer = optax.adamw(
#     learning_rate=optax.schedules.warmup_cosine_decay_schedule(
#         init_value=0.0,
#         peak_value=LEARNING_RATE,
#         warmup_steps=WARMUP_STEPS,
#         decay_steps=MAX_STEPS,
#         end_value=0.0,
#     ),
#     b1=B1,
#     b2=B2,
#     weight_decay=WEIGHT_DECAY,
# )
#TODO: @mazumdera: try optimizer offloading with adamw
optimizer = optax.adafactor(
    learning_rate=optax.schedules.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        decay_steps=MAX_STEPS,
        end_value=0.0,
    ),
)
if MAX_GRAD_NORM is not None:
  optimizer = optax.chain(
      optax.clip_by_global_norm(max_norm=MAX_GRAD_NORM),
      optimizer,
  )


# Training config
ROLLOUT_MESH = [(1, 8), ("fsdp", "tp")]   # simpler mesh for rollout
rollout_mesh = jax.make_mesh(*MESH, devices=jax.devices())
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





# # Now lora_gemma's parameters are annotated with the specified sharding.
# # When lora_gemma is used inside a jitted function, JAX will respect these
# # shardings.

# # You can inspect the sharding of a parameter's value.
# # The sharding will be concrete after being passed through a jitted function.
# @jax.jit
# def get_sharded_kernel(model):
#     return model.base.token_embedder.embedding

# with mesh:
#     sharded_kernel_value = get_sharded_kernel(llama3_1_8b_policy)

# print("Sharding of embed kernel:")
# print(sharded_kernel_value)



# RL cluster


rl_cluster = rl_cluster_lib.RLCluster(
    actor=llama3_1_8b_policy,
    reference=llama3_1_8b,
    tokenizer=model_tokenizer,
    cluster_config=cluster_config,
)

# GRPO Trainer
grpo_trainer = GrpoLearner(
    rl_cluster=rl_cluster,
    reward_fns=[
        match_format_exactly,
        match_format_approximately,
        check_answer,
        check_numbers,
    ],
    grpo_config=grpo_config,
)



# verify if vllm sampler works
from tunix.rl.rollout.base_rollout import RolloutConfig
output = rl_cluster.rollout.generate(
    ["The capital of France is"],
    rollout_config=RolloutConfig(
        n=1, max_tokens_to_generate=64, temperature=0.1
    ),
)


print(f"Output: {output}")


# 


import jax
jax.profiler.start_trace("/home/mazumdera_google_com/tmp/jax_traces/grpo")
with mesh:
  grpo_trainer.train(dataset)
jax.profiler.stop_trace()


# ## Evaluate
# 
# Let's evaluate our model!


# Load checkpoint first.

trained_ckpt_path = os.path.join(CKPT_DIR, str(MAX_STEPS), "model_params")

abs_params = jax.tree.map(
    lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
    nnx.state(llama3_1_8b_policy, nnx.Param),
)
checkpointer = ocp.StandardCheckpointer()
trained_lora_params = checkpointer.restore(trained_ckpt_path, target=abs_params)

nnx.update(
    llama3_1_8b_policy,
    jax.tree.map(
        lambda a, b: b,
        nnx.state(llama3_1_8b_policy, nnx.Param),
        trained_lora_params,
    ),
)


gemma_tokenizer = data_lib.GemmaTokenizer()
sampler = sampler_lib.Sampler(
    transformer=llama3_1_8b_policy,
    tokenizer=gemma_tokenizer,
    cache_config=sampler_lib.CacheConfig(
        cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
        num_layers=model_config.num_layers,
        num_kv_heads=model_config.num_kv_heads,
        head_dim=model_config.head_dim,
    ),
)


(corr, total, accuracy, partial_accuracy, format_accuracy) = evaluate(
    test_dataset,
    sampler,
    **GENERATION_CONFIGS["greedy"],
)
print(
    f"{corr=}, {total=}, {accuracy=}%, {partial_accuracy=}%,"
    f" {format_accuracy=}%"
)


# for eval_example in QUALITATIVE_EVAL_EXAMPLES:
#   question = eval_example["question"]
#   answer = eval_example["answer"]
#   response = generate(
#       question,
#       sampler,
#       temperature=INFERENCE_TEMPERATURE,
#       top_k=INFERENCE_TOP_K,
#       top_p=INFERENCE_TOP_P,
#   )

#   print(f"Question:\n{question}")
#   print(f"Answer:\n{answer}")
#   print(f"Response:\n{response}")
#   print("===============")
