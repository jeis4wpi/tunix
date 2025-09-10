# py script
import os
import logging
import sys

os.environ['TPU_LIBRARY_PATH'] = '/home/linchai_google_com/miniconda3/envs/qwen/lib/python3.12/site-packages/libtpu/libtpu.so'

# Data
BATCH_SIZE = 4

# Model
MESH = [(1, 1), ("fsdp", "tp")]
# LoRA
RANK = 16
ALPHA = 2.0

# Train
MAX_STEPS = 4
EVAL_EVERY_N_STEPS = 20
NUM_EPOCHS = 3


# Checkpoint saving
INTERMEDIATE_CKPT_DIR = "/tmp/content/intermediate_ckpt/"
CKPT_DIR = "/tmp/content/ckpts/"
PROFILING_DIR = "/tmp/content/profiling/"

def create_dir(path):
  try:
    os.makedirs(path, exist_ok=True)
    logging.info(f"Created dir: {path}")
  except OSError as e:
    logging.error(f"Error creating directory '{path}': {e}")


create_dir(INTERMEDIATE_CKPT_DIR)
create_dir(CKPT_DIR)
create_dir(PROFILING_DIR)

import jax
mesh = jax.make_mesh(*MESH)
mesh

import os
import kagglehub

# Log in
if "KAGGLE_USERNAME" not in os.environ or "KAGGLE_KEY" not in os.environ:
  kagglehub.login()


from flax import nnx
import kagglehub
from tunix.models.qwen3 import model
from tunix.models.qwen3 import params
import jax.numpy as jnp

MODEL_CP_PATH = kagglehub.model_download("qwen-lm/qwen-3/transformers/0.6b")

config = (
    model.ModelConfig.qwen3_0_6b()
)  # pick correponding config based on model version
qwen3 = params.create_model_from_safe_tensors(MODEL_CP_PATH, config, mesh, jnp.float32)
# qwen3 = params.create_model_from_safe_tensors(MODEL_CP_PATH, config, mesh)
nnx.display(qwen3)

# from transformers import AutoTokenizer

from tunix.examples.data import translation_dataset as data_lib
tokenizer = data_lib.HFTokenizer(
    MODEL_CP_PATH,
    add_bos=True,
    add_eos=True,
    hf_access_token=os.environ.get('T_HF_TOKEN'),
    )

# def templatize(prompts):
#   out = []
#   for p in prompts:
#     out.append(
#         tokenizer.apply_chat_template(
#             [
#                 {"role": "user", "content": p},
#             ],
#             tokenize=False,
#             add_generation_prompt=True,
#             enable_thinking=True,
#         )
#     )
#   return out

import functools
import humanize
def show_hbm_usage():
  """Displays memory usage per device."""
  fmt_size = functools.partial(humanize.naturalsize, binary=True)

  for d in jax.local_devices():
    stats = d.memory_stats()
    used = stats["bytes_in_use"]
    limit = stats["bytes_limit"]
    print(f"Using {fmt_size(used)} / {fmt_size(limit)} ({used/limit:%}) on {d}")

show_hbm_usage()

# from tunix.generate import sampler

# inputs = templatize([
#     "which is larger 9.9 or 9.11?",
#     "如何制作月饼?",
#     "tell me your name, respond in Chinese",
# ])

# sampler = sampler.Sampler(
#     qwen3,
#     tokenizer,
#     sampler.CacheConfig(
#         cache_size=256, num_layers=28, num_kv_heads=8, head_dim=128
#     ),
# )
# out = sampler(inputs, max_generation_steps=128, echo=True)

# for t in out.text:
#   print(t)
#   print("*" * 30)

# show_hbm_usage()

import qwix
def get_lora_model(base_model, mesh, quantize=False):
  if quantize:
    lora_provider = qwix.LoraProvider(
      module_path=".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj",
      rank=RANK,
      alpha=ALPHA,
      # comment the two args below for LoRA (w/o quantisation).
      weight_qtype="nf4",
      tile_size=256,
    )
  else:
    lora_provider = qwix.LoraProvider(
      module_path=".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj",
      rank=RANK,
      alpha=ALPHA,
    )

  model_input = base_model.get_model_input()
  lora_model = qwix.apply_lora_to_model(
      base_model, lora_provider, **model_input
    )

  with mesh:
    state = nnx.state(lora_model)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(lora_model, sharded_state)

  return lora_model

# Loads the training and validation datasets

from tunix.rl import common
from tunix.sft import peft_trainer

train_ds, validation_ds = data_lib.create_datasets(
    dataset_name='mtnt/en-fr',
    # Uncomment the line below to use a Hugging Face dataset.
    # Note that this requires upgrading the 'datasets' package and restarting
    # the Colab runtime.
    # dataset_name='Helsinki-NLP/opus-100',
    global_batch_size=BATCH_SIZE,
    max_target_length=256,
    num_train_epochs=NUM_EPOCHS,
    tokenizer=tokenizer,
)


def gen_model_input_fn(x: peft_trainer.TrainingInput):
  pad_mask = x.input_tokens != tokenizer.pad_id()
  positions = common.build_positions_from_mask(pad_mask)
  attention_mask = common.make_causal_attn_mask(pad_mask)
  return {
      'input_tokens': x.input_tokens,
      'input_mask': x.input_mask,
      'positions': positions,
      'attention_mask': attention_mask,
  }
  
show_hbm_usage()
from tunix.sft import metrics_logger

import optax

logging_option = metrics_logger.MetricsLoggerOptions(
    log_dir="/tmp/tensorboard/full", flush_every_n_steps=20
)
# training_config = peft_trainer.TrainingConfig(
#     eval_every_n_steps=EVAL_EVERY_N_STEPS,
#     max_steps=MAX_STEPS,
#     metrics_logging_options=logging_option,
# )
# trainer = peft_trainer.PeftTrainer(qwen3, optax.adamw(1e-5), training_config)
# trainer = trainer.with_gen_model_input_fn(gen_model_input_fn)

# with jax.profiler.trace(os.path.join(PROFILING_DIR, "full_training")):
#   with mesh:
#     trainer.train(train_ds, validation_ds)
    
# print("Training full model.")
# show_hbm_usage()


# Since LoRA model is sharing backbone with base model,
# restart Colab runtime so base model is loaded as pre-trained.

# LoRA model
lora_qwen3 = get_lora_model(qwen3, mesh=mesh)
nnx.display(lora_qwen3)

training_config = peft_trainer.TrainingConfig(
    eval_every_n_steps=EVAL_EVERY_N_STEPS,
    max_steps=MAX_STEPS,
    checkpoint_root_directory=CKPT_DIR,
)
lora_trainer = peft_trainer.PeftTrainer(
    lora_qwen3, optax.adamw(1e-3), training_config
).with_gen_model_input_fn(gen_model_input_fn)

print("Start to train lora.")
with jax.profiler.trace(os.path.join(PROFILING_DIR, "peft")):
  with mesh:
    lora_trainer.train(train_ds, validation_ds)
print("Training lora.")
show_hbm_usage()


# # Since LoRA model is sharing backbone with base model,
# # restart Colab runtime so base model is loaded as pre-trained.

# #qlora model
# lora_qwen3_quant = get_lora_model(qwen3, mesh=mesh, quantize=True)
# nnx.display(lora_qwen3_quant)

# training_config = peft_trainer.TrainingConfig(
#     eval_every_n_steps=EVAL_EVERY_N_STEPS,
#     max_steps=MAX_STEPS,
#     checkpoint_root_directory=CKPT_DIR,
# )
# qlora_trainer = peft_trainer.PeftTrainer(
#     lora_qwen3_quant, optax.adamw(1e-3), training_config
# ).with_gen_model_input_fn(gen_model_input_fn)

# with jax.profiler.trace(os.path.join(PROFILING_DIR, "peft")):
#   with mesh:
#     qlora_trainer.train(train_ds, validation_ds)
# print("Training qlora.")
# show_hbm_usage()