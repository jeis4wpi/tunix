

# python scripts/get_maxtext_model.py
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../maxtext')))


import jax.numpy as jnp
from flax import nnx
from tunix.models.gemma import gemma as gemma_lib
import sys
import os
import flax.linen as nn

from MaxText.layers import nnx_wrappers
import MaxText as mt
from MaxText import pyconfig
from tunix.rl.rollout.vllm_rollout import VllmRollout
from tunix.rl.rollout import base_rollout
import transformers
import jax 
from MaxText.integration.tunix.tunix_adaptor import TunixMaxTextLlama
from tunix.rl import utils

show_hbm_usage = utils.show_hbm_usage

show_hbm_usage("Before loading model")
def get_ref_maxtext_model():

  #TODO: @mazumdera: change this to use Gemma2-2b-it
  config = pyconfig.initialize(
      ["", "../maxtext/MaxText/configs/base.yml"], #TODO: @mazumdera: why decode.py?
      base_output_directory="gs://dummy_output_dir",  # This is not used in Tunix.
      run_name="none",
      tokenizer_path="../maxtext/assets/tokenizer.gemma",
      per_device_batch_size=1,
      max_target_length=1024,
      steps=10,
      async_checkpointing="false",
      model_name="llama3.1-8b", #"llama3.1-8b"
      checkpoint_period=5, 
      skip_jax_distributed_system="true",
      weight_dtype="bfloat16",
      attention="dot_product"
  )
  
  def create_model(config):
    return mt.from_pretrained(config, rngs=nnx.Rngs(params=0, dropout=1))

  model = nnx.eval_shape(create_model, config=config)

  abstract_model = nnx.eval_shape(create_model, config=config)
  graphdef, abstract_state = nnx.split(abstract_model)
  print('The abstract NNX state (all leaves are abstract arrays):')
  nnx.display(abstract_state)

  @nnx.jit
  def partial_init(config):
    model = create_model(config)
    # nnx.update(model, checkpoint)
    # shard model
    state = nnx.state(model)
    specs = nnx.get_partition_spec(state)
    state = jax.lax.with_sharding_constraint(state, specs)
    nnx.update(model, state)
    return model

  with jax.sharding.use_mesh(model.mesh), nn.logical_axis_rules(config.logical_axis_rules):
    model = partial_init(config)
  print(model)

  tunix_model = TunixMaxTextLlama(
        base_model=model,
        use_attention_mask=False,  # trust Tunix loss masking
    )
  mesh  = tunix_model.base.mesh

  # Add to_hf_mappings method to the model
  def get_hf_mappings():
    return {
        "model.embed_tokens.weight": "token_embedder.embedding",
        "model.norm.weight": "decoder.decoder_norm.scale", 
        "lm_head.weight": "token_embedder.embedding",  # Often shared with embeddings
        # Add more mappings as needed for your specific model
    }
  
  tunix_model.to_hf_mappings = lambda *args: {}
  tunix_model.to_hf_transpose_keys = lambda *args: {}
  tunix_model.lora_to_hf_mappings = lambda *args: {}

  # Add these lines to properly get the graph definition and state
  graphdef, state = nnx.split(tunix_model)
  tunix_model = nnx.merge(graphdef, state)  # Recreate model in proper NNX format
    
  return tunix_model, mesh

model, mesh = get_ref_maxtext_model()
model_tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

print("mesh", mesh)

print(model)
show_hbm_usage("After loading model")


TOTAL_GENERATION_STEPS = 64
MAX_PROMPT_LENGTH = 64  
TEMPERATURE = 0.9
TOP_P = 1.0
TOP_K = None
cache_config = base_rollout.RolloutConfig(max_tokens_to_generate=TOTAL_GENERATION_STEPS, max_prompt_length=MAX_PROMPT_LENGTH, kv_cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256, temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K)

rollout = VllmRollout(model=model,tokenizer=model_tokenizer,cache_config_or_size=cache_config, mesh=mesh,lora_config=None,model_version="meta-llama/Llama-3.1-8B")


rollout.generate(["hello world", "how are you?"])