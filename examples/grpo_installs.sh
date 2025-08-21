## Use commit 30d7ce8a17739a1a16c7c82d7f5167743f74fef5 for tpu_commons (install this first)
## Use commit 800349c2a50ac1f823d3387cb0f44d0bd6d470d1 for vllm

pip install -q kagglehub

pip install -q tensorflow
pip install -q tensorboardX
pip install -q grain
# pip install --force-reinstall "jax==0.6.2" "jaxlib==0.6.2" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
# pip install "jax[tpu]==0.6.2" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
# pip install -q git+https://github.com/google/tunix
 pip install -e ~/tunix/
pip install -q git+https://github.com/google/qwix

pip uninstall -q -y flax
pip install -q git+https://github.com/google/flax.git@7a429f33fca2179079f163934a11658f6ddcb039

pip install -q tensorflow-datasets
pip install datasets
pip install aqtp
pip install pillow>=11.1.0
pip install pillow
pip install omegaconf
pip install google-cloud-storage
pip install transformers


pip install -q git+https://github.com/AI-Hypercomputer/pathways-utils.git
pip install ipywidgets

pip install nest_asyncio
pip install numba==0.61.2



