from __future__ import annotations

import jax
from flax import nnx

from tunix.sft import utils as sft_utils
from absl.testing import absltest


class MyModel(nnx.Module):

  def __init__(self, din: int, dmid: int, dout: int, *, rngs: nnx.Rngs):
    self.linear1 = nnx.Linear(din, dmid, rngs=rngs)
    self.bn = nnx.BatchNorm(dmid, rngs=rngs)
    self.dropout = nnx.Dropout(0.5, rngs=rngs)
    self.linear2 = nnx.Linear(dmid, dout, rngs=rngs)
    self.config_value = 123
    self.config_list = [1, 2, 3]

  def __call__(self, x: jax.Array, *, train: bool):
    x = self.linear1(x)
    x = self.bn(x, train=train)
    x = self.dropout(x, train=train)
    return self.linear2(x)


def test_colab_nnx_display_lists_submodules():
  model = MyModel(din=10, dmid=20, dout=5, rngs=nnx.Rngs(0))
  summary = sft_utils.safe_display(model)

  assert "linear1" in summary
  assert "bn" in summary
  assert "dropout" in summary
  assert "linear2" in summary
  assert summary.strip()


if __name__ == "__main__":
  absltest.main()
