"""

Please run

[ -d vision_transformer ] || git clone --depth=1 https://github.com/google-research/vision_transformer
mv vision_transformer/vit_jax .

"""

from vit_jax import models
from vit_jax import checkpoint
# from vit_jax.configs import augreg as augreg_config
from vit_jax.configs import models as models_config

import tensorflow as tf # only needed to get dataset
import tensorflow_datasets as tfds # only needed to get dataset
from matplotlib import pyplot as plt


"""
The `filename` contains the model, training dataset and fine tuning dataset.

First we create the dataset loader objects.
"""

filename = 'R26_S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0--oxford_iiit_pet-steps_0k-lr_0.003-res_384'

tfds_name = filename.split('--')[1].split('-')[0]
resolution = int(filename.split('_')[-1])

print(f"tfds_name: {tfds_name}")
print(f"resolution: {resolution}")

ds, ds_info = tfds.load(tfds_name, with_info=True)
ds_info
d = next(iter(ds['test'])) # Get a single example from dataset for inference.


"""
Next, create model objects.

The model structure is specified in `models` subfolder.
"""

model_config = models_config.AUGREG_CONFIGS[filename.split('-')[0]]
model = models.VisionTransformer(num_classes=ds_info.features['label'].num_classes, **model_config)


"""
Next, load checkpoint from the `filename`.
"""

# path = f'gs://vit_models/augreg/{filename}.npz'

# # Non-default checkpoints need to be loaded from local files.
# if not tf.io.gfile.exists(f'{filename}.npz'):
#   tf.io.gfile.copy(path, f'{filename}.npz')
# params = checkpoint.load(path)

import numpy as np
import flax
from flax.training import checkpoints
from vit_jax.checkpoint import recover_tree, _fix_groupnorm
from packaging import version

ckpt_dict = np.load(f'{filename}.npz', allow_pickle=False)
keys, values = zip(*list(ckpt_dict.items()))
params = checkpoints.convert_pre_linen(recover_tree(keys, values))

if isinstance(params, flax.core.FrozenDict):
  params = params.unfreeze()
if version.parse(flax.__version__) >= version.parse('0.3.6'):
  params = _fix_groupnorm(params)


"""
Define some data preprocessing functions.
"""


def pp(img, sz):
  """Simple image preprocessing."""
  img = tf.cast(img, float) / 255.0
  img = tf.image.resize(img, [sz, sz])
  return img

# sanity check to show what image we are inferring.
plt.imsave('input_flax.png', pp(d['image'], resolution).numpy())


"""
Run the JAX Graph for inference
"""

# Inference on batch with single example.
logits, = model.apply({'params': params}, pp(d['image'], resolution).numpy()[None], train=False)



"""
Save the scores as a PNG file
"""

# Plot logits (you can use tf.nn.softmax() to show probabilities instead).
plt.figure(figsize=(10, 4))
plt.bar(list(map(ds_info.features['label'].int2str, range(len(logits)))), logits)
plt.xticks(rotation=90);

plt.savefig('scores_flax.png')