import timm
import torch
import tensorflow as tf # only needed to get dataset
import tensorflow_datasets as tfds # only needed to get dataset
from matplotlib import pyplot as plt

# For available model names, see here:
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer_hybrid.py


"""
The `filename` contains the model, training dataset and fine tuning dataset

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

The model structure is specified in the `timm` package.
"""

timm_model = timm.create_model('vit_small_r26_s32_384', num_classes=ds_info.features['label'].num_classes)


"""
Next, load checkpoint from the `filename`.
"""

path = f'gs://vit_models/augreg/{filename}.npz'

# Non-default checkpoints need to be loaded from local files.
if not tf.io.gfile.exists(f'{filename}.npz'):
  tf.io.gfile.copy(path, f'{filename}.npz')


timm.models.load_checkpoint(timm_model, f'{filename}.npz')


"""
Define some data preprocessing functions.
"""


def pp(img, sz):
  """Simple image preprocessing."""
  img = tf.cast(img, float) / 255.0
  img = tf.image.resize(img, [sz, sz])
  return img

def pp_torch(img, sz):
  """Simple image preprocessing for PyTorch."""
  img = pp(img, sz)
  img = img.numpy().transpose([2, 0, 1])  # PyTorch expects NCHW format.
  return torch.tensor(img[None])

# sanity check to show what image we are inferring.
plt.imsave('input_timm.png', pp(d['image'], resolution).numpy())
plt.imsave('input_timm_1.png', pp(d['image'], resolution).numpy())

"""
Run the PyTorch Graph for inference
"""

with torch.no_grad():
  logits, = timm_model(pp_torch(d['image'], resolution)).detach().numpy()

"""
Save the scores as a PNG file
"""

# Same results as above (since we loaded the same checkpoint).
plt.figure(figsize=(10, 4))
plt.bar(list(map(ds_info.features['label'].int2str, range(len(logits)))), logits)
plt.xticks(rotation=90)

plt.savefig('scores_timm.png')
