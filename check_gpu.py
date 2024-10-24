import tensorflow as tf
tf_gpu = tf.config.list_physical_devices("GPU")
print(f"tensorflow gpu: {tf_gpu}")

import torch
torch_gpu = torch.mps.is_available()
print(f"torch gpu: {torch_gpu}")

from screeninfo import get_monitors
w, h = [(m.width, m.height) for m in get_monitors() if m.is_primary is True][0]
print(f'[screen] width: {w}, height: {h}')