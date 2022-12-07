"""

Please run

[ -d vision_transformer ] || git clone --depth=1 https://github.com/usamamuneeb/vision_transformer
mv vision_transformer/vit_jax .

"""

from vit_jax import train
from vit_jax.configs import augreg as augreg_config
import time
from absl import app, logging
import os

def main(argv):
    logging.set_verbosity(logging.INFO)  # Shows logs during training.

    # Get config for specified model.

    # Note that we can specify simply the model name (in which case the recommended
    # checkpoint for that model is taken), or it can be specified by its full
    # name.

    model_name = 'R_Ti_16'

    config = augreg_config.get_config(model_name)

    # A very small tfds dataset that only has a "train" split. We use this single
    # split both for training & evaluation by splitting it further into 90%/10%.
    config.dataset = 'cifar10'
    config.pp.train = 'train'
    config.pp.test = 'test'
    # tf_flowers only has 3670 images - so the 10% evaluation split will contain
    # 360 images. We specify batch_eval=120 so we evaluate on all but 7 of those
    # images (remainder is dropped).
    config.batch_eval = 120

    # Some more parameters that you will often want to set manually.
    # For example for VTAB we used steps={500, 2500} and lr={.001, .003, .01, .03}

    for lr in [.001, .003, .01, .03]:
        for fromscratch in [False, True]:
            config.base_lr = lr
            config.shuffle_buffer = 1000
            config.total_steps = 500 # 100
            config.warmup_steps = 10
            config.accum_steps = 0  # Not needed with R+Ti/16 model.
            config.pp['crop'] = 224

            workdir = f'./workdirs/{model_name}_{config.dataset}_fromscrath={fromscratch}_lr={config.base_lr}_uid={int(time.time())}'
            print(f"workdir: {workdir}")

            os.makedirs(workdir)
            logging.get_absl_handler().use_absl_log_file('log.txt', workdir)

            # Call main training loop. See repository and above Colab for details.
            state = train.train_and_evaluate(config, workdir, fromscratch)

if __name__ == '__main__':
    app.run(main)
