import tensorflow as tf
tf.config.gpu.set_per_process_memory_fraction(0.3)
tf.config.gpu.set_per_process_memory_growth(True)

import time
import numpy as np

from tensorflow.python.keras.utils import generic_utils
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

from general_utils import setup_logging
from data_utils import load_data, get_nb_patch, gen_batch, get_disc_batch, plot_generated_batch
from models import load_model, DCGAN


def l1_loss(y_true, y_pred):
    return K.sum(K.abs(y_pred - y_true), axis=-1)


## parameters:
batch_size = 4
nb_epoch = 100
n_batch_per_epoch = 50
model_name = 'CNN'
generator = 'upsampling'
image_data_format = 'channels_last'
img_dim = 256
patch_size = [128, 128]
bn_mode = 2
label_smoothing = False
label_flipping = 0
data_folder = '/DL_course_data/Lab6/data_pix2pix/data/processed/'
dset = 'chest_xray'
use_mbd = False
do_plot = False
logging_dir = './pix2pix/logging_dir_pix2pix/'

epoch_size = n_batch_per_epoch * batch_size

# Setup environment (logging directory etc)
setup_logging(model_name, logging_dir=logging_dir)

# Load and rescale data
X_full_train, X_sketch_train, X_full_val, X_sketch_val = load_data(data_folder, dset, image_data_format)
img_dim = X_full_train.shape[-3:]

# Get the number of non overlapping patch and the size of input image to the discriminator
nb_patch, img_dim_disc = get_nb_patch(img_dim, patch_size, image_data_format)

try:

    # Create optimizers
    opt_dcgan = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    opt_discriminator = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    # Load generator model
    generator_model = load_model("generator_unet_%s" % generator, img_dim,
                                 nb_patch, bn_mode, use_mbd, batch_size, do_plot)
    # Load discriminator model
    discriminator_model = load_model("DCGAN_discriminator", img_dim_disc, nb_patch,
                                     bn_mode, use_mbd, batch_size, do_plot)

    # Compile generator model
    generator_model.compile(loss='mae', optimizer=opt_discriminator)
    discriminator_model.trainable = False

    # Define DCGAN model
    DCGAN_model = DCGAN(generator_model, discriminator_model, img_dim, patch_size, image_data_format)

    # Define loss function and loss weights
    loss = [l1_loss, 'binary_crossentropy']
    loss_weights = [1, 1]

    # Compile DCGAN model
    DCGAN_model.compile(loss=loss, loss_weights=loss_weights, optimizer=opt_dcgan)

    # Compile discriminator model
    discriminator_model.trainable = True
    discriminator_model.compile(loss='binary_crossentropy', optimizer=opt_discriminator)

    # Training
    print("Start training")
    for e in range(nb_epoch):
        # Initialize progbar and batch counter
        progbar = generic_utils.Progbar(epoch_size)
        batch_counter = 1
        start = time.time()

        for X_full_batch, X_sketch_batch in gen_batch(X_full_train, X_sketch_train, batch_size):

            # Create a batch to feed the discriminator model
            X_disc, y_disc = get_disc_batch(X_full_batch, X_sketch_batch, generator_model, batch_counter, patch_size,
                                                       image_data_format, label_smoothing=label_smoothing,
                                                       label_flipping=label_flipping)

            # Update the discriminator
            disc_loss = discriminator_model.train_on_batch(X_disc, y_disc)

            # Create a batch to feed the generator model
            X_gen_target, X_gen = next(gen_batch(X_full_train, X_sketch_train, batch_size))
            y_gen = np.zeros((X_gen.shape[0], 2), dtype=np.uint8)
            y_gen[:, 1] = 1

            # Freeze the discriminator
            discriminator_model.trainable = False
            gen_loss = DCGAN_model.train_on_batch(X_gen, [X_gen_target, y_gen])

            # Unfreeze the discriminator
            discriminator_model.trainable = True

            batch_counter += 1
            progbar.add(batch_size, values=[("D logloss", disc_loss),
                                            ("G tot", gen_loss[0]),
                                            ("G L1", gen_loss[1]),
                                            ("G logloss", gen_loss[2])])

            # Save images for visualization
            if batch_counter % (n_batch_per_epoch / 2) == 0:
                # Get new images from validation
                plot_generated_batch(X_full_batch, X_sketch_batch, generator_model,
                                                batch_size, image_data_format, "training",
                                                logging_dir)
                X_full_batch, X_sketch_batch = next(gen_batch(X_full_val, X_sketch_val, batch_size))
                plot_generated_batch(X_full_batch, X_sketch_batch, generator_model,
                                                batch_size, image_data_format, "validation",
                                                logging_dir)

            if batch_counter >= n_batch_per_epoch:
                break

        print("")
        print('Epoch %s/%s, Time: %s' % (e + 1, nb_epoch, time.time() - start))


except KeyboardInterrupt:
    pass
