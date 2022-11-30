from uNet import unet_weight, weighted_loss, dice_coef_loss, dice_coef, sigmoidal_decay
from load_data import load_data
from numpy import uint8
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from scipy.ndimage import binary_dilation, binary_erosion
from sklearn.model_selection import ShuffleSplit 
from data_generator import generator_with_weightmap
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping
import datetime
import os
from tensorboard import program
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from sklearn.model_selection import KFold


gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)
# ----------------------------------------------------------------------------
# Set up hyperparameters
n_base = 8
batch_normalization = True
dropout = True
epochs = 150
batch_size = 8
weight_strength = 1
augmentation = False


img_w, img_h = 240, 240
img_ch = 1

# ----------------------------------------------------------------------------
# Define training label

sub_label = "TL_Unet_wmap_n8_with_aug_e150_w_0_8_lr_5_"
# ----------------------------------------------------------------------------
# Set up learning rate
# initial_learning_rate = 1e-3
# decay_steps = 100
# decay_rate = 0.1
# lr = ExponentialDecay(
#                 initial_learning_rate,
#                 decay_steps=decay_steps,
#                 decay_rate=decay_rate,
#                 staircase=True)
lr = 1e-5

# ----------------------------------------------------------------------------
# Load pretrained model with MRI data 
model = unet_weight(img_w, img_h, img_ch, n_base, lr, batch_normalization, dropout, weight_strength)
model.load_weights('/home/xinyiwan/mri/tf/models/Unet_wmap_no_aug_e100_w_11117-1457_pretrain/variables/variables')

# ----------------------------------------------------------------------------
# Unfreeze the appointed layers
for layer in model.layers:
    if layer.name in ['conv2d_18', 'inputs','activation_18','loss_weights']:
        layer.trainable = True
    else:
        layer.trainable = False

# ----------------------------------------------------------------------------
# Load data
images_path = '/home/xinyiwan/mri/Images_cv'
masks_path = '/home/xinyiwan/mri/Masks_cv'

images, masks = load_data(img_w, img_h, images_path,masks_path)
# masks = masks.astype(uint8)

weight_boundary = np.zeros(masks.shape).astype(uint8)
for i in range(len(masks)):
    weight_boundary[i,:,:,0] = binary_dilation(masks[i,:,:,0], np.ones((3,3))).astype(uint8) - binary_erosion(masks[i,:,:,0],np.ones((3,3))).astype(uint8)

# ShuffleSplit(n_splits=1, test_size=0.2).get_n_splits(images, masks)
# train, val = next(ShuffleSplit(n_splits=1, test_size=0.2).split(images, masks)) 
# Split data into 2 categroies.                                                                     
images_train, images_val = np.split(images,[int(0.8 * len(images))])
masks_train, masks_val = np.split(masks,[int(0.8 * len(masks))])
weight_train, weight_val = np.split(weight_boundary,[int(0.8 * len(weight_boundary))])


# ----------------------------------------------------------------------------
# Initialize generators
train_gen = generator_with_weightmap(images_train, masks_train, weight_train, batch_size, augmentation)
val_gen = generator_with_weightmap(images_val, masks_val, np.zeros(weight_val.shape), batch_size, augmentation)

# ----------------------------------------------------------------------------
# Define log path and model path

log_dir = "/home/xinyiwan/mri/tf/logs/fit/" + sub_label + datetime.datetime.now().strftime("%m%d-%H%M")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model_path = "/home/xinyiwan/mri/tf/models/" + sub_label
save_path = model_path + datetime.datetime.now().strftime("%m%d-%H%M")

# ----------------------------------------------------------------------------
# Set up check point
mc_path = "/home/xinyiwan/mri/tf/Model_checkpoints/" + sub_label + datetime.datetime.now().strftime("%m%d-%H%M") + "_mc.tf"
mc = ModelCheckpoint(
        filepath= mc_path,
        save_weights_only= True,
        monitor='val_loss', 
        save_best_only=True)

early_stopping = EarlyStopping(
                              patience=5,
                              min_delta=0.001,                               
                              monitor="val_loss",
                              restore_best_weights=True
                              )
# ----------------------------------------------------------------------------
# Training and validation

model_history = model.fit_generator(train_gen,
    steps_per_epoch = len(images_train)//batch_size,
    validation_data = val_gen, 
    validation_steps = len(images_val)//batch_size,
    epochs = epochs,  verbose=1,
    callbacks=[tensorboard_callback, mc])

model.save(save_path)
np.save(os.path.join(save_path,'model_history.npy'),model_history.history)

