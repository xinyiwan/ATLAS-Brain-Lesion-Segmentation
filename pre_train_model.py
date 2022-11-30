from uNet import unet_weight, weighted_loss, dice_coef_loss, dice_coef, sigmoidal_decay
from load_data import load_data_MRI
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
from sklearn.model_selection import KFold


gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)
# ----------------------------------------------------------------------------
# Set up hyperparameters
n_base = 8
batch_normalization = True
dropout = True
epochs = 100
batch_size = 8
weight_strength = 1
augmentation = False


img_w, img_h = 240, 240
img_ch = 1

# ----------------------------------------------------------------------------
# Define training label
sub_label = "pre_Unet_wmap_no_aug_e150_w_1"

# ----------------------------------------------------------------------------
# Set up check point
mc_path = "/home/k8s-group5/Model_checkpoints/" + sub_label + "_best_model.h5"
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
# Set up learning rate
# Choose either learning rate as constant or dynamic

# initial_learning_rate = 1e-4
# decay_steps = 100
# decay_rate = 0.1
# lr = ExponentialDecay(
#                 initial_learning_rate,
#                 decay_steps=decay_steps,
#                 decay_rate=decay_rate,
#                 staircase=True)

lr = 1e-4

# ----------------------------------------------------------------------------
# Initialize network
model = unet_weight(img_w, img_h, img_ch, n_base, lr, batch_normalization, dropout, weight_strength)

# ----------------------------------------------------------------------------
# Load data
images_path = '/DL_course_data/Lab3/MRI/Image'
masks_path = '/DL_course_data/Lab3/MRI/Mask'

images, masks = load_data_MRI(images_path,masks_path)
masks = masks.astype(uint8)

weight_boundary = np.zeros(masks.shape).astype(uint8)
for i in range(len(masks)):
    weight_boundary[i,:,:,0] = binary_dilation(masks[i,:,:,0], np.ones((3,3))).astype(int) - binary_erosion(masks[i,:,:,0],np.ones((3,3))).astype(int)

ShuffleSplit(n_splits=1, test_size=0.2).get_n_splits(images, masks)
train, val = next(ShuffleSplit(n_splits=1, test_size=0.2).split(images, masks)) 

# ----------------------------------------------------------------------------
# Initialize generators
train_gen = generator_with_weightmap(images[train], masks[train], weight_boundary[train], batch_size, augmentation)
val_gen = generator_with_weightmap(images[val], masks[val], np.zeros(weight_boundary[val].shape), batch_size, augmentation)

# ----------------------------------------------------------------------------
# Define log path and model path

log_dir = "/home/k8s-group5/logs/fit/" + "pretrain_" + datetime.datetime.now().strftime("%m%d-%H%M")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model_path = "/home/k8s-group5/models/" + sub_label 
save_path = model_path + datetime.datetime.now().strftime("%m%d-%H%M")

# ----------------------------------------------------------------------------
# Training and validation

model_history = model.fit(
    train_gen,
    steps_per_epoch = len(images[train])//batch_size,
    validation_data = val_gen, 
    validation_steps = len(images[val])//batch_size,
    epochs = epochs,  verbose=1,
    callbacks=[tensorboard_callback, mc, early_stopping])

model.save(save_path)


# ----------------------------------------------------------------------------
# If train with K fold
# Training and validation

# k_folds = 2
# kf = KFold(n_splits=k_folds, random_state=None, shuffle=True)
# no_ = 0
# model_path = "models/Unet_wmap_no_aug_e200_w_0.8"

# for train_index, val_index in kf.split(images):

#     x_train_kf, x_val_kf = images[train_index], images[val_index]
#     y_train_kf, y_val_kf = masks[train_index], masks[val_index]
#     weights_train_kf, weights_val_kf = weight_boundary[train_index], weight_boundary[val_index]
    
#     log_dir = "logs/fit/" + datetime.datetime.now().strftime("%m%d-%H%M") +'_kf_{}'.format(no_+1) 
#     tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
#     save_path = model_path + '_kf_{}'.format(no_+1) + datetime.datetime.now().strftime("%m%d-%H%M")
#     no_ += 1

#     model = unet_weight(img_w, img_h, img_ch, n_base, lr, batch_normalization, dropout, weight_strength)

#     train_gen = generator_with_weightmap(images[train_index], masks[train_index], weight_boundary[train_index], batch_size, augmentation)
#     val_gen = generator_with_weightmap(images[val_index], masks[val_index], weight_boundary[val_index], batch_size, augmentation)

#     model_history = model.fit_generator(train_gen,
#         steps_per_epoch = len(images[train_index])//batch_size,
#         validation_data = val_gen, 
#         validation_steps = len(images[val_index])//batch_size,
#         epochs = epochs,  verbose=1,
#         callbacks=[tensorboard_callback, mc, earlystop])

#     model.save(save_path)
