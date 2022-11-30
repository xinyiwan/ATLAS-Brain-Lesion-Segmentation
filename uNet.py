import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, SpatialDropout2D,Conv2DTranspose,Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import os
import glob
from random import shuffle
from tensorflow.keras import backend as K
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

def conv_block(x, n_base, batch_normalization):
    
    x = Conv2D(filters=n_base, kernel_size=(3,3), 
                        strides=(1,1),padding='same')(x)
    if (batch_normalization):
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters=n_base, kernel_size=(3,3), 
                        strides=(1,1),padding='same')(x)
    if (batch_normalization):
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x

def downsample_block(x, n_base, batch_normalization, dropout):
    f = conv_block(x, n_base, batch_normalization)  ## conv1, 
    p = layers.MaxPool2D(pool_size = (2,2))(f)
    if(dropout):
        p = layers.Dropout(0.2)(p)                  ## pool1,
        
    return f, p

def upsample_block(x, f, n_base, batch_normalization, dropout):
    
    x = Conv2DTranspose(filters=n_base, kernel_size=(2,2), 
                         strides=(2,2),padding='same')(x)
    x = Concatenate()([x,f])
    if(dropout):
        x = layers.Dropout(0.2)(x)
    x = conv_block(x, n_base, batch_normalization)
        
    return x

def unet_weight(img_w, img_h, img_ch, n_base, LR, batch_normalization, dropout, weight_strength):
    
    ## Encoder part
    model = Sequential()
    image = layers.Input((img_w, img_h, img_ch),name='inputs')
    weight = layers.Input((img_w, img_h, img_ch),name='loss_weights')
    
    
    f1, p1 = downsample_block(image, n_base, batch_normalization, dropout)
    f2, p2 = downsample_block(p1, n_base*2, batch_normalization, dropout)
    f3, p3 = downsample_block(p2, n_base*4, batch_normalization, dropout)
    f4, p4 = downsample_block(p3, n_base*8, batch_normalization, dropout)
    
    
    ## Bottleneck
    bottleneck = conv_block(p4, n_base*16, batch_normalization)
    
    ## Decoder part
    p5 = upsample_block(bottleneck, f4, n_base*8, batch_normalization, dropout)
    p6 = upsample_block(p5, f3, n_base*4, batch_normalization, dropout)
    p7 = upsample_block(p6, f2, n_base*2, batch_normalization, dropout)
    p8 = upsample_block(p7, f1, n_base, batch_normalization, dropout)

    
    ## 1 Convo layer
    p9 = Conv2D(filters=1, kernel_size=(1,1), 
                            padding='same')(p8)
    outputs = Activation('sigmoid')(p9)
    
    model = tf.keras.Model(inputs=[image,weight], outputs=outputs,name="U-Net_wp")
    model.summary()
    model.compile(loss = weighted_loss(weight,weight_strength),
            optimizer = Adam(learning_rate = LR),
            metrics = [dice_coef, tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    
    return model

def unet(img_w, img_h, img_ch, n_base, batch_normalization, dropout):
    
    ## Encoder part
    model = Sequential()
    inputs = layers.Input((img_w, img_h, img_ch))
    # weight = layers.Input((img_w, img_h, img_ch))
    # inputs= layers.concatenate([image,weight],axis=-1)

    # inputs = layers.Input((img_w, img_h, img_ch))
    
    f1, p1 = downsample_block(inputs, n_base, batch_normalization, dropout)
    f2, p2 = downsample_block(p1, n_base*2, batch_normalization, dropout)
    f3, p3 = downsample_block(p2, n_base*4, batch_normalization, dropout)
    f4, p4 = downsample_block(p3, n_base*8, batch_normalization, dropout)
    
    
    ## Bottleneck
    bottleneck = conv_block(p4, n_base*16, batch_normalization)
    
    ## Decoder part
    p5 = upsample_block(bottleneck, f4, n_base*8, batch_normalization, dropout)
    p6 = upsample_block(p5, f3, n_base*4, batch_normalization, dropout)
    p7 = upsample_block(p6, f2, n_base*2, batch_normalization, dropout)
    p8 = upsample_block(p7, f1, n_base, batch_normalization, dropout)

    
    ## 1 Convo layer
    p9 = Conv2D(filters=1, kernel_size=(1,1), 
                            padding='same')(p8)
    outputs = Activation('sigmoid')(p9)
    

    model = tf.keras.Model(inputs=inputs, outputs=outputs,name="U-Net")
    model.summary()
    
    return model


## Define metrics
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 0.0001) / (K.sum(y_true_f) + K.sum(y_pred_f) + 0.0001)

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def weighted_loss(weight_map, weight_strength): 
    def weighted_dice_loss(y_true, y_pred):
        y_true_f = K.flatten(y_true) 
        y_pred_f = K.flatten(y_pred)
        weight_f = K.flatten(weight_map) 
        weight_f = weight_f * weight_strength 
        weight_f = 1/(weight_f + 1)
        weighted_intersection = K.sum(weight_f * (y_true_f * y_pred_f))
        return -(2. * weighted_intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())
    return weighted_dice_loss


# def weighted_loss(weight_map, weight_strength):
#     def weighted_dice_loss(y_true, y_pred):
#         y_true_f = K.flatten(y_true)
#         y_pred_f = K.flatten(y_pred)
#         weight_f = K.flatten(weight_map)
#         weight_f = weight_f * (1 - weight_strength) # e.g. weight_strength = 0.4 boundary = 0.4
#         weight_f = 1 - weight_f 
#         weighted_intersection = K.sum(weight_f*(y_true_f*y_pred_f))
#         return-(2.*weighted_intersection+K.epsilon()) / (K.sum(y_true_f)+K.sum(y_pred_f)+K.epsilon())
#     return weighted_dice_loss

def sigmoidal_decay(e, start=0, end=100, lr_start=1e-3, lr_end=1e-5):
    if e < start:
        return lr_start
    
    if e > end:
        return lr_end
    
    middle = (start + end) / 2
    s = lambda x: 1 / (1 + np.exp(-x))
    
    return s(13 * (-e + middle) / np.abs(end - start)) * np.abs(lr_start - lr_end) + lr_end
