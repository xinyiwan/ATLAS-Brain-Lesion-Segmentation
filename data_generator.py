import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from numpy import concatenate


def generator_with_weightmap(x_train, y_train, weight_train, batch_size, aug):
    while True:
               
        for ind in (range(0, len(x_train), batch_size)):
            
            batch_img = x_train[ind:ind+batch_size]
            batch_weightmap = weight_train[ind:ind+batch_size]
            batch_label = y_train[ind:ind+batch_size]
            
            # Sanity check assures batch size always satisfied
            # by repeating the last 2-3 images at last batch.
            if len(batch_img) == batch_size:
                pass
            else:
                for tmp in range(batch_size - len(batch_img)):
                    batch_img = np.append(batch_img, np.expand_dims(batch_img[-1],axis=0), axis = 0)
                    batch_weightmap = np.append(batch_weightmap, np.expand_dims(batch_weightmap[-1],axis=0), axis = 0)
                    batch_label = np.append(batch_label, np.expand_dims(batch_label[-1], axis=0), axis = 0)
        
            backgound_value = x_train.min()
            if aug:
                data_gen_args = dict(rotation_range=10.,
                                        width_shift_range=0.1,
                                        height_shift_range=0.1,
                                        cval = backgound_value,
                                        zoom_range=0.2,
                                        horizontal_flip = True)
            else:
                data_gen_args = dict()
            
            image_datagen = ImageDataGenerator(**data_gen_args)
            weights_datagen = ImageDataGenerator(**data_gen_args)
            mask_datagen = ImageDataGenerator(**data_gen_args)
            
            image_generator = image_datagen.flow(batch_img, shuffle=False,
                                                 batch_size=batch_size,
                                                 seed=1)

            weights_generator = weights_datagen.flow(batch_weightmap, shuffle=False,
                                                 batch_size=batch_size,
                                                 seed=1)
            
            mask_generator = mask_datagen.flow(batch_label, shuffle=False,
                                               batch_size=batch_size,
                                               seed=1)
            
            image = image_generator.next()
            weight = weights_generator.next()
            label = mask_generator.next()
            
            
            yield ([image,weight], label)



def generator(x_train, y_train, batch_size):
    while True:
               
        for ind in (range(0, len(x_train), batch_size)):
            
            batch_img = x_train[ind:ind+batch_size]
            batch_label = y_train[ind:ind+batch_size]
            
            # Sanity check assures batch size always satisfied
            # by repeating the last 2-3 images at last batch.
            if len(batch_img) == batch_size:
                pass
            else:
                for tmp in range(batch_size - len(batch_img)):
                    batch_img = np.append(batch_img, np.expand_dims(batch_img[-1],axis=0), axis = 0)
                    batch_label = np.append(batch_label, np.expand_dims(batch_label[-1], axis=0), axis = 0)
        
            backgound_value = x_train.min()
            data_gen_args = dict(rotation_range=10.,
                                     width_shift_range=0.1,
                                     height_shift_range=0.1,
                                     cval = backgound_value,
                                     zoom_range=0.2,
                                     horizontal_flip = True)
            
            image_datagen = ImageDataGenerator(**data_gen_args)
            mask_datagen = ImageDataGenerator(**data_gen_args)
            
            image_generator = image_datagen.flow(batch_img, shuffle=False,
                                                 batch_size=batch_size,
                                                 seed=1)
            
            mask_generator = mask_datagen.flow(batch_label, shuffle=False,
                                               batch_size=batch_size,
                                               seed=1)
            
            image = image_generator.next()
            label = mask_generator.next()
            
            
            yield (image, label)