import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input
from tensorflow.keras.layers import Flatten, add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
import numpy as np
import math
from sklearn.model_selection import train_test_split 

config = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

from tensorflow.keras import layers, Model, Input

def build_vgg_functional(vgg_name, num_classes):
    inputs = Input(shape=(32, 32, 3))
    x = inputs
    for l in config[vgg_name]:
        if l == 'M':
            x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
        else:
            x = layers.Conv2D(l, kernel_size=3, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)

    x = layers.AveragePooling2D(pool_size=1, strides=1)(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs)


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    
    return lr
class VGG:
    def __init__(self, path, x_tr, y_tr, x_v, y_v, model_name,num_classes,subtract_pixel_mean = True,):
        assert len(y_tr.shape)==2
        assert len(y_v.shape)==2
        assert len(x_tr.shape)==4
        assert len(x_v.shape)==4
        
        self.path = path #"path_of_folder/" for saving and loading
        
        # if subtract pixel mean is enabled
        if subtract_pixel_mean:
            x_train_mean = np.mean(x_tr, axis=0)
            x_tr -= x_train_mean
            x_v -= x_train_mean          
        
        self.x_tr = x_tr
        self.y_tr = y_tr
        self.x_v = x_v
        self.y_v = y_v
        self.subtract_pixel_mean = subtract_pixel_mean
        self.model = build_vgg_functional(model_name, num_classes=num_classes)
        self.model.save(path)
    def train(self, batch_size = 32, epochs = 200, 
              data_augmentation = True, optimizer=None,loss='categorical_crossentropy', 
              save_best_only=False,save_weights_only=False):
              
        if optimizer is None:
              optimizer=Adam(learning_rate=lr_schedule(0))
              
        self.optimizer = optimizer
        self.model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=['acc'])
              
        # prepare callbacks for model saving and for learning rate adjustment.
        checkpoint = ModelCheckpoint(filepath=self.path,
                                     monitor='val_acc',
                                     verbose=1,
                                     save_best_only=save_best_only, save_weights_only=save_weights_only)

        lr_scheduler = LearningRateScheduler(lr_schedule)

        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                       cooldown=0,
                                       patience=5,
                                       min_lr=0.5e-6)

        callbacks = [checkpoint, lr_reducer, lr_scheduler]

        # run training, with or without data augmentation.
        if not data_augmentation:
            print('Not using data augmentation.')
            self.model.fit(self.x_tr, self.y_tr,
                      batch_size=batch_size,
                      epochs=epochs,
                      validation_data=(self.x_v, self.y_v),
                      shuffle=True,
                      callbacks=callbacks)
        else:
            print('Using real-time data augmentation.')
            # this will do preprocessing and realtime data augmentation:
            datagen = ImageDataGenerator(
                # set input mean to 0 over the dataset
                featurewise_center=False,
                # set each sample mean to 0
                samplewise_center=False,
                # divide inputs by std of dataset
                featurewise_std_normalization=False,
                # divide each input by its std
                samplewise_std_normalization=False,
                # apply ZCA whitening
                zca_whitening=False,
                # randomly rotate images in the range (deg 0 to 180)
                rotation_range=0,
                # randomly shift images horizontally
                width_shift_range=0.1,
                # randomly shift images vertically
                height_shift_range=0.1,
                # randomly flip images
                horizontal_flip=True,
                # randomly flip images
                vertical_flip=False)

            # compute quantities required for featurewise normalization
            # (std, mean, and principal components if ZCA whitening is applied).
            datagen.fit(self.x_tr)

            steps_per_epoch =  math.ceil(len(self.x_tr) / batch_size)
            # fit the model on the batches generated by datagen.flow().
            self.model.fit(x=datagen.flow(self.x_tr, self.y_tr, batch_size=batch_size),
                      verbose=0,
                      epochs=epochs,
                      validation_data=(self.x_v, self.y_v),
                      steps_per_epoch=steps_per_epoch,
                      callbacks=callbacks)
        return  