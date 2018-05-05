from hyperas.distributions import uniform
from hyperas import optim

from matplotlib import *

import numpy as np
import pandas as pd
from skimage import io, color, exposure, transform

import gc
import glob
import h5py
import time
import skimage

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras import backend as K

K.set_image_data_format('channels_first')

NUM_CLASSES = 43
IMG_SIZE = 48


def preprocess_img(img):
    hsv = color.rgb2hsv(img)
    hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
    img = color.hsv2rgb(hsv)

    min_side = min(img.shape[:-1])
    centre = img.shape[0] // 2, img.shape[1] // 2
    img = img[centre[0] - min_side // 2:centre[0] + min_side // 2,
          centre[1] - min_side // 2:centre[1] + min_side // 2,
          :]

    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.rollaxis(img, -1)

    return img


def get_class(img_path):
    return int(img_path.split('/')[-2])


def lr_schedule(epoch):
    return lr * (0.1 ** int(epoch / 10))


def cnn_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(3, IMG_SIZE, IMG_SIZE), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout({{uniform(0,1)}}))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout({{uniform(0,1)}}))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout({{uniform(0,1)}}))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout({{uniform(0,1)}}))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    return model

def create_model(x_train, y_train, x_test, y_test):
   start = time.time()
   model = cnn_model()
   lr = 0.01
   sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
   model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

   batch_size = 32
   nb_epoch = 1

   model.fit(X, Y, batch_size=batch_size, epochs=nb_epoch, validation_split=0.2, shuffle=True,
             callbacks=[LearningRateScheduler(lr_schedule), ModelCheckpoint('model.h5', save_best_only=True)])

   stop = time.time()
   print('\n\n time for training model is {}\n\n'.format(stop - start))

   y_pred = model.predict_classes(X_test)
   acc = np.sum(y_pred == y_test)/np.size(y_pred)
   print("Test accuracy = {}".format(acc))


def get_train_data():
    # preprocess data
    start = time.time()
    try:
        with  h5py.File('X.h5') as hf:
            x_train, y_train = hf['imgs'][:], hf['labels'][:]
        print("Loaded images from X.h5")

    except (IOError,OSError, KeyError):
        print("Error in reading X.h5. Processing all images...")
        root_dir = 'GTSRB/Final_Training/Images/'
        imgs = []
        labels = []

        all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))
        np.random.shuffle(all_img_paths)
        for img_path in all_img_paths:
            try:
                img = preprocess_img(io.imread(img_path))
                label = get_class(img_path)
                imgs.append(img)
                labels.append(label)

                if len(imgs)%1000 == 0:
                    print("Processed {}/{}".format(len(imgs), len(all_img_paths)))
            except (IOError, OSError):
                print('missed', img_path)
                pass

        x_train = np.array(imgs, dtype='float32')
        y_train = np.eye(NUM_CLASSES, dtype='uint8')[labels]

        with h5py.File('X.h5','w') as hf:
            hf.create_dataset('imgs', data=X)
            hf.create_dataset('labels', data=Y)
    stop = time.time()
    print('\n\n### time for preprocess data is {}\n\n'.format(stop - start))

    return x_train, y_train

def get_test_data():
    # test data
    test = pd.read_csv('GT-final_test.csv', sep=',',error_bad_lines=False, names=['Filename', '1', '2', '3', '4', '5', '6', 'ClassId'], dtype={'Filename': str})

    x_test = []
    y_test = []

    for file_name, class_id  in zip(list(test['Filename']), list(test['ClassId'])):
        img_path = os.path.join('GTSRB/Final_Test/Images/', file_name + '.ppm')
        x_test.append(preprocess_img(io.imread(img_path)))
        y_test.append(class_id)

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    return x_test, y_test

x_train, y_train = get_train_data()
x_test, y_test = get_test_data()

data = x_train, y_train, x_test, y_test
best_run, best_model = optim.minimize(model=create_model,
                                        data=data,
                                        algo=tpe.suggest,
                                        max_evals=10,
                                        trials=Trials())
