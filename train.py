from matplotlib import *

import numpy as np
import pandas as pd
from skimage import io, color, exposure, transform
import gc
import glob
import h5py
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
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    return model


try:
    with  h5py.File('X.h5') as hf: 
        X, Y = hf['imgs'][:], hf['labels'][:]
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

    X = np.array(imgs, dtype='float32')
    Y = np.eye(NUM_CLASSES, dtype='uint8')[labels]

    with h5py.File('X.h5','w') as hf:
        hf.create_dataset('imgs', data=X)
        hf.create_dataset('labels', data=Y)

model = cnn_model()

#train data
lr = 0.01
sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

batch_size = 32
nb_epoch = 20

model.fit(X, Y, batch_size=batch_size, epochs=nb_epoch, validation_split=0.2, shuffle=True,
          callbacks=[LearningRateScheduler(lr_schedule), ModelCheckpoint('model.h5', save_best_only=True)])

# test data
test = pd.read_csv('GT-final_test.csv', sep=',',error_bad_lines=False, names=['Filename', '1', '2', '3', '4', '5', '6', 'ClassId'], dtype={'Filename': str})

try:

    X_test = []
    y_test = []

    for file_name, class_id  in zip(list(test['Filename']), list(test['ClassId'])):
        img_path = os.path.join('GTSRB/Final_Test/Images/', file_name + '.ppm')
        X_test.append(preprocess_img(io.imread(img_path)))
        y_test.append(class_id)
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    y_pred = model.predict_classes(X_test)
    acc = np.sum(y_pred == y_test)/np.size(y_pred)
    print("Test accuracy = {}".format(acc))
finally:
    print('FINISHED')

