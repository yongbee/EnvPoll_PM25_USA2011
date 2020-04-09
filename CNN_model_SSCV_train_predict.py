from keras.models import Sequential, load_model, Model
from keras.layers import Input, Dense, Activation, Conv2D, Dropout, Flatten
from keras import optimizers, utils, initializers, regularizers
import keras.backend as K
import numpy as np
import pandas as pd
import random
import os

SEED_NUM = 20000
pred_file_name = 'CNN_SSCV_prediction_Initialization_SEED' + str(SEED_NUM)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"

data_path = 'v10_170713_5x5_include_na_dataset.npz'
label_path = "v10_170713_5x5_include_na_label.npz"
x_tr_blended = np.load(data_path)['arr_0']
y_tr_blended = np.load(label_path)['arr_0']

all_locations = np.unique(x_tr_blended[:, 28*12+1])
random.seed(1000)
random_locations = random.sample(list(all_locations), len(all_locations)) 
x_tr = []; y_tr = []
for location in random_locations:
    x_tr.append(x_tr_blended[x_tr_blended[:, 28*12+1]==location])
    y_tr.append(y_tr_blended[x_tr_blended[:, 28*12+1]==location])
x_tr = np.vstack(x_tr)
y_tr =np.hstack( y_tr)

def split_train_validation(data_set, label_set, fold, k):
    """split train set and validation set"""
    quo = int(len(data_set) / k)
    
    x_train = np.delete(data_set, range(quo*fold,quo*(fold+1)), 0)
    y_train = np.delete(label_set, range(quo*fold,quo*(fold+1)), 0)
    x_test = data_set[quo*fold:quo*(fold+1)]
    y_test = label_set[quo*fold:quo*(fold+1)]
    return x_train, y_train, x_test, y_test

def norm_by_std_nan(train, val):
    mask = np.ma.array(train, mask=np.isnan(train))
    mean = np.mean(mask, 0)
    std = np.std(mask, 0)

    train = (train - mean) / std
    train = np.where(train == np.nan, 0, train)
    train = np.nan_to_num(train)

    val = (val-mean)/std
    val = np.where(val == np.nan, 0, val)
    val = np.nan_to_num(val)
    return train, val

fold = 10
epochs = 200
tr_batch_size = 100
ev_batch_size = 100
Input_width = 5
Input_height = 5
num_channels = 28
noise_std = 0.1

n_conv = 64
n_hidden = 128

for fold_num in range(fold):
    save_directory = 'prediction/SSCV/fold'+str(fold_num)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    x_train, y_train, x_val, y_val = split_train_validation(x_tr, y_tr, fold_num, fold)
    x_train, x_val = norm_by_std_nan(x_train, x_val)
    x_train = x_train.reshape(len(x_train), Input_width, Input_height, num_channels)
    x_val   = x_val.reshape(len(x_val), Input_width, Input_height, num_channels)
    
    model = Sequential([
    Conv2D(n_conv, (3,3), kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=SEED_NUM), bias_initializer=initializers.Constant(0.1), input_shape=(Input_width, Input_height, num_channels), padding='same', kernel_regularizer=regularizers.l1(0.005)),
    Activation('relu'),
    Flatten(),
    Dropout(0.5, seed=SEED_NUM),
    Dense(n_hidden, activation='elu', kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=SEED_NUM), bias_initializer=initializers.Constant(0.1), kernel_regularizer=regularizers.l1(0.005)),
    Dense(1, activation='linear')
    ])
    nadam = optimizers.Nadam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    model.compile(optimizer=nadam, loss='mse', metrics=['mae'])
    
    for epoch in range(epochs):
        """train"""
        model.fit(x_train, y_train, epochs=1, batch_size=tr_batch_size, verbose=1)
        pred = model.predict(x_val, batch_size=ev_batch_size).reshape(len(x_val),)
        val_r2 = 1 - (np.sum(np.square(y_val - pred)) / np.sum(np.square(y_val - np.mean(y_val))))
        print("epoch:{}, validation set r-squared:{}".format(epoch, val_r2))

        sv_pred = np.array(pred).reshape(len(pred), 1)
        np.savetxt(save_directory + '/' + pred_file_name + '_epoch' + str(epoch) + ".csv", sv_pred, delimiter=',')