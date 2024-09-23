from pdb import set_trace
import matplotlib.pyplot as plt
import random
import time
import numpy as np
from os.path import join as pjoin
from copy import deepcopy
import pandas as pd
from sklearn.model_selection import train_test_split

from rnn import InLayer, RNNLayer, RNNBuilder, check_weights_and_gradient_shapes, eval_loss_func_rnn
from utils import eval_activation_func, show_info
from config import get_config

import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=get_config()["debug_mode"])


class NarxNetwork(RNNBuilder):
    """ Nonlinear autoregressive network with exogenous inputs """
    def __init__(self, name):
        super(self.__class__, self).__init__(name)

        n_in = 2
        self.input_layer = InLayer(n_in, self, name="input_1")
        n_dim = 10
        self.layer_1 = RNNLayer(n_dim, self, name="layer_1", act_func="sigmoid", is_output=False)
        # self.layer_2 = RNNLayer(n_dim, self, name="layer_2", act_func="linear", is_output=True)
        self.layer_2 = RNNLayer(1, self, name="layer_2", act_func="linear", is_output=True)

    def define(self):
        self.layer_1([
            (self.input_layer, [0]),
            (self.layer_2, [1]),
        ])
        self.layer_2([
            (self.layer_1, [0]),
        ])


def preprocess_maglev_data():
    dataset_path = "./datasets/MagLev"
    n_step = 5
    n_features = 2

    df_x = pd.read_csv(pjoin(dataset_path, "maglev_u.txt"), header=None)
    df_y = pd.read_csv(pjoin(dataset_path, "maglev_y.txt"), header=None)
    
    n_datapoint = len(df_y)

    # data = []
    # for idx in range(0, n_datapoint-1):
        # x_in = df_x.iloc[idx, 0]
        # y_in = df_y.iloc[idx, 0]
        # y_out = df_y.iloc[idx+1, 0]
        # data.append([x_in, y_in, y_out])
    
    # data = np.array(data)
    # X_full = data[:, :-1]
    # Y_full = data[:, -1]
    
    # X_train, X_test, y_train, y_test = train_test_split(
        # X_full, Y_full, test_size=0.2, random_state=1)
    # X_train, X_val, y_train, y_val = train_test_split(
        # X_train, y_train, test_size=0.25, random_state=1)
#    set_trace()  
    data_x = []
    data_y = []
    for idx in range(0, n_datapoint-1):
        row_x = np.zeros((n_step, n_features))
        row_y = np.zeros((n_step, 1))
        
        err = False
        for idx_step in range(0, n_step):
            x_in = df_x.iloc[idx+idx_step, 0]
            y_in = df_y.iloc[idx+idx_step, 0]
            try:
                y_out = df_y.iloc[idx+idx_step+1, 0]
            except:
                err = True
                break
#            tf.keras.models.Sequential()        
            row_x[idx_step, 0] = x_in
            row_x[idx_step, 1] = y_in
            row_y[idx_step] = y_out

        if err is True:
            break        

        data_x.append(row_x)
        data_y.append(row_y) 
    data_x = np.array(data_x) # (3996, 5, 2)
    data_y = np.array(data_y) # (3996, 5, 1)
    
    # X_train1, X_test, y_train1, y_test = train_test_split(
        # data_x, data_y, test_size=0.2, random_state=1)
    # X_train, X_val, y_train, y_val = train_test_split(
        # X_train1, y_train1, test_size=0.25, random_state=1)

    # For time series dataset we usually don't shuffle the data
    X_train = data_x[:2397, :, :]
    X_val = data_x[2397:3196, :, :]
    X_test = data_x[3196:, :, :]
    
    y_train = data_y[:2397, :, :]
    y_val = data_y[2397:3196, :, :]
    y_test = data_y[3196:, :, :]
    
    # set_trace()
    min_val = np.min(X_train)
    max_val = np.max(X_train)
    X_train = (X_train-min_val) / (max_val-min_val)
    X_val = (X_val-min_val) / (max_val-min_val)
    X_test = (X_test-min_val) / (max_val-min_val)

    min_val = np.min(y_train)
    max_val = np.max(y_train)
    y_train = (y_train-min_val) / (max_val-min_val)
    y_val = (y_val-min_val) / (max_val-min_val)                                                       
    y_test = (y_test-min_val) / (max_val-min_val)

    return (X_train, y_train, X_val, y_val, X_test, y_test)


class MaglevDataloader:
    def __init__(self, data, labels, batch_size=16, shuffle=False):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.n_inst = data.shape[0]       
        self.idx_list = list(range(self.n_inst))        

        self.on_epoch_end()

    def next(self):
        data_batch = self.data[self.idx_list[self.first_idx:self.last_idx]]
        labels_batch = self.labels[self.idx_list[self.first_idx:self.last_idx]]
        self.first_idx += self.batch_size
        self.last_idx += self.batch_size
        if self.first_idx >= self.n_inst:
            self.on_epoch_end()

#        data_batch_dict = {"input_1": data_batch}
#        labels_batch_dict = {"layer_2": labels_batch}
#        return (data_batch_dict, labels_batch_dict)
#
        return ([data_batch], [labels_batch])

    def on_epoch_end(self):
        if self.shuffle is True:
            random.shuffle(self.idx_list)
        self.first_idx = 0
        self.last_idx = self.batch_size


def main():
    (X_train, y_train, X_val, y_val, X_test, y_test) = preprocess_maglev_data()
    optimizer_params = {
        "learning_rate": 0.001, 
    }
    batch_size = 128

    maglev_train_loader = MaglevDataloader(X_train, y_train, batch_size=batch_size, shuffle=True)
    maglev_val_loader = MaglevDataloader(X_val, y_val, batch_size=batch_size)
    maglev_test_loader = MaglevDataloader(X_test, y_test, batch_size=1)

    model = NarxNetwork("NARX")
    model.define()
    model.compile(
        loss_name="MSE",
    )
#    set_trace()
    (train_loss_list, val_loss_list) = model.fit(maglev_train_loader, val_dataloader=maglev_val_loader,
#        epochs=1000,
#        epochs=1500,
         epochs=500,
        steps_per_epoch=X_train.shape[0]//batch_size,
        val_steps_per_epoch=X_val.shape[0]//batch_size,
        optimizer_params=optimizer_params,
        loss_weights={"layer_2": 1.0},
        inputs=['input_1'],
        outputs=['layer_2'],
    )

    base_list = range(len(train_loss_list))
    plt.plot(base_list, train_loss_list, color="red", label="Training")
    plt.plot(base_list, val_loss_list, color="blue", label="Validation")
    plt.legend(loc='upper left')
    plt.title("Loss")
    plt.show()

    test_loss, y_pred_all, y_true_all = model.evaluate(maglev_test_loader,steps_per_epoch=X_test.shape[0]//1, 
        loss_weights={"layer_2": 1.0},
        inputs=['input_1'],
        outputs=['layer_2'],
    )
    print("Test loss: %f" % (test_loss))

    base_list = range(len(y_pred_all))
    plt.plot(base_list, y_pred_all, color="red", label="Prediction")
    plt.plot(base_list, y_true_all, color="blue", label="Ground truth")
    plt.legend(loc='upper left')
    plt.title("Prediction and ground truth")
    plt.show()


#    set_trace() 
    
if __name__ == '__main__':
    main()
