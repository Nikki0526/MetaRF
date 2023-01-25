import csv
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as keras_backend
from sklearn import metrics
from pandas.core.frame import DataFrame
import sys
import time
import numpy as np
import kennard_stone as ks
import random
from tqdm import tqdm


BASE_DIR = os.getcwd() # The current working directory
input_path = os.path.join(BASE_DIR, 'data/Data_After_Preprocessing_Buchwald_Hartwig_HTE.csv') # Relative path
output_path = os.path.join(BASE_DIR, 'model/model_trained.h5') 

data_all = pd.read_csv(input_path) # Read the csv file with relative path
reagent = 'additive_number'
dft_num = 120 # Number of the original reaction encoding

list_train = [1,2,3,4]
list_val = [5]
list_test = [6,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]

train_all = data_all[data_all[reagent].isin(list_train)].sample(frac=1)
test_all = data_all[data_all[reagent].isin(list_test)]

class Model(keras.Model):
    def __init__(self):
        super().__init__()
        self.hidden1 = keras.layers.Dense(40, input_shape=(feature_num,))
        self.hidden2 = keras.layers.Dense(40)
        self.out = keras.layers.Dense(1)
        
    def forward(self, x):
        x = keras.activations.relu(self.hidden1(x))
        x = keras.activations.relu(self.hidden2(x))
        x = self.out(x)
        return x

    def call(self, x):
        x = keras.activations.relu(self.hidden1(x))
        x = keras.activations.relu(self.hidden2(x))
        x = self.out(x)
        return x

def loss_function(pred_y, y):
  return keras_backend.mean(keras.losses.mean_squared_error(y, pred_y))

def np_to_tensor(list_of_numpy_objs):
    return (tf.convert_to_tensor(obj) for obj in list_of_numpy_objs)
    

def compute_loss(model, x, y, loss_fn=loss_function):
    logits = model.forward(x)
    mse = loss_fn(y, logits)
    return mse, logits


def compute_gradients(model, x, y, loss_fn=loss_function):
    with tf.GradientTape() as tape:
        loss, _ = compute_loss(model, x, y, loss_fn)
    return tape.gradient(loss, model.trainable_variables), loss


def apply_gradients(optimizer, gradients, variables):
    optimizer.apply_gradients(zip(gradients, variables))

    
def train_batch(x, y, model, optimizer):
    tensor_x, tensor_y = np_to_tensor((x, y))
    gradients, loss = compute_gradients(model, tensor_x, tensor_y)
    apply_gradients(optimizer, gradients, model.trainable_variables)
    return loss

class YieldGenerator():

    def __init__(self, K=10, list_used = list_train, data = data_all):
        self.K = K
        self.data = data_all
        self.list_used = list_used
    

    def batch(self, x = None, force_new=False):
        number = random.choice(list_train)
        data_used = train_all[train_all[reagent] == number]
        data_used = data_used.sample(self.K)
        x = data_used.iloc[:,0:feature_num].values
        y = data_used[['yield']].values
        return x, y

def train_maml(model, epochs, dataset, lr_inner=0.00001, batch_size=1, log_steps=1000):

    optimizer = keras.optimizers.Adam()
    
    # Step 2: instead of checking for convergence, we train for a number
    # of epochs
    for _ in range(epochs):
        total_loss = 0
        losses = []
        losses_test = []
        start = time.time()
        # Step 3 and 4
        for i, t in enumerate(random.sample(dataset, len(dataset))):
            x, y = np_to_tensor(t.batch())
            model.forward(x)  # run forward pass to initialize weights
            with tf.GradientTape() as test_tape:
                # test_tape.watch(model.trainable_variables)
                # Step 5
                with tf.GradientTape() as train_tape:
                    train_loss, _ = compute_loss(model, x, y)
                # Step 6
                gradients = train_tape.gradient(train_loss, model.trainable_variables)
                k = 0
                model_copy = copy_model(model, x)
                for j in range(len(model_copy.layers)):
                    model_copy.layers[j].kernel = tf.subtract(model.layers[j].kernel,
                                tf.multiply(lr_inner, gradients[k]))
                    model_copy.layers[j].bias = tf.subtract(model.layers[j].bias,
                                tf.multiply(lr_inner, gradients[k+1]))
                    k += 2
                # Step 8
                test_loss, logits = compute_loss(model_copy, x, y)

            # Step 8
            gradients = test_tape.gradient(test_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            # Logs
            total_loss += test_loss
            loss = total_loss / (i+1.0)
            losses.append(loss)
            

def copy_model(model, x):

    copied_model = Model()
    copied_model.forward(tf.convert_to_tensor(x))
    
    copied_model.set_weights(model.get_weights())
    return copied_model

def generate_dataset(K, train_size=60, test_size=10):

    def _generate_dataset_train(size):
        return [YieldGenerator(K=K,list_used=list_train,data=data_all) for _ in range(size)]

    return _generate_dataset_train(train_size) 

def eval_all_ks_tune_num_val(model,tune_num=10):

  y_test_all = []
  y_pred_all = []

  for i in range(len(list_val)):
    y_test,y_pred,df1,df2,df_plot = eval_ks_tune_num(model,number = list_val[i],tune_num=tune_num)

    y_test_all += y_test
    y_pred_all += y_pred

  r2 = metrics.r2_score(y_test_all, y_pred_all)
  rmse = np.sqrt(metrics.mean_squared_error(y_test_all, y_pred_all))

  return r2,rmse

def eval_ks_tune_num(model, data = data_all, num_steps=(0, 1, 2,3,4,5,6,7,8,9,10), lr=0.00001, number = 16,tune_num=10):

    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    df_plot = pd.DataFrame()
    tmp = data_all_copy[data_all_copy[reagent] == number]

    data_used = tmp.reset_index().drop(columns=['index'])
    data_used_plot = tmp.reset_index()

    X = data_used.iloc[:,feature_num+2:feature_num+4]
    y = data_used.iloc[:,feature_num+1:feature_num+2]

    all_num = len(data_used)

    X_train_, X_test_, y_train_, y_test_ = ks.train_test_split(X, y, test_size = 1 - tune_num/all_num)
    top_k_idx = X_train_.index.tolist()

    data_sampled = data_used.loc[top_k_idx]
    data_sampled_plot = data_used_plot.loc[top_k_idx].set_index(["index"])

    # batch used for training
    data_minus = data_used.append(data_sampled).drop_duplicates(keep=False)

    x_test = data_minus.iloc[:,0:feature_num].values
    y_test = data_minus[['yield']].values.flatten().tolist()
    
    x = data_sampled.iloc[:,0:feature_num].values
    y = data_sampled[['yield']].values

    
    # copy model so we can use the same model multiple times
    copied_model = copy_model(model, x)
    
    # use SGD for this part of training as described in the paper
    optimizer = keras.optimizers.SGD(learning_rate=lr)
    
    # run training and log fit results
    fit_res,best_res = evaluation(copied_model, optimizer, x, y, x_test, y_test, num_steps)
    
    #y_pred = np.clip(best_res[0][1].numpy().flatten(), 0, 100)
    #y_pred = best_res[0][1].numpy().flatten()
    y_pred = fit_res[1][1].numpy().flatten().tolist()

    
    df1 = data_sampled
    df2 = data_minus
    df_plot = data_sampled_plot

    return y_test,y_pred,df1,df2,df_plot

def evaluation(model, optimizer, x, y, x_test, y_test, num_steps=(0, 1, 2,3,4,5,6,7,8,9,10)):

    fit_res = []
    best_res = []
    min_loss = 1000000000
    
    tensor_x_test, tensor_y_test = np_to_tensor((x_test, y_test))
    
    # If 0 in fits we log the loss before any training
    if 0 in num_steps:
        loss, logits = compute_loss(model, tensor_x_test, tensor_y_test)
        fit_res.append((0, logits, loss))
        
    for step in range(1, np.max(num_steps) + 1):
        train_batch(x, y, model, optimizer)
        loss, logits = compute_loss(model, tensor_x_test, tensor_y_test)
        if step in num_steps:
            fit_res.append(
                (
                    step, 
                    logits,
                    loss
                )
            )

        if loss < min_loss:
          best_res = []
          best_res.append(
                (
                    step, 
                    logits,
                    loss
                )
            )
          min_loss = loss
    
    return fit_res,best_res

val_times = 3
feature_num = 320
data_all_copy = data_all.copy(deep=True)
result = pd.DataFrame()
train_ds = generate_dataset(K=20,train_size=80)

# model selection with validation set
model_list = []
for i in range(val_times): 
  maml = Model()  
  train_maml(maml, 20, train_ds, log_steps=10)
  model_list.append(maml)
  r2,rmse = eval_all_ks_tune_num_val(maml,tune_num = 5)

  c={"r2_val" : [r2]}
  frame = pd.DataFrame(c)
  frame['val_times'] = i
  result = result.append(frame)

tmp = result[result['r2_val']==result['r2_val'].max()]
model_list[tmp.at[0,'val_times']].save_weights(output_path) # Save the model