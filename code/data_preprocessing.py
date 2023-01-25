import os
import csv
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

# Parameters (can be changed according to experiment setting)

BASE_DIR = os.getcwd() # The current working directory
input_path = os.path.join(BASE_DIR, 'data/Original_Data_Buchwald_Hartwig_HTE.csv') # Relative path
output_path = os.path.join(BASE_DIR, 'data/Data_After_Preprocessing_Buchwald_Hartwig_HTE.csv')

data = pd.read_csv(input_path) 
reagent = 'additive_number' 

list_train = [1,2,3,4]
list_test = [5,6,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]

dft_num = 120 # Number of the original reaction encoding

# Random forest
    
train = data[data[reagent].isin(list_train)].sample(frac=1)
test = data[data[reagent].isin(list_test)]
  
X_test = test.iloc[:,0:dft_num]
X_train = train.iloc[:,0:dft_num]
y_test = test.iloc[:,dft_num:dft_num+1].values
y_train = train.iloc[:,dft_num:dft_num+1].values 

regressor_student = RandomForestRegressor(n_estimators=200,max_features=61)
regressor_student.fit(X_train, y_train) 

y_pred = regressor_student.predict(X_test)  

per_tree_pred = np.array([tree.predict(data.iloc[:,0:dft_num].values) for tree in regressor_student.estimators_])
per_tree_pred_df = pd.DataFrame(per_tree_pred).T

rf_result = per_tree_pred_df
rf_result[reagent] = data[reagent]
rf_result['yield'] = data['yield_dft']
data_all = pd.concat([data.iloc[:,0:dft_num],rf_result],axis=1)
feature_num = dft_num + 200

# Dimension reduction
data_all_copy = data_all.copy(deep=True)
feature = data_all_copy.iloc[:,0:feature_num]
feature = feature.values

X = feature
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)

x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  # normalization

tsne = pd.DataFrame(X_norm)
tsne.columns = ['tsne_1','tsne_2']

data_all_copy = pd.concat([data_all_copy, tsne], axis=1)
data_all_copy.to_csv(os.path.join(BASE_DIR, output_path),index=False)