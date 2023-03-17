import os
import csv
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error

BASE_DIR = os.getcwd() # The current working directory
input_path = os.path.join(BASE_DIR, 'data/Data_After_Preprocessing_Buchwald_Hartwig_HTE.csv') # Relative path

data = pd.read_csv(input_path) # Read the csv file with relative path
reagent = 'additive_number'

list_train = [1,2,3,4]
list_val = [5]
list_test = [6,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]

dft_num = 120 # Number of the original reaction encoding

def eval_all_baseline(sample_num=5): 
  y_test_all = []
  y_pred_all = []

  for i in tqdm(range(len(list_test))):
    y_test,y_pred = eval_baseline(number = list_test[i],sample_num=sample_num)

    y_test_all += y_test
    y_pred_all += y_pred

  rmse = np.sqrt(metrics.mean_squared_error(y_test_all, y_pred_all))
  r2 = metrics.r2_score(y_test_all, y_pred_all)
  mae = mean_absolute_error(y_test_all, y_pred_all)
  print(rmse)
  print(r2)
  print(mae)
  return rmse,r2,mae

def eval_baseline(number = 16,sample_num=5): 

    tmp1 = data[data[reagent].isin(list_train)]
    X_train = tmp1.iloc[:,0:dft_num]
    y_train = tmp1[['yield']]
    test = pd.DataFrame()

    tmp = data[data[reagent] == number]
    df1 = tmp.sample(sample_num)
    df2 = pd.concat([tmp, df1]).drop_duplicates(keep=False)

    X_train = pd.concat([X_train, df1.iloc[:,0:dft_num]])
    y_train = pd.concat([y_train, df1[['yield']]])
    test = pd.concat([test, df2])

    X_train = X_train.values
    y_train = y_train.values
    X_test = test.iloc[:,0:dft_num].values
    y_test = test[['yield']].values.flatten().tolist()

    regressor = regressor_model
    regressor.fit(X_train, y_train.ravel())
    y_pred = regressor.predict(X_test).tolist()

    return y_test,y_pred

# RF
regressor_model = RandomForestRegressor()
eval_all_baseline(sample_num=5)

# Neural network
'''
regressor_model = MLPRegressor()
eval_all_baseline(sample_num=5)'''

# SVM
'''
regressor_model = SVR(kernel='linear')
eval_all_baseline(sample_num=5)'''

# Linear model
'''
regressor_model = LinearRegression()
eval_all_baseline(sample_num=5)'''

# RXNFP
'''
import pkg_resources
import torch
from rxnfp.models import SmilesClassificationModel
from rdkit import Chem
from rdkit.Chem import rdChemReactions
import pandas as pd

def generate_buchwald_hartwig_rxns(df):
    """
    Converts the entries in the excel files from Sandfort et al. to reaction SMILES.
    """
    df = df.copy()
    fwd_template = "[F,Cl,Br,I]-[c;H0;D3;+0:1](:[c,n:2]):[c,n:3].[NH2;D1;+0:4]-[c:5]>>[c,n:2]:[c;H0;D3;+0:1](:[c,n:3])-[NH;D2;+0:4]-[c:5]"
    methylaniline = "Cc1ccc(N)cc1"
    pd_catalyst = "O=S(=O)(O[Pd]1~[NH2]C2C=CC=CC=2C2C=CC=CC1=2)C(F)(F)F"
    methylaniline_mol = Chem.MolFromSmiles(methylaniline)
    rxn = rdChemReactions.ReactionFromSmarts(fwd_template)
    products = []

    for i, row in df.iterrows():
        reacts = (Chem.MolFromSmiles(row["aryl_halide_smiles"]), methylaniline_mol)
        rxn_products = rxn.RunReactants(reacts)

        rxn_products_smiles = set([Chem.MolToSmiles(mol[0]) for mol in rxn_products])
        assert len(rxn_products_smiles) == 1
        products.append(list(rxn_products_smiles)[0])

    df["product"] = products
    rxns = []

    for i, row in df.iterrows():
        reactants = Chem.MolToSmiles(
            Chem.MolFromSmiles(
                f"{row['aryl_halide_smiles']}.{methylaniline}.{pd_catalyst}.{row['ligand_smiles']}.{row['base_smiles']}.{row['additive_smiles']}"
            )
        )
        rxns.append(f"{reactants.replace('N~', '[NH2]')}>>{row['product']}")

    return rxns

data['rxn'] = generate_buchwald_hartwig_rxns(data)
train_df = data[data[reagent].isin(list_train)]
test_df = pd.DataFrame()

for i in range(len(list_test)):
  tmp = data[data[reagent] == list_test[i]]
  df1 = tmp.sample(5)
  df2 = tmp.append(df1).drop_duplicates(keep=False)

  train_df = train_df.append(df1)
  test_df = test_df.append(df2)

train_df = train_df[['rxn','yield_smiles']]
test_df = test_df[['rxn','yield_smiles']]
train_df.columns = ['text', 'labels']
test_df.columns = ['text', 'labels']

mean = train_df.labels.mean()
std = train_df.labels.std()
train_df['labels'] = (train_df['labels'] - mean) / std
test_df['labels'] = (test_df['labels'] - mean) / std

model_args = {
     'num_train_epochs': 15, 'overwrite_output_dir': True,
    'learning_rate': 0.00009659, 'gradient_accumulation_steps': 1,
    'regression': True, "num_labels":1, "fp16": False,
    "evaluate_during_training": False, 'manual_seed': 42,
    "max_seq_length": 300, "train_batch_size": 16,"warmup_ratio": 0.00,
    "config" : { 'hidden_dropout_prob': 0.7987 } 
}

model_path =  pkg_resources.resource_filename(
                "rxnfp",
                f"models/transformers/bert_pretrained" # change pretrained to ft to start from the other base model
)

yield_bert = SmilesClassificationModel("bert", model_path, num_labels=1, 
                                       args=model_args, use_cuda=torch.cuda.is_available())

yield_bert.train_model(train_df, output_dir=f"outputs_buchwald_hartwig_test_project", eval_df=test_df)

model_path = os.path.join(BASE_DIR, 'outputs_buchwald_hartwig_test_project/checkpoint-765-epoch-15') # Relative path

trained_yield_bert = SmilesClassificationModel('bert', model_path,
                                  num_labels=1, args={
                                      "regression": True
                                  }, use_cuda=torch.cuda.is_available())

yield_predicted = trained_yield_bert.predict(test_df.text.values.tolist())[0]
yield_predicted = yield_predicted * std + mean

yield_true = test_df.labels.values
yield_true = yield_true * std + mean

rmse = np.sqrt(metrics.mean_squared_error(yield_predicted, yield_true))
mae = mean_absolute_error(yield_predicted, yield_true)
print(rmse)
print(mae)'''

# GemNet
'''https://drive.google.com/file/d/1K532m1gGivUBmLu6RUZ-GAQxPk6mbqKS/view?usp=sharing'''

# DRFP
'''
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from drfp import DrfpEncoder
from xgboost import XGBRegressor

data['rxn'] = generate_buchwald_hartwig_rxns(data)

def eval_baseline_dfrp(number = 16,sample_num=5): 

    train_df = data[data[reagent].isin(list_train)]
    test_df = pd.DataFrame()

    tmp = data[data[reagent] == number]
    df1 = tmp.sample(sample_num)
    df2 = tmp.append(df1).drop_duplicates(keep=False)

    train_df = train_df.append(df1)
    test_df = test_df.append(df2)

    train_df = train_df[['rxn','yield_smiles']]
    test_df = test_df[['rxn','yield_smiles']]
    train_df.columns = ['text', 'labels']
    test_df.columns = ['text', 'labels']

    X_train, mapping = DrfpEncoder.encode(
            train_df.text.to_numpy(),
            n_folded_length=2048,
            radius=3,
            rings=True,
            mapping=True,
        )
    X_train = np.asarray(
            X_train,
            dtype=np.float32,
        )
    y_train = train_df.labels.to_numpy()

    X_test, _ = DrfpEncoder.encode(
            test_df.text.to_numpy(),
            n_folded_length=2048,
            radius=3,
            rings=True,
            mapping=True,
        )
    X_test = np.asarray(
            X_test,
            dtype=np.float32,
        )
    y_test = test_df.labels.to_numpy()

    valid_indices = np.random.choice(
        np.arange(len(X_train)), int(0.1 * len(X_train)), replace=False
    )

    X_valid = X_train[valid_indices]
    y_valid = y_train[valid_indices]
    train_indices = list(set(range(len(X_train))) - set(valid_indices))

    X_train = X_train[train_indices]
    y_train = y_train[train_indices]

    model = XGBRegressor()
    model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                early_stopping_rounds=20,
                verbose=False,
            )
    y_pred = model.predict(X_test, ntree_limit=model.best_ntree_limit)
    y_pred[y_pred < 0.0] = 0.0

    return y_test.tolist(),y_pred.tolist()

def eval_all_baseline_DRFP(sample_num=5): 
  y_test_all = []
  y_pred_all = []

  for i in tqdm(range(len(list_test))):
    y_test,y_pred = eval_baseline_dfrp(number = list_test[i],sample_num=sample_num)

    y_test_all += y_test
    y_pred_all += y_pred

  rmse = np.sqrt(metrics.mean_squared_error(y_test_all, y_pred_all))
  r2 = metrics.r2_score(y_test_all, y_pred_all)
  mae = mean_absolute_error(y_test_all, y_pred_all)
  print(rmse)
  print(r2)
  print(mae)
  return rmse,r2,mae

eval_all_baseline_drfp()'''



















