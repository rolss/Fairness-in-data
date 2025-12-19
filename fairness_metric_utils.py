#import libraries
import numpy as np
import pandas as pd
np.random.seed(0)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from itertools import combinations
from functools import reduce
import matplotlib.pyplot as plt

def performance_metrics(y_test, y_pred):
  precision = metrics.precision_score(y_test, y_pred)
  recall = metrics.recall_score(y_test, y_pred)
  accuracy = metrics.accuracy_score(y_test, y_pred)
  f1_score = metrics.f1_score(y_test, y_pred)
  print("Precision: "+str(precision)+ ", Recall: "+str(recall)+ ", Accuracy: "+str(accuracy)+", F1: "+str(f1_score))

  return precision, recall, accuracy, f1_score

def compute_predictions(df, target_variable, sensible_attribute, target, target_variable_labels=['0','1']):
  Y = df[target_variable]
  X = df.drop(target_variable, axis=1)
  X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
  sensible_indexes=df[sensible_attribute].loc[list(X_test.index)]
  model = RandomForestClassifier(random_state = 1234).fit(X_train,y_train)
  #feature_importance(model, X, X.columns)
  y_pred = model.predict(X_test)
  cm =confusion_matrix(y_test, y_pred, labels=target_variable_labels)
  #performance_metrics(y_test, y_pred)
  return sensible_indexes, y_pred, y_test, X_test

def compute_cm_group(df, sensible_attribute, sensible_indexes, y_pred, y_test, X_test, target_variable_labels=['0','1']):
  sensible_values = df[sensible_attribute].unique()
  idx_dict = {value: [] for value in sensible_values}

  # Iterate over the sensible_indexes DataFrame and store the indexes in the dictionary
  for idx, value in sensible_indexes.items():  # idx = DataFrame index, value = row (Series) of 'race'
      actual_value = value  # Extract the actual value from the 'race' column

      # Only store if the actual value is in sensible_values
      if actual_value in sensible_values:
          idx_dict[actual_value].append(idx)  # Append the index where the value appears

  # Print the dictionary containing the indexes for each value
  # print(idx_dict)
  index_mapping = {test_idx: original_idx  for original_idx, test_idx in zip(df.index, X_test.index)}
  # print(index_mapping)
  index_mapping = {test_idx: original_idx  for original_idx, test_idx in zip(df.index, X_test.index)}

  # Initialize a dictionary to store the predictions for each group
  y_pred_dict = {}
  y_test_dict = {}
  cm_dict= {}

  # Iterate over the groups in idx_dict
  for group, indices in idx_dict.items():
      # Map the original indices to the corresponding X_test indices
      test_indices = [index_mapping[original_idx] for original_idx in indices if original_idx in index_mapping]
      #print(test_indices)
      # Extract the predictions for the current group using the mapped test indices
      y_pred_group = y_pred[test_indices]  # Use the mapped indices to extract from y_pred
      y_test_group = y_test[indices].array
      # Store the predictions in the dictionary
      y_pred_dict[group] = y_pred_group
      y_test_dict[group] = y_test_group
  # Print the predictions for each group
  for group, preds in y_pred_dict.items():
      #print(f"Predictions for group {group}: {preds}")
      #print(f"Test labels for group {group}: {y_test_dict[group]}")
      y_test_group = y_test[indices].array
      if len(y_test_dict[group]) == 0:
          print(f"Warning: Group {group} has no samples in the test set. Skipping confusion matrix calculation.")
      else:
        cm_dict[group] = confusion_matrix(y_test_dict[group],preds, labels=np.unique(y_test_dict[group]))
  return cm_dict

def retrieve_values(cm):
  # Check if the confusion matrix has the expected shape
  if cm.shape == (2, 2):
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TP = cm[1][1]
  elif cm.shape == (1, 1):  # Handle case where only one class is present
    TN = cm[0][0]
    FP = 0  # Set FP, FN, TP to 0 as they don't apply in this case
    FN = 0
    TP = 0
  else:
    raise ValueError("Unexpected confusion matrix shape")
  total = TN + FP + FN + TP
  return TP, TN, FP, FN, total

def mapping_numbers_into_labels(group, sensible_attribute, mapping, dataset_path):
  df=pd.read_csv(dataset_path)
  s = sensible_attribute.split('-')
  mapped_group =''
  for index, attribute in enumerate(s):
    #print(mapping[attribute][int(group[index])])
    mapped_group = mapped_group+' ['+mapping[attribute][int(group[index])]+']'
  return mapped_group

def convert_types(obj):
    if isinstance(obj, dict):
        return {convert_types(k): convert_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_types(i) for i in obj]
    elif isinstance(obj, np.integer):  # Convert np.int64, np.int32 to int
        return int(obj)
    elif isinstance(obj, np.floating):  # Convert np.float64, np.float32 to float
        return float(obj)
    else:
        return obj

def compute_fairness_metrics(cm_dict, m, sensible_attribute, mapping, dataset_path):
  fairness_dict={}
  for group, cm in cm_dict.items():
    TP, TN, FP, FN, len_group = retrieve_values(cm)
    fairness_metric = 0
    if FP!=0:
      if m=='PPE':
        fairness_metric = (FP)/(TN+FP) #PredictiveEquality or FalsePositiveRate (FPR)
      elif m=='FPR':
        fairness_metric = (FP)/(TP+FP) #FalsDiscoveryRate 
      elif m=='FPP':
        fairness_metric= (FP)/len_group #FalsePositiveParity
      elif m=='FPN':
        fairness_metric= (FP)/(FN+FP) 
    if FN!=0:
      if m=='FPA':
        fairness_metric= (FN)/(TN+FN) #FORParity
      elif m=='EOP':
        fairness_metric= (FN)/(TP+FN) #EqualOpportunity
      elif m=='FNP':
        fairness_metric= (FN)/len_group #FalseNegativeParity
      elif m=='FNE':
        fairness_metric=(FN)/(FP+FN)
    if FP!=0 and FN!=0:
      if m=='ERR':
        fairness_metric= (FP+FN)/len_group #ErrorRate
    if TP!=0:
      if m=='GFA':
        fairness_metric = (TP+FP)/len_group #Group Fairness
      elif m=='PPA':
        fairness_metric = (TP)/(TP+FP) #Predictive Parity
      elif m=='OAE':
        fairness_metric= (TP+TN)/len_group #OverallAccuracyEquality
    if FP!=0 or FN!=0:
      fairness_dict[group] = fairness_metric
    else:
      mapped_group = mapping_numbers_into_labels(group, sensible_attribute, mapping, dataset_path)
      print(f"Warning: Group {mapped_group} {group} has only FN. Skipping fairness metrics calculation.")
  fairness_dict = convert_types(fairness_dict)  # Ensure Python-native types
  print(m, fairness_dict)
  return fairness_dict

def get_fractions(cm_dict):
  fractions_dict= {}
  for group, cm in cm_dict.items():
    TP, TN, FP, FN, len_group = retrieve_values(cm)
    fractions_dict[group] =(TP+TN)/len_group
  return fractions_dict

def get_max_min(data):
  max_key = max(data, key=data.get)  # Key with max value
  min_key = min(data, key=data.get)  # Key with min value

  max_value = round(data[max_key], 3)  # Max value rounded to 3 decimals
  min_value = round(data[min_key], 3)  # Min value rounded to 3 decimals

  # Print the results
  print(f"Max value: {max_value} (Key: {max_key})")
  print(f"Min value: {min_value} (Key: {min_key})")
  diff= max_value - min_value
  if max_value != 0:
    ratio = min_value/max_value
  else:
    ratio = 0
  print(f"Max-min: {round(max_value - min_value, 3)}")
  print(f"Min-max: {round(min_value/max_value, 3)}")
  #print(f"Min/Max: {round(min_value / max_value, 3)}")
  return diff, ratio

def get_attribute_analysis(df, target_variable, sensible_attribute, fair_metrics, dataset_path, mapping, target_variable_labels=['0','1']):
  sensible_indexes={}
  y_pred= []
  y_test= []
  differences = {}
  ratios= {}
  sensible_indexes, y_pred, y_test, X_test= compute_predictions(df, target_variable, sensible_attribute, target_variable_labels)
  cm_dict={}
  fairness_metrics_dict={}
  cm_dict= compute_cm_group(df, sensible_attribute, sensible_indexes, y_pred, y_test, X_test, target_variable_labels)
  for m in fair_metrics:
    fairness_metrics_dict[m]= compute_fairness_metrics(cm_dict, m, sensible_attribute, mapping, dataset_path)
    differences[m], ratios[m] = get_max_min(fairness_metrics_dict[m])
  return fairness_metrics_dict, differences, ratios

# ------------------- REWEIGHTING FUNCTIONS ----------------------------------------------
# TODO: Clean afterwards

def compute_fairness_metrics_and_counts(cm_dict, m, sensible_attribute, mapping, dataset_path):
  fairness_dict={}
  count_group= {}
  for group, cm in cm_dict.items():
    TP, TN, FP, FN, len_group = retrieve_values(cm)
    fairness_metric = 0
    if FP!=0:
      if m=='PPE':
        fairness_metric = (FP)/(TN+FP) #PredictiveEquality or FalsePositiveRate (FPR)
      elif m=='FPR':
        fairness_metric = (FP)/(TP+FP) #FalsDiscoveryRate
      elif m=='FPP':
        fairness_metric= (FP)/len_group #FalsePositiveParity
      elif m=='FPN':
        fairness_metric= (FP)/(FN+FP)
    if FN!=0:
      if m=='FPA':
        fairness_metric= (FN)/(TN+FN) #FORParity
      elif m=='EOP':
        fairness_metric= (FN)/(TP+FN) #EqualOpportunity
      elif m=='FNP':
        fairness_metric= (FN)/len_group #FalseNegativeParity
      elif m=='FNE':
        fairness_metric=(FN)/(FP+FN)
    if FP!=0 and FN!=0:
      if m=='ERR':
        fairness_metric= (FP+FN)/len_group #ErrorRate
    if TP!=0:
      if m=='GFA':
        fairness_metric = (TP+FP)/len_group #Group Fairness
      elif m=='PPA':
        fairness_metric = (TP)/(TP+FP) #Predictive Parity
      elif m=='OAE':
        fairness_metric= (TP+TN)/len_group #OverallAccuracyEquality
    if FP!=0 or FN!=0:
      fairness_dict[group] = fairness_metric
    else:
      mapped_group = mapping_numbers_into_labels(group, sensible_attribute, mapping, dataset_path)
      print(f"Warning: Group {mapped_group} {group} has only FN. Skipping fairness metrics calculation.")
    count_group[group]= len_group # len_group = TN + FP + FN + TP
  count_group = convert_types(count_group)
  fairness_dict = convert_types(fairness_dict)  # Ensure Python-native types
  # print("After: ", m, fairness_dict)
  return fairness_dict, count_group

def get_test_pred_fairness(df, target_variable, sensible_attribute, fair_metrics, dataset_path, mapping, target_variable_labels=['0','1']):
  sensible_indexes={}
  y_pred= []
  y_val= []
  
  # Now returns X_test, y_test, and the trained model as well
  sensible_indexes, y_pred, y_val, X_val, X_train, y_train, X_test, y_test, model = compute_predictions_reweighting(df, target_variable, sensible_attribute, target_variable_labels)
  
  cm_dict={}
  fairness_metrics_dict={}
  
  # Compute confusion matrix and fairness metrics on VALIDATION set
  cm_dict = compute_cm_group(df, sensible_attribute, sensible_indexes, y_pred, y_val, X_val, target_variable_labels)
  for m in fair_metrics:
    fairness_metrics_dict[m], count_groups = compute_fairness_metrics_and_counts(cm_dict, m, sensible_attribute, mapping, dataset_path)
  
  return y_train, X_train, X_val, y_val, y_pred, X_test, y_test, fairness_metrics_dict, count_groups, model

def compute_predictions_reweighting(df, target_variable, sensible_attribute, target, target_variable_labels=['0','1']):
  Y = df[target_variable]
  X = df.drop(target_variable, axis=1)
  
  # First split: 70% train, 30% temp (will be split into val and test)
  X_train, X_temp, y_train, y_temp = train_test_split(X, Y, test_size=0.3, random_state=1)
  
  # Second split: Split temp 50/50 into validation (15%) and test (15%) 
  X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=1)

  # X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.3, random_state=1)
  
  # Get sensible indexes for validation set (used for fairness metrics computation)
  sensible_indexes = df[sensible_attribute].loc[list(X_val.index)]
  
  # Model 1: Fit on training, predict on validation

  # --- Random Forest Version ---
  model = RandomForestClassifier(random_state = 1234).fit(X_train, y_train)
  y_pred = model.predict(X_val)  

  # --- XGB Version ---
  # import xgboost as xgb
  # # Convert string combinations to numeric codes for the sensible_attribute column only
  # X_train_numeric = X_train.copy()
  # if sensible_attribute in X_train_numeric.columns:
  #   X_train_numeric[sensible_attribute] = X_train_numeric[sensible_attribute].astype('category').cat.codes

  # X_val_numeric = X_val.copy()
  # if sensible_attribute in X_val_numeric.columns:
  #   X_val_numeric[sensible_attribute] = X_val_numeric[sensible_attribute].astype('category').cat.codes

  # model = xgb.XGBClassifier(random_state = 1234, eval_metric='logloss')
  # model.fit(X_train_numeric, y_train)
  # y_pred = model.predict(X_val_numeric)
  
  cm = confusion_matrix(y_val, y_pred, labels=target_variable_labels)
  print(sensible_attribute)
  performance_metrics(y_val, y_pred)
  
  return sensible_indexes, y_pred, y_val, X_val, X_train, y_train, X_test, y_test, model


def evaluate_model_on_test(model, df, sensible_attribute, X_test, y_test, fair_metrics, mapping, dataset_path, target_variable_labels=[0, 1]):
  # Get sensible indexes for test set
  sensible_indexes_test = df[sensible_attribute].loc[list(X_test.index)]

  # --- XGB Version ---
  # X_test_numeric = X_test.copy()
  # if sensible_attribute in X_test_numeric.columns:
  #   X_test_numeric[sensible_attribute] = X_test_numeric[sensible_attribute].astype('category').cat.codes
  # # Predict on test set
  # y_pred_test = model.predict(X_test_numeric)

  # --- Random Forest version ---
  y_pred_test = model.predict(X_test)
  
  # Compute performance metrics
  precision, recall, accuracy, f1 = performance_metrics(y_test, y_pred_test)
  
  # Compute confusion matrix per group on TEST set
  cm_dict={}
  cm_dict = compute_cm_group(df, sensible_attribute, sensible_indexes_test, y_pred_test, y_test, X_test, target_variable_labels)
  
  # Compute fairness metrics on TEST set
  fairness_metrics_dict = {}
  count_groups = {}
  for m in fair_metrics:
    fairness_metrics_dict[m], count_groups = compute_fairness_metrics_and_counts(cm_dict, m, sensible_attribute, mapping, dataset_path)
  
  cm = confusion_matrix(y_test, y_pred_test, labels=target_variable_labels)

  return y_pred_test, (precision, recall, accuracy, f1), fairness_metrics_dict, count_groups, cm_dict, cm


def compute_fairness_metrics_for_penalty(y_pred, y_test, X_test, sensible_attribute, fair_metrics, mapping, dataset_path, target_variable_labels=[0, 1]):
  # --------------------------------
  
  # Load original dataframe to get individual attribute columns
  df_original = pd.read_csv(dataset_path)
  
  # Parse the combined attribute into individual attributes
  individual_attrs = sensible_attribute.split('-')
  
  fairness_metrics_dict = {}
  count_groups_dict = {}
  
  # First, compute metrics for the COMBINED attribute (actual values)
  # Create df with combined column for indexing
  df_combined = df_original.copy()
  if len(individual_attrs) > 1:
    df_combined[sensible_attribute] = reduce(lambda x, y: x.astype(str) + y.astype(str), [df_combined[col] for col in individual_attrs])
  
  sensible_indexes_combined = df_combined[sensible_attribute].loc[list(X_test.index)] # from compute_predictions original function
  
  cm_dict_combined = compute_cm_group(
    df_combined, sensible_attribute, sensible_indexes_combined, 
    y_pred, y_test, X_test, target_variable_labels
  )
  
  fairness_metrics_dict[sensible_attribute] = {}
  for m in fair_metrics:
    fairness_metrics_dict[sensible_attribute][m], count_groups_dict[sensible_attribute] = \
      compute_fairness_metrics_and_counts(cm_dict_combined, m, sensible_attribute, mapping, dataset_path)
  
  # Now compute metrics for each INDIVIDUAL attribute (for predicted values / harmonic mean)
  for attr in individual_attrs:
    sensible_indexes_individual = df_original[attr].loc[list(X_test.index)]
    cm_dict_individual = compute_cm_group(
      df_original, attr, sensible_indexes_individual,
      y_pred, y_test, X_test, target_variable_labels
    )
    
    fairness_metrics_dict[attr] = {}
    for m in fair_metrics:
      fairness_metrics_dict[attr][m], count_groups_dict[attr] = \
        compute_fairness_metrics_and_counts(cm_dict_individual, m, attr, mapping, dataset_path)
  
  return fairness_metrics_dict, count_groups_dict



"""
def compute_penalty_2(fairness_metrics_dict, df, s1, s2, m):
  penalty_harmonic= {}
  penalty_geometric= {}
  penalty_arthmetic={}
  s3 = str(s1)+'-'+str(s2) # [age-sex]
  for i in range(0,df[s1].nunique()):
    for j in range(0,df[s2].nunique()):
      if i in fairness_metrics_dict[s1][m] and j in fairness_metrics_dict[s2][m]: # if exists
        a = fairness_metrics_dict[s1][m][i] # e.g. value for [age], m, on i = 0 or 1
        b = fairness_metrics_dict[s2][m][j] # e.g. value for [sex], m, on j = 0 or 1
        k=str(i)+str(j) # [01]
        if k in fairness_metrics_dict[s3][m]:
          c = fairness_metrics_dict[s3][m][k] # actual value | e.g. value for [age-sex], m, on k = 00, 01, 10, 11
        else:
          c = 0
        harmonic_prevision = (2*a*b)/(a+b)
        harmonic_penalty = penalty_percentage(c, harmonic_prevision)
        penalty_harmonic[k] = harmonic_penalty
        geometric_prevision = math.sqrt(a*b)
        geometric_penalty = penalty_percentage(c, geometric_prevision)
        penalty_geometric[k] = geometric_penalty
        arithmetic_prevision = (a+b)/2
        arithmetic_penalty = penalty_percentage(c, arithmetic_prevision)
        penalty_arthmetic[k] = arithmetic_penalty
  return penalty_harmonic, penalty_geometric, penalty_arthmetic
"""

