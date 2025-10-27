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