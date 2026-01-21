#import libraries
import numpy as np
import pandas as pd
import math
from itertools import combinations
from functools import reduce
import pandas as pd
import matplotlib.pyplot as plt


def mapping_numbers_into_labels(group, sensible_attribute, mapping, dataset_path):
  df=pd.read_csv(dataset_path)
  s = sensible_attribute.split('-')
  mapped_group =''
  for index, attribute in enumerate(s):
    #print(mapping[attribute][int(group[index])])
    mapped_group = mapped_group+' ['+mapping[attribute][int(group[index])]+']'
  return mapped_group

def penalty_percentage(actual_value, predicted_value):
  if predicted_value == 0:
    return 0.0
  return ((predicted_value-actual_value)*100)/predicted_value

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

def compute_penalty_3(fairness_metrics_dict, df, s1, s2, s3, m):
  penalty_harmonic= {}
  penalty_geometric= {}
  penalty_arthmetic={}
  s4 = str(s1)+'-'+str(s2)+'-'+str(s3)
  for i in range(0,df[s1].nunique()):
    for j in range(0,df[s2].nunique()):
      for k in range(0,df[s3].nunique()):
        if i in fairness_metrics_dict[s1][m] and j in fairness_metrics_dict[s2][m] and k in fairness_metrics_dict[s3][m]:
          a = fairness_metrics_dict[s1][m][i]
          b = fairness_metrics_dict[s2][m][j]
          c = fairness_metrics_dict[s3][m][k]
          w=str(i)+str(j)+str(k)
          if w in fairness_metrics_dict[s4][m]:
            d = fairness_metrics_dict[s4][m][w]
          else:
            d = 0
          harmonic_prevision = (3)/((1/a)+(1/b)+(1/c))
          harmonic_penalty = penalty_percentage(d, harmonic_prevision)
          penalty_harmonic[w] = harmonic_penalty
          geometric_prevision = math.pow((a*b*c), 1/3)
          geometric_penalty = penalty_percentage(d, geometric_prevision)
          penalty_geometric[w] = geometric_penalty
          arithmetic_prevision = (a+b+c)/3
          arithmetic_penalty = penalty_percentage(d, arithmetic_prevision)
          penalty_arthmetic[w] = arithmetic_penalty
          #print(s1+':'+str(i)+'= '+str(a)+', '+s2+':'+str(j)+'= '+str(b)+', '+s3+':'+str(k)+'= '+str(c))
          #print('actual value: '+str(d))
          #print(harmonic_prevision, geometric_prevision, arithmetic_prevision)
  return penalty_harmonic, penalty_geometric, penalty_arthmetic

def compute_penalty_4(fairness_metrics_dict, df, s1, s2, s3, s4, m):
  penalty_harmonic= {}
  penalty_geometric= {}
  penalty_arthmetic={}
  s5 = str(s1)+'-'+str(s2)+'-'+str(s3)+'-'+str(s4)
  for i in range(0,df[s1].nunique()):
    for j in range(0,df[s2].nunique()):
      for k in range(0,df[s3].nunique()):
        for l in range(0,df[s4].nunique()):
          if i in fairness_metrics_dict[s1][m] and j in fairness_metrics_dict[s2][m] and k in fairness_metrics_dict[s3][m] and l in fairness_metrics_dict[s4][m]:
            a = fairness_metrics_dict[s1][m][i]
            b = fairness_metrics_dict[s2][m][j]
            c = fairness_metrics_dict[s3][m][k]
            d = fairness_metrics_dict[s4][m][l]
            w=str(i)+str(j)+str(k)+str(l)
            if w in fairness_metrics_dict[s5][m]:
              e = fairness_metrics_dict[s5][m][w]
            else:
              e= 0
            harmonic_prevision = (4)/((1/a)+(1/b)+(1/c)+(1/d))
            harmonic_penalty = penalty_percentage(e, harmonic_prevision)
            penalty_harmonic[w] = harmonic_penalty
            geometric_prevision = math.pow((a*b*c*d), 1/4)
            geometric_penalty = penalty_percentage(e, geometric_prevision)
            penalty_geometric[w] = geometric_penalty
            arithmetic_prevision = (a+b+c+d)/4
            arithmetic_penalty = penalty_percentage(e, arithmetic_prevision)
            penalty_arthmetic[w] = arithmetic_penalty
  return penalty_harmonic, penalty_geometric, penalty_arthmetic

def plot_penalty_short(penalty_dict, pair, m, mapping, dataset_path):
  if penalty_dict is None:
    return
  fig, axes = plt.subplots(1, 3, figsize=(10, 8), constrained_layout=True)
  fig.suptitle(pair+' '+m, fontsize=12)
  for ax, (penalty, values) in zip(axes, penalty_dict.items()):
      df = pd.DataFrame.from_dict(values, orient='index', columns=[penalty])
      df = df.round(3)  # Round values to 3 decimal places
      #ax.axis('tight')
      ax.axis('off')  # Hide axis lines
      ax.plot([], [])  # Invisible plot to prevent collapse
      new_index = []
      for numbers in df.index:
        new_index.append(mapping_numbers_into_labels(numbers, pair, mapping, dataset_path))
      table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=new_index,
                        cellLoc='center', loc='center',
                        cellColours=[['lightcoral' if val > 0 else 'lightgreen' for val in row] for row in df.values],  bbox=[0, 0, 1, 1])  # Force full Axes usage)

      table.auto_set_font_size(False)
      table.set_fontsize(10)
      table.scale(1.2, 1.2)

  plt.show(block=False)

def plot_penalty(penalty_dict, pair, m, mapping, dataset_path):
  if penalty_dict is None:
    return
  fig, axes = plt.subplots(1, 3, figsize=(14, 8), constrained_layout=True)
  fig.suptitle(pair+' '+m, fontsize=12)
  for ax, (penalty, values) in zip(axes, penalty_dict.items()):
      df = pd.DataFrame.from_dict(values, orient='index', columns=[penalty])
      df = df.round(3)  # Round values to 3 decimal places
      #ax.axis('tight')
      ax.axis('off')  # Hide axis lines
      ax.plot([], [])  # Invisible plot to prevent collapse
      new_index = []
      for numbers in df.index:
        new_index.append(mapping_numbers_into_labels(numbers, pair, mapping, dataset_path))
      table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=new_index,
                        cellLoc='center', loc='center',
                        cellColours=[['lightcoral' if val > 0 else 'lightgreen' for val in row] for row in df.values],  bbox=[0, 0, 1, 1])  # Force full Axes usage)

      table.auto_set_font_size(False)
      table.set_fontsize(10)
      table.scale(1.2, 1.2)

  plt.show(block=False)

def print_tables_penalty_2(fairness_metrics_dict, df, pair, m, mapping, dataset_path):
    s1, s2 = pair.split('-')
    penalty_values = {}
    penalty_values['harmonic'], penalty_values['geometric'], penalty_values['arithmetic'] = compute_penalty_2(fairness_metrics_dict, df, s1, s2, m)
    penalty_dict={}
    penalty_dict[pair] = penalty_values
    plot_penalty_short(penalty_dict[pair], pair, m, mapping, dataset_path)

def print_tables_penalty_3(fairness_metrics_dict, df, pair, m, mapping, dataset_path):
    s1, s2, s3 = pair.split('-')
    penalty_values = {}
    penalty_values['harmonic'], penalty_values['geometric'], penalty_values['arithmetic'] = compute_penalty_3(fairness_metrics_dict, df, s1, s2, s3, m)
    penalty_dict={}
    penalty_dict[pair] = penalty_values
    plot_penalty(penalty_dict[pair], pair, m, mapping, dataset_path)

def print_tables_penalty_4(fairness_metrics_dict, df, pair, m, mapping, dataset_path):
    s1, s2, s3, s4 = pair.split('-')
    penalty_values = {}
    penalty_values['harmonic'], penalty_values['geometric'], penalty_values['arithmetic'] = compute_penalty_4(fairness_metrics_dict, df, s1, s2, s3, s4, m)
    penalty_dict={}
    penalty_dict[pair] = penalty_values
    plot_penalty_long(penalty_dict[pair], pair, m, mapping, dataset_path)

def plot_penalty_long(penalty_dict, pair, m, mapping, dataset_path):
  if penalty_dict is None:
    return
  fig, axes = plt.subplots(1, 3, figsize=(16, 10), constrained_layout=True)
  fig.suptitle(pair+' '+m, fontsize=12)
  for ax, (penalty, values) in zip(axes, penalty_dict.items()):
      df = pd.DataFrame.from_dict(values, orient='index', columns=[penalty])
      df = df.round(3)  # Round values to 3 decimal places
      #ax.axis('tight')
      ax.axis('off')  # Hide axis lines
      ax.plot([], [])  # Invisible plot to prevent collapse
      new_index = []
      for numbers in df.index:
        new_index.append(mapping_numbers_into_labels(numbers, pair, mapping, dataset_path))
      table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=new_index,
                        cellLoc='center', loc='center',
                        cellColours=[['lightcoral' if val > 0 else 'lightgreen' for val in row] for row in df.values],  bbox=[0, 0, 1, 1])  # Force full Axes usage)

      table.auto_set_font_size(False)
      table.set_fontsize(10)
      table.scale(1.2, 1.2)

  plt.show(block=False)


### -------------------------------------- REWEIGHTING FUNCTIONS ------------------------------------
def harmonic_mean_2(a,b):
  return (2)/((1/a)+(1/b))

def harmonic_mean_3(a,b,c):
  return (3)/((1/a)+(1/b)+(1/c))

def harmonic_mean_4(a,b,c,d):
  return (4)/((1/a)+(1/b)+(1/c)+(1/d))

# Penalty computation for each subgroup combination k
def compute_penalty(actual_values, predicted_values):
  penalties= {}
  # k = subgroup combination (00, 01, 10, etc...)
  for k in actual_values.keys():
    penalties[k] = penalty_percentage(actual_values[k], predicted_values[k])
  return penalties

# def penalty_percentage(actual_value, predicted_value):
#   return ((predicted_value-actual_value)*100)/predicted_value

def actual_predicted_values_2(fairness_metrics_dict, df, s1, s2, m):
  actual_values= {}
  predicted_values= {}
  s3 = str(s1)+'-'+str(s2)
  for i in range(0,df[s1].nunique()):
    for j in range(0,df[s2].nunique()):
      if i in fairness_metrics_dict[s1][m] and j in fairness_metrics_dict[s2][m]:
        a = fairness_metrics_dict[s1][m][i]
        b = fairness_metrics_dict[s2][m][j]
        k=str(i)+str(j)
        if k in fairness_metrics_dict[s3][m]:
          c = fairness_metrics_dict[s3][m][k]
        else:
          c = 0
        actual_values[k]= c
        predicted_values[k]= harmonic_mean_2(a,b)
  return actual_values, predicted_values

def actual_predicted_values_3(fairness_metrics_dict, df, s1, s2, s3, m):
  actual_values= {}
  predicted_values= {}
  s4 = str(s1)+'-'+str(s2)+'-'+str(s3)
  for i in range(0,df[s1].nunique()):
    for j in range(0,df[s2].nunique()):
      for k in range(0,df[s3].nunique()):
        if i in fairness_metrics_dict[s1][m] and j in fairness_metrics_dict[s2][m] and k in fairness_metrics_dict[s3][m]:
          a = fairness_metrics_dict[s1][m][i]
          b = fairness_metrics_dict[s2][m][j]
          c = fairness_metrics_dict[s3][m][k]
          w=str(i)+str(j)+str(k)
          if w in fairness_metrics_dict[s4][m]:
            d = fairness_metrics_dict[s4][m][w]
          else:
            d = 0
          actual_values[w]= d
          predicted_values[w]= harmonic_mean_3(a,b,c)
  return actual_values, predicted_values

def actual_predicted_values_4(fairness_metrics_dict, df, s1, s2, s3, s4, m):
  actual_values= {}
  predicted_values= {}
  s5 = str(s1)+'-'+str(s2)+'-'+str(s3)+'-'+str(s4)
  for i in range(0,df[s1].nunique()):
    for j in range(0,df[s2].nunique()):
      for k in range(0,df[s3].nunique()):
        for l in range(0,df[s4].nunique()):
          if i in fairness_metrics_dict[s1][m] and j in fairness_metrics_dict[s2][m] and k in fairness_metrics_dict[s3][m] and l in fairness_metrics_dict[s4][m]:
            a = fairness_metrics_dict[s1][m][i]
            b = fairness_metrics_dict[s2][m][j]
            c = fairness_metrics_dict[s3][m][k]
            d = fairness_metrics_dict[s4][m][l]
            w=str(i)+str(j)+str(k)+str(l)
            if w in fairness_metrics_dict[s5][m]:
              e = fairness_metrics_dict[s5][m][w]
            else:
              e= 0
            actual_values[w]= e
            predicted_values[w]= harmonic_mean_4(a,b,c,d)
  return actual_values, predicted_values
