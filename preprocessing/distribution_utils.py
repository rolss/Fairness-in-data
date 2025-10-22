#import libraries
import numpy as np
import pandas as pd
np.random.seed(0)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from itertools import combinations
from functools import reduce
import matplotlib.pyplot as plt


def print_occurencies(df):
  value_counts_per_column = df.apply(lambda col: col.value_counts())

  # Print results
  print(value_counts_per_column)

def plot_occurencies(df, excluded_column=None):
  # Create pie charts for each column
  for col in df.columns:
      if excluded_column == None or col != excluded_column:  # Exclude the filtering column itself if not needed
          value_counts = df[col].value_counts()

          # Plot pie chart
          plt.figure(figsize=(5, 5))
          plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
          plt.title(f"Distribution of {col}")
          plt.show()

def plot_groups(df, target, col):
  # Create pie charts
    counts_C0 = df[df[target] == 0][col].value_counts()
    counts_C1 = df[df[target] == 1][col].value_counts()

    counts_df = pd.DataFrame({'class=0': counts_C0, 'class=1': counts_C1}).fillna(0)

    # Bar plot
    ax = counts_df.plot(kind='bar', figsize=(7, 5), width=0.7, color=['red', 'green'])

    # Add labels on top of bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.0f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add labels and title
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.title(f"Distribution of {col} by class values")
    plt.xticks(rotation=45)
    plt.legend(title="Class Value")

    # Show the plot
    plt.show()

def create_all_plots(dataset_path, protected_attributes, target):
  pairs_dict= {}
  for i in range(1,len(protected_attributes)+1):
    pairs_dict[i] = ['-'.join(pair) for pair in combinations(protected_attributes, i)]
  #print(pairs_dict)
  for i in range(1, len(protected_attributes)+1):
    df=pd.read_csv(dataset_path)
    for sensible_attribute in pairs_dict[i]:
      s = sensible_attribute.split('-')
      print(s)
      df=pd.read_csv(dataset_path)
      if len(s)>1:
        df[sensible_attribute] = reduce(lambda x, y: x.astype(str) + y.astype(str), [df[col] for col in s])
        df = df.drop(columns=s)

      plot_groups(df, target, sensible_attribute)

def plot_percentage(dataset_path, target, protected_attributes):
    pairs_dict = {}
    for i in range(1, len(protected_attributes) + 1):
        pairs_dict[i] = ['-'.join(pair) for pair in combinations(protected_attributes, i)]
    
    df = pd.read_csv(dataset_path)

    for i in range(1, len(protected_attributes) + 1):
        for sensible_attribute in pairs_dict[i]:
            s = sensible_attribute.split('-')
            df = pd.read_csv(dataset_path)

            if len(s) > 1:
                df[sensible_attribute] = reduce(lambda x, y: x.astype(str) + y.astype(str), [df[col] for col in s])
                df = df.drop(columns=s)

            # Calculate percentages
            counts_C0 = df[df[target] == 0][sensible_attribute].value_counts()
            counts_C1 = df[df[target] == 1][sensible_attribute].value_counts()
            perc = (counts_C1 / counts_C0) * 100

            # Create figure
            plt.figure(figsize=(8, 5))
            ax = perc.plot(kind='bar', color='skyblue', edgecolor='black')

            # Set y-axis limit to avoid overlapping
            plt.ylim(0, perc.max() * 1.2)  # Adds some space above the highest bar

            # Add labels to bars
            for i, v in enumerate(perc):
                plt.text(i, v + (perc.max() * 0.05), f"{v:.1f}%", 
                         ha='center', fontsize=10, fontweight='bold', 
                         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))

            # Customize plot
            plt.xlabel(sensible_attribute)
            plt.ylabel("Percentage (%)")
            plt.title(f"Percentage of Class=1 over Class=0 for {sensible_attribute}")
            plt.xticks(rotation=45)
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            # Show plot
            plt.show()