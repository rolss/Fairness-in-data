# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
# from sklearn.metrics import confusion_matrix
# from fairness_metric_utils import performance_metrics, compute_cm_group

from scipy.spatial.distance import jensenshannon



"""
Compute sample weights based on intersectional penalties for collapsed string groups.

Parameters:
  df (pd.DataFrame): Dataset with true labels, predictions, and group column.
  y_true_col (str): Column name of ground truth labels.
  y_pred_col (str): Column name of model predictions.
  group_col (str): Column name of the combined group (e.g., "sex-race").
  penalties (dict): Mapping from group strings (e.g., "femaleBlack") to penalty scores.
  pattern_counts (dict): Optional dictionary of group counts.
  lambda_ (float): Reweighting strength.
  focus_on (str): Type of error to apply reweighting on.
  normalize_counts (bool): Whether to normalize pattern counts.

Returns:
  pd.Series: Sample weights.
"""
def compute_sample_weights_flat_group(df, y_true_col, y_pred_col, group_col, penalties, pattern_counts, lambda_=1.0, focus_on="fp"):

    weights = np.ones(len(df))  # Initialize weights to 1
    for i in range(len(df)):
        # Get the group for this row (sex-race) (attribute value combination)
        group = df.iloc[i][group_col]

        # Get the predicted and actual values for this row
        y_true = df.iloc[i][y_true_col]
        y_pred = df.iloc[i][y_pred_col]

        # Apply conditions for False Positives (or False Negatives, depending on focus)
        if focus_on == "fp" and y_true == 0 and y_pred == 1:
            # False positive — compute penalty and weight
            penalty = penalties.get(group, 0) # get value of the penalty for group
            weight = 1 + lambda_ * (penalty / 100)  # Scale the penalty

            # Debugging output: print row index, group, penalty, and computed weight
            print(f"Row {i}, group={group}, penalty={penalty:.2f}, weight={weight:.2f}")

            # Assign the computed weight
            weights[i] = weight

        elif focus_on == "fn" and y_true == 1 and y_pred == 0:
            penalty = penalties.get(group, 0)
            weight = 1 + lambda_ * (penalty / 100)
            weights[i] = weight

    # List of weights where the index corresponds to the row i in df (assigning more importance to it)
    return weights


def compute_sample_weights_flat_group_allmetrics(df, y_true, y_pred, group_col, focus_on, lambda_=1.0):
    # Initialize weights to 1
    weights = np.ones(len(df))

    # Create a DataFrame for convenience
    data = df.copy()
    data["y_true"] = y_true
    data["y_pred"] = y_pred

    # Step 1: Compute FP and FN counts per group
    group_stats = defaultdict(lambda: {"fp": 0, "fn": 0, "tn": 0, "tp": 0})

    for _, row in data.iterrows():
        g = row[group_col] # Attribute value combination for subgroup "group_col"
        yt, yp = row["y_true"], row["y_pred"] # true and pred value for that row

        if yp == 1 and yt == 0:
            group_stats[g]["fp"] += 1
        elif yp == 0 and yt == 1:
            group_stats[g]["fn"] += 1
        elif yp == 0 and yt == 0:
            group_stats[g]["tn"] += 1
        elif yp == 1 and yt == 1:
            group_stats[g]["tp"] += 1

    # Step 2: Compute metric per group
    metric_per_group = {}
    for g, counts in group_stats.items(): # tuple (attribute_value, {fp: x, fn: y})
        fp = counts["fp"]
        fn = counts["fn"]
        tn = counts["tn"]
        tp = counts["tp"]

        
        if focus_on == "fpn" or focus_on == "fne":
            denom = fp + fn
            if focus_on == "fpn":
                metric_per_group[g] = fp / denom if denom > 0 else 0.0
            else:
                metric_per_group[g] = fn / denom if denom > 0 else 0.0
        elif focus_on == "ppe":
            denom = tn + fp
            metric_per_group[g] = fp / denom if denom > 0 else 0.0
        elif focus_on == "fpr":
            denom = tp + fp
            metric_per_group[g] = fp / denom if denom > 0 else 0.0
        elif focus_on == "fpa":
            denom = tn + fn
            metric_per_group[g] = fn / denom if denom > 0 else 0.0
        elif focus_on == "eop":
            denom = tp + fn
            metric_per_group[g] = fn / denom if denom > 0 else 0.0
        elif focus_on == "fpp":
            denom = tp + fp + tn + fn
            metric_per_group[g] = fp / denom if denom > 0 else 0.0
        elif focus_on == "fnp":
            denom = tp + fp + tn + fn
            metric_per_group[g] = fn / denom if denom > 0 else 0.0
        # Add more metrics as needed
        

    # Step 3: Compute marginal metrics (i.e., individual attribute values)
    marginals = defaultdict(lambda: {"fp": 0, "fn": 0, "tn": 0, "tp": 0})
    for idx, row in data.iterrows():
        g = row[group_col]
        yt, yp = row["y_true"], row["y_pred"]
        for attr_val in str(g): # independent attribute value: 0 or 1
            if yp == 1 and yt == 0:
                marginals[attr_val]["fp"] += 1
            elif yp == 0 and yt == 1:
                marginals[attr_val]["fn"] += 1
            elif yp == 0 and yt == 0:
                marginals[attr_val]["tn"] += 1
            elif yp == 1 and yt == 1:
                marginals[attr_val]["tp"] += 1

    metric_marginals = {}
    for val, counts in marginals.items(): # tuple (attribute_value, {fp: x, fn: y})
        fp = counts["fp"]
        fn = counts["fn"]
        tn = counts["tn"]
        tp = counts["tp"]

        # Reflect the previous chain of if-elif here
        if focus_on == "fpn" or focus_on == "fne":
            denom = fp + fn
            if focus_on == "fpn":
                metric_marginals[val] = fp / denom if denom > 0 else 0.0
            else:
                metric_marginals[val] = fn / denom if denom > 0 else 0.0
        elif focus_on == "ppe":
            denom = tn + fp
            metric_marginals[val] = fp / denom if denom > 0 else 0.0
        elif focus_on == "fpr":
            denom = tp + fp
            metric_marginals[val] = fp / denom if denom > 0 else 0.0
        elif focus_on == "fpa":
            denom = tn + fn
            metric_marginals[val] = fn / denom if denom > 0 else 0.0
        elif focus_on == "eop":
            denom = tp + fn
            metric_marginals[val] = fn / denom if denom > 0 else 0.0
        elif focus_on == "fpp":
            denom = tp + fp + tn + fn
            metric_marginals[val] = fp / denom if denom > 0 else 0.0
        elif focus_on == "fnp":
            denom = tp + fp + tn + fn
            metric_marginals[val] = fn / denom if denom > 0 else 0.0
        # Add more metrics as needed

    # Step 4: Compute expected metric using mean aggregation
    predicted_metric = {}
    for g in metric_per_group:
        # Assumes group is encoded like '01', '10', etc. 
        # Creates a list of the form ['0','1']
        attr_vals = list(str(g)) 

        # Get the (marginal) metric value for each independent attribute value '0' or '1' to compute mean
        mean_val = np.mean([metric_marginals.get(a, 0) for a in attr_vals])
        predicted_metric[g] = mean_val

    # Step 5: Compute penalty and weights
    # for i in range(len(df)):
    #     # for every row i corresponding to column group_col, get the attribute value combination
    #     g = df.iloc[i][group_col]

    #     # Get the fpn value of g (attribute value combination)
    #     actual = metric_per_group.get(g, 0) 
    #     predicted = predicted_metric.get(g, 1)  # Avoid division by 0

    #     penalty = 100 * (predicted - actual) / predicted if predicted != 0 else 0
    #     weight = 1 + lambda_ * (penalty / 100) 
    #     weights[i] = weight
    # return weights
    
    # New step 5
    group_penalties = {}
    for g in metric_per_group:
        actual = metric_per_group.get(g, 0)
        predicted = predicted_metric.get(g, 1)
        penalty = 100 * (predicted - actual) / predicted if predicted != 0 else 0
        group_penalties[g] = penalty

    return group_penalties



def compute_sample_weights_flat_group_allmetrics(df, y_true, y_pred, group_col, focus_on, lambda_=1.0):
    # Initialize weights to 1
    weights = np.ones(len(df))

    # Create a DataFrame for convenience
    data = df.copy()
    data["y_true"] = y_true
    data["y_pred"] = y_pred

    # Step 1: Compute FP and FN counts per group
    group_stats = defaultdict(lambda: {"fp": 0, "fn": 0, "tn": 0, "tp": 0})

    for _, row in data.iterrows():
        g = row[group_col] # Attribute value combination for subgroup "group_col"
        yt, yp = row["y_true"], row["y_pred"] # true and pred value for that row

        if yp == 1 and yt == 0:
            group_stats[g]["fp"] += 1
        elif yp == 0 and yt == 1:
            group_stats[g]["fn"] += 1
        elif yp == 0 and yt == 0:
            group_stats[g]["tn"] += 1
        elif yp == 1 and yt == 1:
            group_stats[g]["tp"] += 1

    # Step 2: Compute metric per group
    metric_per_group = {}
    for g, counts in group_stats.items(): # tuple (attribute_value, {fp: x, fn: y})
        fp = counts["fp"]
        fn = counts["fn"]
        tn = counts["tn"]
        tp = counts["tp"]

        
        if focus_on == "fpn" or focus_on == "fne":
            denom = fp + fn
            if focus_on == "fpn":
                metric_per_group[g] = fp / denom if denom > 0 else 0.0
            else:
                metric_per_group[g] = fn / denom if denom > 0 else 0.0
        elif focus_on == "ppe":
            denom = tn + fp
            metric_per_group[g] = fp / denom if denom > 0 else 0.0
        elif focus_on == "fpr":
            denom = tp + fp
            metric_per_group[g] = fp / denom if denom > 0 else 0.0
        elif focus_on == "fpa":
            denom = tn + fn
            metric_per_group[g] = fn / denom if denom > 0 else 0.0
        elif focus_on == "eop":
            denom = tp + fn
            metric_per_group[g] = fn / denom if denom > 0 else 0.0
        elif focus_on == "fpp":
            denom = tp + fp + tn + fn
            metric_per_group[g] = fp / denom if denom > 0 else 0.0
        elif focus_on == "fnp":
            denom = tp + fp + tn + fn
            metric_per_group[g] = fn / denom if denom > 0 else 0.0
        # Add more metrics as needed
        

    # Step 3: Compute marginal metrics (i.e., individual attribute values)
    marginals = defaultdict(lambda: {"fp": 0, "fn": 0, "tn": 0, "tp": 0})
    for idx, row in data.iterrows():
        g = row[group_col]
        yt, yp = row["y_true"], row["y_pred"]
        for attr_val in str(g): # independent attribute value: 0 or 1
            if yp == 1 and yt == 0:
                marginals[attr_val]["fp"] += 1
            elif yp == 0 and yt == 1:
                marginals[attr_val]["fn"] += 1
            elif yp == 0 and yt == 0:
                marginals[attr_val]["tn"] += 1
            elif yp == 1 and yt == 1:
                marginals[attr_val]["tp"] += 1

    metric_marginals = {}
    for val, counts in marginals.items(): # tuple (attribute_value, {fp: x, fn: y})
        fp = counts["fp"]
        fn = counts["fn"]
        tn = counts["tn"]
        tp = counts["tp"]

        # Reflect the previous chain of if-elif here
        if focus_on == "fpn" or focus_on == "fne":
            denom = fp + fn
            if focus_on == "fpn":
                metric_marginals[val] = fp / denom if denom > 0 else 0.0
            else:
                metric_marginals[val] = fn / denom if denom > 0 else 0.0
        elif focus_on == "ppe":
            denom = tn + fp
            metric_marginals[val] = fp / denom if denom > 0 else 0.0
        elif focus_on == "fpr":
            denom = tp + fp
            metric_marginals[val] = fp / denom if denom > 0 else 0.0
        elif focus_on == "fpa":
            denom = tn + fn
            metric_marginals[val] = fn / denom if denom > 0 else 0.0
        elif focus_on == "eop":
            denom = tp + fn
            metric_marginals[val] = fn / denom if denom > 0 else 0.0
        elif focus_on == "fpp":
            denom = tp + fp + tn + fn
            metric_marginals[val] = fp / denom if denom > 0 else 0.0
        elif focus_on == "fnp":
            denom = tp + fp + tn + fn
            metric_marginals[val] = fn / denom if denom > 0 else 0.0
        # Add more metrics as needed

    # Step 4: Compute expected metric using mean aggregation
    predicted_metric = {}
    for g in metric_per_group:
        # Assumes group is encoded like '01', '10', etc. 
        # Creates a list of the form ['0','1']
        attr_vals = list(str(g)) 

        # Get the (marginal) metric value for each independent attribute value '0' or '1' to compute mean
        mean_val = np.mean([metric_marginals.get(a, 0) for a in attr_vals])
        predicted_metric[g] = mean_val

    # Step 5: Compute penalty and weights
    # for i in range(len(df)):
    #     # for every row i corresponding to column group_col, get the attribute value combination
    #     g = df.iloc[i][group_col]

    #     # Get the fpn value of g (attribute value combination)
    #     actual = metric_per_group.get(g, 0) 
    #     predicted = predicted_metric.get(g, 1)  # Avoid division by 0

    #     penalty = 100 * (predicted - actual) / predicted if predicted != 0 else 0
    #     weight = 1 + lambda_ * (penalty / 100) 
    #     weights[i] = weight
    # return weights
    
    # New step 5
    group_penalties = {}
    for g in metric_per_group:
        actual = metric_per_group.get(g, 0)
        predicted = predicted_metric.get(g, 1)
        penalty = 100 * (predicted - actual) / predicted if predicted != 0 else 0
        group_penalties[g] = penalty

    return group_penalties


# ---------------------------- JSD CODE -------------------------------------------


def is_categorical(series):
    return series.dtype == "object" or str(series.dtype) == "category"

def compute_jsd_categorical(original_col, resampled_col):
    """
    Compute JSD for categorical columns.
    We build 'all_categories' and reindex both distributions because JSD requires
    comparing probability vectors on the same support and in the same order.
    """
    all_categories = pd.Series(list(original_col) + list(resampled_col)).unique()
    p = original_col.value_counts(normalize=True).reindex(all_categories, fill_value=0).values
    q = resampled_col.value_counts(normalize=True).reindex(all_categories, fill_value=0).values
    # SciPy provides the distance (square root), so squaring it converts it back to the actual divergence
    return jensenshannon(p, q) ** 2

def compute_jsd_numeric(original_col, resampled_col, bins=20):
    """
    Compute JSD for numeric columns using histogram approximation.
    """
    combined = pd.concat([original_col, resampled_col])
    bin_edges = np.histogram_bin_edges(combined, bins=bins)
    p, _ = np.histogram(original_col, bins=bin_edges, density=True)
    q, _ = np.histogram(resampled_col, bins=bin_edges, density=True)
    p = p / p.sum() if p.sum() > 0 else np.zeros_like(p)
    q = q / q.sum() if q.sum() > 0 else np.zeros_like(q)
    return jensenshannon(p, q) ** 2


def build_jsd_table(original_df: pd.DataFrame,
                    resampled_dfs: dict,
                    columns: list,
                    bins: int = 20) -> pd.DataFrame:
    """
    Computes the Jensen–Shannon divergence (JSD) between the original and each resampled dataset, column by column.
    
    - original_df: the original dataset (baseline)
    - resampled_dfs: dictionary of resampled datasets like {"Resampled": resampled_df}
    - columns: list of columns to evaluate
    - bins: number of bins for numeric JSD computation
    
    Returns a DataFrame where each cell is the JSD score (0.0 = identical distribution)
    """
    # Original vs itself = 0.0 (baseline)
    jsd_scores = {"Original": {col: 0.0 for col in columns}}

    for label, res_df in resampled_dfs.items():
        jsd_scores[label] = {}
        for col in columns:
            orig_col = original_df[col]
            res_col = res_df[col]
            categorical = is_categorical(orig_col)
            jsd = compute_jsd_categorical(orig_col, res_col) if categorical else \
                  compute_jsd_numeric(orig_col, res_col, bins=bins)
            jsd_scores[label][col] = jsd


    # Convert the nested dictionary to a DataFrame
    result_df = pd.DataFrame.from_dict(jsd_scores, orient='index')
    result_df = result_df.reindex(columns=columns)
    return result_df




# VISUALIZATION: Distribution Comparison Before/After Reweighting

# - - - - - PLOTS - - - - - 
def plot_distribution_comparison(original_train_df, resampled_train_df, column, weight_type):

    # Get distributions - what % of the data belongs to each subgroup (column=sex-race)
    orig_dist = original_train_df[column].value_counts(normalize=True).sort_index() * 100
    res_dist = resampled_train_df[column].value_counts(normalize=True).sort_index() * 100
    
    # Align indices
    # all_categories = orig_dist.index.union(res_dist.index).sort_values()
    # labels = ['00', '01', '10', '11']  # Specific order
    labels = original_train_df[column].value_counts().index.sort_values()
    
    # Plot
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, orig_dist.values, width, label='Original', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, res_dist.values, width, label='Weighted', color='coral', alpha=0.8)
    
    # Add value labels on bars
    for bar, val in zip(bars1, orig_dist.values):
        ax.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=9, color='steelblue')
    for bar, val in zip(bars2, res_dist.values):
        ax.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=9, color='coral')
    
    ax.set_xlabel(column, fontsize=12)
    ax.set_ylabel('Percentage', fontsize=12)
    ax.set_title(f'Distribution Comparison: {column}\nOriginal vs Weighted Dataset (resampled) on {weight_type}', fontsize=14)
    
    # x-axis
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45 if len(labels) > 4 else 0, ha='right' if len(labels) > 4 else 'center')
    
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()



def plot_jsd_bar_chart(resampled_jsd, sensible_attribute, weight_type):
    # Get JSD values and sort by magnitude (highest first)
    jsd_values = resampled_jsd.sort_values(ascending=True)  # ascending for horizontal bar chart

    # Create color palette - highlight sensitive attribute differently
    colors = ['coral' if col == sensible_attribute else 'steelblue' for col in jsd_values.index]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Horizontal bar chart
    bars = ax.barh(jsd_values.index, jsd_values.values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    # Add value labels on bars
    for bar, val in zip(bars, jsd_values.values):
        ax.annotate(f'{val:.6f}', 
                    xy=(val, bar.get_y() + bar.get_height()/2),
                    xytext=(5, 0), textcoords='offset points',
                    ha='left', va='center', fontsize=9)

    # Labels
    ax.set_xlabel('Jensen-Shannon Divergence', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title(f'JSD by Feature: Distribution Shift After Reweighting\n(Higher = More Change) | Weights used: {weight_type}', fontsize=14)

    # Mean JSD line (interesting addition)
    ax.axvline(x=resampled_jsd.mean(), color='red', linestyle='--', linewidth=1.5, label=f'Mean JSD ({resampled_jsd.mean():.6f})')

    # Add legend, patch is used because bars don't have labels in this case
    
    legend_elements = [
        Patch(facecolor='coral', edgecolor='black', label=f'Sensitive Attribute'),
        Patch(facecolor='steelblue', edgecolor='black', label='Other Features'),
        plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1.5, label=f'Mean JSD')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_performance_metrics_bar_chart(performance_per_metric):
    list_of_metrics = ['fpn', 'fne', 'ppe', 'fpr', 'fpa', 'eop', 'fpp', 'fnp']
    limit = 5
    list_of_metrics_subset = list_of_metrics[:limit]

    # Create subplots for each performance metric
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten() # Fixes error

    metrics_names = ['Precision', 'Recall', 'Accuracy', 'F1-Score']
    colors = plt.cm.Set3(np.linspace(0, 1, len(list_of_metrics_subset))) # Using Set3 colormap, create a color for each fairness metric

    for idx, metric_name in enumerate(metrics_names):
        ax = axes[idx]
        
        # Extract values for this specific performance metric across all fairness metrics
        values = [performance_per_metric[m][idx] for m in list_of_metrics_subset]
        
        # Create bar chart
        x_pos = np.arange(len(list_of_metrics_subset))
        bars = ax.bar(x_pos, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
        
        # Subplot labels
        ax.set_title(f'{metric_name} Across Fairness Metrics', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Fairness Metric Used for Weighting', fontsize=12)
        ax.set_ylabel(f'{metric_name} Score', fontsize=12)
        # Configure x-axis bar labels with as many labels as fairness metrics
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.upper() for m in list_of_metrics_subset])
        
        # Add values on top of bars -----------------------
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Add grid and configure y-axis
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.set_ylim(min(values) - 0.01, max(values) + 0.01)

    plt.suptitle('Performance Metrics - Weights made with Different Fairness Metrics', 
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.show()

def plot_performance_metrics_individual(performance_per_metric):
    # Individual plots for each performance metric
    import matplotlib.pyplot as plt

    list_of_metrics = ['fpn', 'fne', 'ppe', 'fpr', 'fpa', 'eop', 'fpp', 'fnp']
    limit = len(list_of_metrics)
    list_of_metrics_subset = list_of_metrics[:limit]
    metrics_names = ['Precision', 'Recall', 'Accuracy', 'F1-Score']
    colors = plt.cm.Set3(np.linspace(0, 1, len(list_of_metrics_subset)))

    # Create individual plot for each performance metric
    for idx, metric_name in enumerate(metrics_names):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract values for this specific performance metric across all fairness metrics
        values = [performance_per_metric[m][idx] for m in list_of_metrics_subset]
        
        # Create bar chart
        x_pos = np.arange(len(list_of_metrics_subset))
        bars = ax.bar(x_pos, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Subplot labels
        ax.set_title(f'{metric_name} Across Different Fairness Metrics', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Fairness Metric Used for Weighting', fontsize=13)
        ax.set_ylabel(f'{metric_name} Score', fontsize=13)
        # Configure x-axis bar labels with as many labels as fairness metrics
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.upper() for m in list_of_metrics_subset], rotation=0, ha='center', fontsize=11)
        
        # Add values on top of bars
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add grid and configure y-axis
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.set_ylim(min(values) - 0.01, max(values) + 0.01)
        
        plt.tight_layout()
        plt.show()


def plot_performance_metrics_individual_models(performance_per_metric_model):
    # Individual line graphs for each performance metric
    import matplotlib.pyplot as plt

    list_of_metrics = ['fpn', 'fne', 'ppe', 'fpr', 'fpa', 'eop', 'fpp', 'fnp']
    limit = len(list_of_metrics)
    list_of_metrics_subset = list_of_metrics[:limit]
    metrics_names = ['Precision', 'Recall', 'Accuracy', 'F1-Score']

    model_types = ['GB', 'RF', 'XGB']
    model_colors = {'GB': '#1f77b4', 'RF': '#ff7f0e', 'XGB': '#2ca02c'}

    # Create individual line graph for each performance metric
    for idx, metric_name in enumerate(metrics_names):
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plot line for each model type
        for model in model_types:
            # Extract values for the specific performance metric across all fairness metrics, for a specific model type
            values = [performance_per_metric_model[model][m][idx] for m in list_of_metrics_subset]
            x_pos = np.arange(len(list_of_metrics_subset))

            ax.plot(x_pos, values,
                    marker='o',
                    color=model_colors[model], 
                    label=model, 
                    linewidth=3, 
                    markersize=10,
                    alpha=0.8)
            
        
        # Customize plot
        ax.set_title(f'{metric_name} Comparison Across Models and Fairness Metrics', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Fairness Metric Used for Weighting', fontsize=13)
        ax.set_ylabel(f'{metric_name} Score', fontsize=13)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.upper() for m in list_of_metrics_subset], rotation=0, ha='center', fontsize=11)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, linestyle='--', axis='both')
        ax.set_ylim(min(values) - 0.05, max(values) + 0.01)
        
        # Add legend
        ax.legend(loc='best', fontsize=12, framealpha=0.9)
        
        plt.tight_layout()
        plt.show()


def plot_performance_metrics_comparison_models(performance_per_metric_model, fairness_metric):

    import matplotlib.pyplot as plt

    model_types = ['GB', 'RF', 'XGB']
    metrics_names = ['Precision', 'Recall', 'Accuracy', 'F1-Score']


    # Prepare data for plotting
    precision_values = [performance_per_metric_model[model][fairness_metric][0] for model in model_types] # list of precision values for each model, where fairness_metric = 'fpn'
    recall_values = [performance_per_metric_model[model][fairness_metric][1] for model in model_types]
    accuracy_values = [performance_per_metric_model[model][fairness_metric][2] for model in model_types]
    f1_values = [performance_per_metric_model[model][fairness_metric][3] for model in model_types]

    # Set up the plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Width of each bar and positions
    bar_width = 0.2
    x = np.arange(len(metrics_names))

    # Create bars for each model
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    for idx, model in enumerate(model_types):
        # At each iteration plot bars for each model and all performance metrics
        values = [precision_values[idx], recall_values[idx], accuracy_values[idx], f1_values[idx]]
        positions = x + (idx - 1) * bar_width
        bars = ax.bar(positions, values, bar_width, label=model, color=colors[idx], alpha=0.8, edgecolor='black', linewidth=1.2)
        
        # Add value labels on top of bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Plot labels
    ax.set_title(f'Model Performance Comparison for {fairness_metric.upper()} Fairness Metric', 
                fontsize=16, pad=20)
    ax.set_xlabel('Performance Metrics', fontsize=13)
    ax.set_ylabel('Scores', fontsize=13)

    # x-axis bar labels
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=11)

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_ylim(min(values) - 0.01, max(values) + 0.01)

    # Add legend
    ax.legend(loc='upper right', fontsize=11, ncol=1)

    plt.tight_layout()
    plt.show()


def plot_before_after_performance_bars(baseline_performance, after_performance_variants):

    metrics_names = ['Precision', 'Recall', 'Accuracy', 'F1-Score']
    model_types = ['RF', 'GB', 'XGB']
    colors_before = '#95a5a6'  # Gray for before
    colors_after = ['#3498db', '#e74c3c', '#2ecc71']  # Blue, Red, Green

    # Create 3 separate figures, one for each variant
    for variant_name, after_performance in after_performance_variants.items():
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, (ax, model) in enumerate(zip(axes, model_types)):
            x = np.arange(len(metrics_names))
            bar_width = 0.35
            
            # Create grouped bars
            bars_before = ax.bar(x - bar_width/2, baseline_performance[model], bar_width, 
                                label='Before (Baseline)', color=colors_before, alpha=0.7, edgecolor='black')
            bars_after = ax.bar(x + bar_width/2, after_performance[model], bar_width, 
                                label=f'After', color=colors_after[idx], alpha=0.8, edgecolor='black')
            
            # Add value labels
            for bars in [bars_before, bars_after]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}',
                        ha='center', va='bottom', fontsize=9)
            
            # Customize subplot
            ax.set_title(f'{model} Model: Before vs After', fontsize=14, fontweight='bold')
            ax.set_xlabel('Performance Metrics', fontsize=11)
            ax.set_ylabel('Score', fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels(metrics_names, fontsize=10)
            ax.set_ylim(0.55, 0.92)
            ax.grid(True, alpha=0.3, linestyle='--', axis='y')
            ax.legend(loc='lower right', fontsize=10)

        plt.suptitle(f'Model Performance: Before vs After - {variant_name}', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()


def plot_before_after_performance_lines(baseline_performance, after_performance_variants):

    metrics_names = ['Precision', 'Recall', 'Accuracy', 'F1-Score']
    model_types = ['RF', 'GB', 'XGB']
    model_names = {'RF': 'Random Forest', 'GB': 'Gradient Boosting', 'XGB': 'XGBoost'}
    x = np.arange(len(metrics_names))

    # Colors per model
    colors_dict = {'RF': '#3498db', 'GB': '#e74c3c', 'XGB': '#2ecc71'}
    markers_dict = {'RF': 'o', 'GB': 's', 'XGB': '^'}

    for variant_name, after_performance in after_performance_variants.items():
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, model in enumerate(model_types):
                ax = axes[idx]

                # Plot baseline (before) - dashed line
                ax.plot(x, baseline_performance[model], 
                        linestyle='--', linewidth=2, 
                        color=colors_dict[model], 
                        marker=markers_dict[model], 
                        markersize=8,
                        alpha=0.6,
                        label='Before')

                # Plot after reweighting - solid line
                ax.plot(x, after_performance[model], 
                        linestyle='-', linewidth=3, 
                        color=colors_dict[model], 
                        marker=markers_dict[model], 
                        markersize=10,
                        alpha=0.9,
                        label='After')

                # Subplot labels
                ax.set_title(f'{model_names[model]}', 
                                fontsize=14, fontweight='bold', pad=15)
                ax.set_xlabel('Performance Metrics', fontsize=11)
                ax.set_ylabel('Score', fontsize=11)
                ax.grid(True, alpha=0.3, linestyle='--', axis='both')
                ax.legend(loc='lower right', fontsize=10, framealpha=0.95)

                # x-axis 
                ax.set_xticks(x)
                ax.set_xticklabels(metrics_names, fontsize=10)

                # y-axis
                ax.set_ylim(0.60, 0.88)
        

        # title
        plt.suptitle('Performance Comparison: Before (Baseline) vs After (FPN Reweighting)', 
                fontsize=16, fontweight='bold', y=1.02)

        plt.tight_layout()
        plt.show()


def plot_before_after_performance_diff(baseline_performance, after_performance_variants):
    metrics_names = ['Precision', 'Recall', 'Accuracy', 'F1-Score']
    model_types = ['RF', 'GB', 'XGB']
    colors_dict = {'RF': '#3498db', 'GB': '#e74c3c', 'XGB': '#2ecc71'}
    bar_width = 0.25

    # Calculate the difference (after - before)
    for variant_name, after_performance in after_performance_variants.items():
        performance_delta = {}
        for model in model_types:
            performance_delta[model] = [
                after_performance[model][i] - baseline_performance[model][i] 
                for i in range(len(metrics_names))
            ]

        fig, ax = plt.subplots(figsize=(12, 7))
        x = np.arange(len(metrics_names))

        # Create bars for each model
        for idx, model in enumerate(model_types):
            positions = x + (idx - 1) * bar_width
            deltas = performance_delta[model]
            
            bars = ax.bar(positions, deltas, bar_width, 
                        label=model, 
                        color=colors_dict[model], 
                        alpha=0.8, 
                        edgecolor='black', 
                        linewidth=1.2)
            
            # Add value labels
            for bar, delta in zip(bars, deltas):
                height = bar.get_height()
                label_y = height + 0.002 if height >= 0 else height - 0.005
                va = 'bottom' if height >= 0 else 'top'
                ax.text(bar.get_x() + bar.get_width()/2., label_y,
                    f'{delta:+.4f}',
                    ha='center', va=va, fontsize=9, fontweight='bold')

        # Add horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)

        # Customize plot
        ax.set_title(f'Performance Change — {variant_name} (Δ = After - Before)', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Performance Metrics', fontsize=13)
        ax.set_ylabel('Change in Score', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names, fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.legend(loc='lower right', fontsize=12, framealpha=0.9)
        ax.set_ylim(-0.08, 0.09)

        # Add text annotation explaining positive/negative
        ax.text(0.02, 0.98, 'Positive values = Improvement\nNegative values = Degradation', 
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()
        plt.show()