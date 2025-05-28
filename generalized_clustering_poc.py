import sys
import sqlite3
import os
import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import plotly.express as px
import plotly
import re
import math
import hashlib  # For generating unique IDs for itemsets
import json
import itertools
from mlxtend.frequent_patterns import fpmax
from typing import Dict, List, Set, Tuple, Optional, Union, Any, FrozenSet, TypedDict
from collections import defaultdict, Counter
from scipy.cluster.hierarchy import fcluster, linkage, dendrogram
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path="env/.env")
DB_PATH = os.environ.get("DB_PATH")
CONFIG_FILE = os.environ.get("CONFIG_FILE", "clustering_config.json")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# -------------------------------------------------------------------
# compute_score:
#   Compute a score for an itemset based on a trade-off between 
#   support (coverage) and cohesion (aggregated column weights).
#
#   score = γ * support + (1–γ) * (normalized cohesion)
#
#   normalized cohesion is defined as the sum of weights for the
#   columns in the itemset divided by the sum over all columns.
#
#   γ ∈ [0,1] is a parameter that lets users stress coverage or cohesion.
# -------------------------------------------------------------------
def compute_score(itemset, support, weight_matrix, gamma):
    total_weight_itemset = 0.0
    for item in itemset:
        # Extract column name (assumes one-hot encoding: "colName___value")
        col_name = item.split("___")[0]
        total_weight_itemset += weight_matrix.get(col_name, 1.0)

    total_possible_weight = sum(abs(value) for value in (weight_matrix.values()))
    normalized_cohesion = (total_weight_itemset / total_possible_weight) if total_possible_weight != 0 else total_weight_itemset

    return gamma * support + (1 - gamma) * normalized_cohesion

# -------------------------------------------------------------------
# rank_maximal_frequent_itemsets:
#   Given a tabular dataset TD and weight matrix for columns,
#   discover maximal frequent itemsets (using FPMax), compute their scores,
#   rank them and optionally prune the collection.
#
#   Additionally, this function associates a unique id with each itemset based
#   on a hash of the column names and their values.
#
#   gamma controls the trade-off between coverage (support) and cohesion.
# -------------------------------------------------------------------
def rank_maximal_frequent_itemsets(TD: pd.DataFrame, weight_matrix, min_support=0.1, max_collection=10, gamma=0.5, row_id_colname=''):
    TD_to_encode = TD.drop(columns=[row_id_colname]) if row_id_colname and row_id_colname in TD.columns else TD
    # Convert to string, then replace all NaN-like values
    TD_to_encode = TD_to_encode.astype(str).replace(
        ['None', 'none', 'NONE', 'nan', 'NaN', 'NAN', 'NaT', 'nat', 'NAT', 'null', 'NULL'],
        np.nan
    )

    df_encoded = pd.get_dummies(TD_to_encode, prefix_sep="___", dummy_na=False)

    frequent_itemsets = fpmax(df_encoded, min_support=min_support, use_colnames=True)
    frequent_itemsets['score'] = frequent_itemsets.apply(
        lambda row: compute_score(row['itemsets'], row['support'], weight_matrix, gamma), axis=1
    )
    
    def get_supporting_row_ids(itemset):
        mask = df_encoded[list(itemset)].all(axis=1)
        if row_id_colname in TD.columns:
            return TD.loc[mask, row_id_colname].tolist()
        else:
            return TD.loc[mask].index.tolist()

    frequent_itemsets[row_id_colname] = frequent_itemsets['itemsets'].apply(get_supporting_row_ids)
    ranked_itemsets = frequent_itemsets.sort_values(by='score', ascending=False)
    if max_collection == -1:
        pruned_itemsets = ranked_itemsets.copy()
    else:
        pruned_itemsets = ranked_itemsets.head(max_collection).copy()
    pruned_itemsets.reset_index(drop=True, inplace=True)
    
    def compute_itemset_id(itemset, prefix_sep="___"):
        details = []
        for item in itemset: 
            parts = item.split(prefix_sep, 1)
            col, val = parts if len(parts) == 2 else (parts[0], "")
            if val != 'nan':  # Skip NaN-like values
                details.append(f"{col}:{val}")
        details.sort()
        hash_input = "|".join(details)
        return hashlib.md5(hash_input.encode("utf-8")).hexdigest()

    pruned_itemsets['unique_id'] = pruned_itemsets['itemsets'].apply(lambda items: compute_itemset_id(items))
    return pruned_itemsets

# -------------------------------------------------------------------
# logistic_loss_gradient:
#   Compute the gradient of the logistic loss with respect to δ.
#
#   Here δ is the difference in scores for two clusters.
# -------------------------------------------------------------------
def logistic_loss_gradient(delta: float) -> float:
    """Compute the gradient of the logistic loss with respect to δ."""
    sigma = 1 / (1 + np.exp(-delta))
    return -(1 - sigma)

# -------------------------------------------------------------------
# update_parameters_with_feedback:
#   When a user provides preference feedback (preferring cluster_A over cluster_B),
#   update both the column weights and the γ value.
#
#   Each cluster is defined by a list of column names. Their support (coverage)
#   values are given as supp_A and supp_B.
#
#   Normalized cohesion for a cluster is computed as (sum_{j in cluster} w_j)/(sum of all weights).
#
#   Updates:
#     - Column weights: update only the cohesion part.
#     - γ: update based on the sensitivity of δ with respect to γ.
#
#   lr_weights and lr_gamma are the learning rates for the weight and γ updates.
# -------------------------------------------------------------------
def update_parameters_with_feedback(
    weights: Dict[str, float], 
    gamma: float, 
    cluster_A: List[str], 
    cluster_B: List[str], 
    supp_A: float, 
    supp_B: float, 
    lr_weights: float = 0.9, 
    lr_gamma: float = 0.3
) -> Tuple[Dict[str, float], float]:
    """Update parameters based on feedback."""
    total_possible_weight = sum(abs(value) for value in (weights.values()))

    # Compute normalized cohesion for each cluster.
    cohesion_A = sum(weights[j] for j in cluster_A) / total_possible_weight
    cohesion_B = sum(weights[j] for j in cluster_B) / total_possible_weight

    # Compute the current scores for cluster_A and cluster_B.
    score_A = gamma * supp_A + (1 - gamma) * cohesion_A
    score_B = gamma * supp_B + (1 - gamma) * cohesion_B
    delta = score_A - score_B

    # Compute logistic loss gradient with respect to delta.
    grad = logistic_loss_gradient(delta)

    # Update weights for the cohesion part.
    # Flipped the updates: now weights for cluster_A are decreased by (-grad) (thus effectively increased)
    # and weights for cluster_B are increased by (-grad) (thus effectively decreased) when grad is negative.
    for j in cluster_A:
        weights[j] -= lr_weights * (1 - gamma) / total_possible_weight * grad
    for j in cluster_B:
        weights[j] += lr_weights * (1 - gamma) / total_possible_weight * grad

    # Derivative of δ with respect to γ:
    # d(score_A)/dγ = supp_A - cohesion_A, and d(score_B)/dγ = supp_B - cohesion_B.
    ddelta_dgamma = (supp_A - cohesion_A) - (supp_B - cohesion_B)

    # Update gamma.
    gamma -= lr_gamma * grad * ddelta_dgamma
    # Ensure γ stays within [0,1].
    gamma = np.clip(gamma, 0, 1)

    return weights, gamma

# -------------------------------------------------------------------
# extract_columns:
#   Helper function to extract column names from a one-hot encoded item.
#
#   Given an itemset (e.g., frozenset({'col1___A', 'col3___B'})), this returns
#   the corresponding list of column names: ['col1', 'col3'].
# -------------------------------------------------------------------
def extract_columns(itemset: FrozenSet[str]) -> List[str]:
    """Extract column names from a one-hot encoded item."""
    return list({item.split("___")[0] for item in itemset})

# -------------------------------------------------------------------
# remove_columns_with_values_common_to_all_itemsets:
#   Given a pandas DataFrame 'pruned_itemsets' (which must contain an 'itemsets' column),
#   this function:
#     1. Builds a list of ordered_details dictionaries for all itemsets.
#     2. Identifies key-value pairs that are common to all dictionaries.
#     3. Removes those common entries from each dictionary.
#   Returns a new list of dictionaries with the common entries removed.
# -------------------------------------------------------------------
def remove_columns_with_values_common_to_all_itemsets(
    pruned_itemsets: pd.DataFrame, 
    prefix_sep: str = "___"
) -> List[Dict[str, str]]:
    """Remove columns with values common to all itemsets."""
    # Build a list of ordered_details dictionaries.
    details_list = []
    for _, row in pruned_itemsets.iterrows():
        itemset = row['itemsets']
        details = {}
        for item in itemset:
            parts = item.split(prefix_sep, 1)
            col, val = parts if len(parts) == 2 else (parts[0], "")
            details[col] = val
        ordered_details = {key: details[key] for key in sorted(details)}
        details_list.append(ordered_details)

    # Determine the common key-value pairs across all dictionaries.
    if not details_list:
        return details_list
    common = details_list[0].copy()
    for d in details_list[1:]:
        keys_to_remove = []
        keys_to_remove.extend(
            key
            for key in list(common.keys())
            if key not in d or d[key] != common[key]
        )
        for key in keys_to_remove:
            del common[key]

    # Remove the common entries from each dictionary.
    filtered_list = []
    for dd in details_list:
        new_dict = {k: v for k, v in dd.items() if k not in common}
        filtered_list.append(new_dict)

    return filtered_list


def extract_column_from_item(item: str) -> str:
    """Extract column name from a one-hot encoded item."""
    return item.split("___")[0]

def extract_value_from_item(item: str) -> str:
    """Extract value from a one-hot encoded item."""
    parts = item.split("___", 1)
    return parts[1] if len(parts) > 1 else ""

def itemset_to_column_dict(itemset) -> Dict[str, str]:
    """Convert itemset to dictionary mapping columns to values."""
    result = {}
    for item in itemset:
        col = extract_column_from_item(item)
        val = extract_value_from_item(item)
        result[col] = val
    return result

def cluster_itemsets_by_similarity(pruned_itemsets: pd.DataFrame, 
                                  row_id_colname: str, 
                                  similarity_threshold: float = 0.4) -> Dict[int, List[int]]:
    """
    Cluster itemsets based on similarity in columns and row coverage.
        
    Returns:
    --------
    Dict[int, List[int]]
        Dictionary mapping cluster IDs to lists of itemset indices
    """
    n_itemsets = len(pruned_itemsets)
    if n_itemsets <= 1:
        return {0: [0]} if n_itemsets == 1 else {}
    
    similarity_matrix = np.zeros((n_itemsets * (n_itemsets - 1)) // 2)
    
    idx = 0
    for i in range(n_itemsets):
        for j in range(i+1, n_itemsets):
            itemset_i = pruned_itemsets.iloc[i]['itemsets']
            itemset_j = pruned_itemsets.iloc[j]['itemsets']
            
            # Calculate column similarity (Jaccard)
            cols_i = {extract_column_from_item(item) for item in itemset_i}
            cols_j = {extract_column_from_item(item) for item in itemset_j}
            col_similarity = len(cols_i & cols_j) / len(cols_i | cols_j) if cols_i | cols_j else 0
            
            # Calculate row similarity (Jaccard)
            rows_i = set(pruned_itemsets.iloc[i][row_id_colname])
            rows_j = set(pruned_itemsets.iloc[j][row_id_colname])
            row_similarity = len(rows_i & rows_j) / len(rows_i | rows_j) if rows_i | rows_j else 0
            
            composite_similarity = 0.5 * col_similarity + 0.5 * row_similarity
            
            similarity_matrix[idx] = 1 - composite_similarity
            idx += 1
    
    Z = linkage(similarity_matrix, method='average')
    
    clusters = fcluster(Z, 1 - similarity_threshold, criterion='distance')
    
    cluster_groups = defaultdict(list)
    for i, cluster_id in enumerate(clusters):
        cluster_groups[cluster_id].append(i)
    
    return dict(cluster_groups)

def extract_common_columns(itemset_indices: List[int], pruned_itemsets: pd.DataFrame) -> List[str]:
    """Find columns that appear in all itemsets in a group."""
    if not itemset_indices:
        return []
    
    column_sets = []
    for idx in itemset_indices:
        itemset = pruned_itemsets.iloc[idx]['itemsets']
        columns = {extract_column_from_item(item) for item in itemset}
        column_sets.append(columns)
    
    common_columns = column_sets[0].copy()
    for column_set in column_sets[1:]:
        common_columns &= column_set
    
    return sorted(common_columns)

def find_column_value_distributions(itemset_indices: List[int], pruned_itemsets: pd.DataFrame) -> Dict[str, Counter]:
    """Find value distributions for each column across itemsets in a group."""
    column_values = defaultdict(Counter)
    
    for idx in itemset_indices:
        itemset = pruned_itemsets.iloc[idx]['itemsets']
        for item in itemset:
            col = extract_column_from_item(item)
            val = extract_value_from_item(item)
            column_values[col][val] += 1
    
    return column_values

def find_constant_columns_across_clusters(clusters: List[Dict], pruned_itemsets: pd.DataFrame) -> Dict[str, str]:
    """
    Identify columns that appear in all clusters with the same value.
    
    Returns:
    --------
    Dict[str, str]
        Dictionary mapping column names to their constant values
    """
    if not clusters:
        return {}
    
    column_cluster_counts = defaultdict(int)
    column_value_counters = defaultdict(Counter)
    
    for cluster in clusters:
        cluster_columns = set()
        for idx in cluster["itemset_indices"]:
            itemset = pruned_itemsets.iloc[idx]['itemsets']
            for item in itemset:
                col = extract_column_from_item(item)
                val = extract_value_from_item(item)
                if col not in cluster_columns:
                    column_cluster_counts[col] += 1
                    cluster_columns.add(col)
                column_value_counters[col][val] += 1
    
    # Find columns that appear in all clusters with a single value
    constant_columns = {}
    num_clusters = len(clusters)
    for col, count in column_cluster_counts.items():
        if count == num_clusters:  # Column must appear in all clusters
            counter = column_value_counters[col]
            if len(counter) == 1:  # Column has only one value
                constant_columns[col] = next(iter(counter))
    
    return constant_columns

def get_cluster_metadata(itemset_indices: List[int], pruned_itemsets: pd.DataFrame, row_id_colname: str, constant_columns: Dict[str, str]) -> Dict:
    """
    Get metadata for a cluster of itemsets.
        
    Returns:
    --------
    Dict
        Dictionary with cluster metadata
    """
    common_cols = extract_common_columns(itemset_indices, pruned_itemsets)

    common_cols = [col for col in common_cols if col not in constant_columns]
    
    value_distributions = find_column_value_distributions(itemset_indices, pruned_itemsets)

    value_distributions = {col: counter for col, counter in value_distributions.items() if col not in constant_columns}
    
    all_rows = set()
    for idx in itemset_indices:
        all_rows.update(pruned_itemsets.iloc[idx][row_id_colname])
    
    avg_support = 0
    if itemset_indices:
        avg_support = sum(pruned_itemsets.iloc[idx]['support'] for idx in itemset_indices) / len(itemset_indices)
    
    all_columns = set()
    for idx in itemset_indices:
        itemset = pruned_itemsets.iloc[idx]['itemsets']
        for item in itemset:
            all_columns.add(extract_column_from_item(item))
    
    return {
        "itemset_indices": itemset_indices,
        "size": len(itemset_indices),
        "common_columns": common_cols,
        "all_columns": sorted(all_columns),
        "value_distributions": value_distributions,
        "row_coverage": len(all_rows),
        "row_coverage_percent": len(all_rows) / len(set().union(*(set(pruned_itemsets.iloc[i][row_id_colname]) for i in range(len(pruned_itemsets))))) if pruned_itemsets.shape[0] > 0 else 0,
        "avg_support": avg_support
    }

def cluster_hierarchically(pruned_itemsets: pd.DataFrame, row_id_colname: str, similarity_threshold: float = 0.4) -> tuple[List[Dict], Dict[str, str]]:
    """
    Perform hierarchical clustering of itemsets, removing columns with constant values and invalid clusters.
    
    Returns:
    --------
    tuple[List[Dict], Dict[str, str]]
        List of cluster metadata dictionaries and dictionary of constant columns
    """
    clusters = cluster_itemsets_by_similarity(pruned_itemsets, row_id_colname, similarity_threshold)
    
    initial_clusters = []
    for cluster_id, itemset_indices in clusters.items():
        metadata = get_cluster_metadata(itemset_indices, pruned_itemsets, row_id_colname, constant_columns={})
        metadata["cluster_id"] = cluster_id
        initial_clusters.append(metadata)
    
    # Find columns with constant values across all clusters
    constant_columns = find_constant_columns_across_clusters(initial_clusters, pruned_itemsets)
    
    # Recompute cluster metadata, excluding constant columns and invalid clusters
    cluster_metadata = []
    for cluster_id, itemset_indices in clusters.items():
        metadata = get_cluster_metadata(itemset_indices, pruned_itemsets, row_id_colname, constant_columns)
        metadata["cluster_id"] = cluster_id
        # Only include clusters with non-empty common_columns after removing constant columns
        if metadata["common_columns"]:
            cluster_metadata.append(metadata)
    
    return sorted(cluster_metadata, key=lambda x: x["size"], reverse=True), constant_columns

def print_hierarchical_clusters(clusters: List[Dict], constant_columns: Dict[str, str], pruned_itemsets: pd.DataFrame, row_id_colname: str):
    """
    Print hierarchical clustering results, including constant columns.

    Parameters:
    -----------
    clusters : List[Dict]
        List of cluster metadata dictionaries
    constant_columns : Dict[str, str]
        Dictionary of columns with constant values
    pruned_itemsets : pd.DataFrame
        DataFrame containing itemsets
    row_id_colname : str
        Name of the column containing row IDs
    """
    def _get_common_cols_str(common_cols):
        return ', '.join(common_cols)

    def _get_all_cols_str(all_cols):
        return ', '.join(all_cols)

    def _get_most_common_str(values):
        most_common = values.most_common()
        return ', '.join(f'{val} ({count})' for val, count in most_common) if most_common else ''

    def _get_itemset_dict(itemset):
        return {k: v for k, v in itemset_to_column_dict(itemset).items() if k not in constant_columns}

    def _get_percent(len_row_ids, row_coverage):
        return (len_row_ids / row_coverage * 100) if row_coverage > 0 else 0

    def _get_example_ids(row_ids):
        example_ids = ", ".join(str(id) for id in list(row_ids)[:5])
        more = f" and {len(row_ids)-5} more" if len(row_ids) > 5 else ""
        return f"{example_ids}{more}"

    print("\n" + "="*80)
    print("HIERARCHICAL CLUSTERING RESULTS")
    print("="*80)

    # Print constant columns
    if constant_columns:
        print("\nColumns with constant values across all clusters:")
        for col, val in constant_columns.items():
            print(f"  {col}: {val}")

    # Print cluster results
    for i, cluster in enumerate(clusters):
        itemset_indices = cluster["itemset_indices"]
        common_cols = cluster["common_columns"]
        all_cols = cluster["all_columns"]
        row_coverage = cluster["row_coverage"]
        row_coverage_percent = cluster["row_coverage_percent"] * 100

        print(f"\n{'-'*80}")
        print(f"CLUSTER {i}: {len(itemset_indices)} itemsets, {row_coverage} rows covered ({row_coverage_percent:.1f}%)")
        print(f"{'-'*80}")

        if common_cols:
            print(f"Common columns: {_get_common_cols_str(common_cols)}")
        print(f"All columns: {_get_all_cols_str(all_cols)}")

        if common_cols and any(col in cluster["value_distributions"] for col in common_cols):
            print("\nCommon columns and their values:")

        # Display value distributions for common columns
        for col in common_cols:
            if col not in cluster["value_distributions"]:
                continue
            values = cluster["value_distributions"][col]
            most_common_str = _get_most_common_str(values)
            if most_common_str:
                print(f"  {col}: {most_common_str}")

        # Print example itemsets
        print("\nExample itemsets:")
        for idx in itemset_indices:
            itemset = pruned_itemsets.iloc[idx]['itemsets']
            row_ids = pruned_itemsets.iloc[idx][row_id_colname]

            # Convert itemset to more readable format, excluding constant columns
            itemset_dict = _get_itemset_dict(itemset)

            print(f"  Itemset {idx}: {itemset_dict}")
            percent = _get_percent(len(row_ids), row_coverage)
            print(f"    Matching rows: {len(row_ids)} ({percent:.1f}% of cluster)")
            if len(row_ids) > 0:
                print(f"    Example row IDs: {_get_example_ids(row_ids)}")

    print("\n" + "="*80)
    print(f"TOTAL: {len(clusters)} clusters")
    print("="*80 + "\n")


# Define TypedDict for the enriched itemset structure
class EnrichedItemset(TypedDict):
    itemset: Dict[str, str]  # Column name to value mapping
    # The second key is dynamic (row_id_colname) with a list of IDs

# Type alias for the column groups dictionary
ColumnGroupDict = Dict[str, List[Dict[str, Union[Dict[str, str], List[Union[int, str]]]]]]

def group_itemsets_by_columns(
    filtered_details_list: List[Dict[str, str]],
    pruned_itemsets: pd.DataFrame,
    row_id_colname: str = ''
) -> Tuple[ColumnGroupDict, ColumnGroupDict, ColumnGroupDict]:
    """
    Group itemsets by their column names.
    
    Args:
        filtered_details_list: List of dictionaries with column-value pairs
        pruned_itemsets: DataFrame containing the original itemsets with row IDs
        row_id_colname: Name of the column containing row IDs
        
    Returns:
        Tuple of three dictionaries (very_interesting, mildly_interesting, uninteresting)
        Each dictionary maps column patterns to lists of enriched itemsets
    """
    column_groups = {}
    
    for idx, itemset in enumerate(filtered_details_list):
        # Get sorted list of column names for this itemset
        columns = sorted(itemset.keys())
        # Create consistent key regardless of column order
        column_key = "|".join(columns)
        
        # Initialize list if key doesn't exist
        if column_key not in column_groups:
            column_groups[column_key] = []
        
        # Get the row IDs for this itemset
        row_ids = []
        if row_id_colname and idx < len(pruned_itemsets) and row_id_colname in pruned_itemsets.iloc[idx]:
            row_ids = pruned_itemsets.iloc[idx][row_id_colname]
        
        # Add both the itemset details and row IDs to the group
        enriched_itemset = {
            'itemset': itemset,
            row_id_colname: row_ids
        }
        
        column_groups[column_key].append(enriched_itemset)
    
    # Split into three categories
    very_interesting_itemsets = {}
    mildly_interesting_itemsets = {}
    uninteresting_itemsets = {}
    
    for key, itemsets in column_groups.items():
        group_size = len(itemsets)
        if group_size == 1:
            very_interesting_itemsets[key] = itemsets
        elif group_size == 2:
            mildly_interesting_itemsets[key] = itemsets
        else:  # 3 or more
            uninteresting_itemsets[key] = itemsets
    
    return very_interesting_itemsets, mildly_interesting_itemsets, uninteresting_itemsets

def _print_category_header(category_num: int, category_name: str, description: str) -> None:
    """Print a formatted header for a category of itemsets."""
    print("\n" + "*"*80)
    print(f"CATEGORY {category_num}: {category_name} ({description})")
    print("*"*80)

def _print_single_itemset(
    itemset_counter: int, 
    itemset: Dict[str, str], 
    row_ids: List[Any], 
    row_id_colname: str
) -> int:
    """Print details for a single itemset and return the updated counter."""
    print(f"  Itemset {itemset_counter}:")
    print(f"    Common characteristics: {itemset}")
    print(f"    Number of matching rows: {len(row_ids)}")
    if row_ids:
        matching_ids = ", ".join(str(id) for id in row_ids)
        print(f"    Matching {row_id_colname} values: {matching_ids}")
    return itemset_counter + 1

def _print_pattern_itemsets(
    pattern: str, 
    itemsets: List[Dict[str, Any]], 
    itemset_counter: int, 
    row_id_colname: str,
    show_pattern_count: bool = False
) -> int:
    """Print all itemsets for a given pattern and return the updated counter."""
    pattern_display = f"{pattern} ({len(itemsets)} itemsets)" if show_pattern_count else pattern
    print(f"\nPattern: {pattern_display}")
    
    for item_data in itemsets:
        itemset = item_data['itemset']
        row_ids = item_data.get(row_id_colname, [])
        itemset_counter = _print_single_itemset(itemset_counter, itemset, row_ids, row_id_colname)
    
    return itemset_counter

def print_itemset_details(
    very_interesting_itemsets: Dict[str, List[Dict[str, Any]]],
    mildly_interesting_itemsets: Dict[str, List[Dict[str, Any]]],
    uninteresting_itemsets: Dict[str, List[Dict[str, Any]]],
    unassigned_items: List[Any],
    row_id_colname: str = ''
) -> None:  # sourcery skip: extract-duplicate-method
    """
    Print details for itemsets in all categories.
    
    Args:
        very_interesting_itemsets: Dictionary of unique column pattern itemsets
        mildly_interesting_itemsets: Dictionary of column patterns with 2 itemsets
        uninteresting_itemsets: Dictionary of column patterns with 3+ itemsets
        unassigned_items: List of items not belonging to any itemset
        row_id_colname: Name of the column containing row IDs
    """
    print("\n" + "="*80)
    print("CLUSTERING RESULTS SUMMARY")
    print("="*80)

    # Print very interesting itemsets (unique column patterns)
    _print_category_header(1, "VERY INTERESTING ITEMSETS", "Unique Column Patterns")
    if not very_interesting_itemsets:
        print("  No itemsets in this category.")

    itemset_counter = 0
    for pattern, itemsets in very_interesting_itemsets.items():
        itemset_counter = _print_pattern_itemsets(pattern, itemsets, itemset_counter, row_id_colname)

    # Print mildly interesting itemsets (2 per pattern)
    _print_category_header(2, "MILDLY INTERESTING ITEMSETS", "2 Itemsets with Same Columns")
    if not mildly_interesting_itemsets:
        print("  No itemsets in this category.")

    for pattern, itemsets in mildly_interesting_itemsets.items():
        itemset_counter = _print_pattern_itemsets(pattern, itemsets, itemset_counter, row_id_colname)

    # Print uninteresting itemsets (3+ per pattern)
    _print_category_header(3, "LESS INTERESTING ITEMSETS", "3+ Itemsets with Same Columns")
    if not uninteresting_itemsets:
        print("  No itemsets in this category.")

    for pattern, itemsets in uninteresting_itemsets.items():
        itemset_counter = _print_pattern_itemsets(pattern, itemsets, itemset_counter, row_id_colname, True)

    # Print unassigned items
    _print_category_header(4, f"UNASSIGNED {row_id_colname.upper()}'S", "Not in Any Itemset")
    print(f"  Number of unassigned {row_id_colname}'s: {len(unassigned_items)}")
    if unassigned_items:
        unassigned_str = ", ".join(str(id) for id in unassigned_items)
        print(f"  Unassigned {row_id_colname} values: {unassigned_str}")

    print("\n" + "="*80)
    print(f"TOTAL: {itemset_counter} itemsets and {len(unassigned_items)} unassigned {row_id_colname}'s")
    print("="*80 + "\n")

# -------------------------------------------------------------------
# collect_ranking_feedback:
#   Collect ranking feedback from the user via the CLI.
#
#   Returns a list of indices representing the user's ranking 
#   (with the first index being the highest ranked itemset).
# -------------------------------------------------------------------
def collect_ranking_feedback(pruned_itemsets: pd.DataFrame) -> Optional[List[int]]:
    """Collect ranking feedback from user."""
    while True:
        ranking_input = input("Your ranking: ").strip()
        ranking_order = [int(x.strip()) for x in ranking_input.split(",") if x.strip().isdigit()]
        if not ranking_order:
            return None
        if len(ranking_order) == len(pruned_itemsets):
            return ranking_order
        print(f"Invalid ranking: You provided {len(ranking_order)} indices, but there are {len(pruned_itemsets)} itemsets.")
        print("Please try again. Note that you may hit <Enter> if the default ranking is satisfactory")

# -------------------------------------------------------------------
# update_weights_with_ranking:
#   Update weights and γ based on ranking feedback.
#
#   The ranking_order is a list of indices indicating the user's ranking
#   (with the first index being the best).
#   For each pair (i, j) with i < j in the ranking order, the itemset at ranking_order[i] is preferred
#   over the itemset at ranking_order[j].
# -------------------------------------------------------------------
def update_weights_with_ranking(
    pruned_itemsets: pd.DataFrame, 
    ranking_order: List[int], 
    weights: Dict[str, float], 
    gamma: float, 
    lr_weights: float, 
    lr_gamma: float, 
    row_id_colname: str = ''
) -> Tuple[Dict[str, float], float]:
    """Update weights based on ranking feedback."""
    for i in range(len(ranking_order)):
        for j in range(i+1, len(ranking_order)):
            higher_idx = ranking_order[i]
            lower_idx = ranking_order[j]
            higher_itemset = pruned_itemsets.iloc[higher_idx]
            lower_itemset = pruned_itemsets.iloc[lower_idx]
            cluster_A = extract_columns(higher_itemset['itemsets'])
            cluster_B = extract_columns(lower_itemset['itemsets'])
            supp_A = higher_itemset['support']
            supp_B = lower_itemset['support']
            weights, gamma = update_parameters_with_feedback(
                weights, gamma,
                cluster_A, cluster_B,
                supp_A, supp_B,
                lr_weights=lr_weights, lr_gamma=lr_gamma
            )
    return weights, gamma


def identify_fully_correlated_columns(
    df: pl.DataFrame, 
    row_id_colname: str = ''
) -> List[Set[str]]:
    """Identify fully correlated columns."""
    # Convert to pandas DataFrame (temporary copy for computation)
    tmp_pdf = df.to_pandas()
    
    # Drop the row_id_colname column if provided and present
    if row_id_colname and row_id_colname in tmp_pdf.columns:
        tmp_pdf = tmp_pdf.drop(columns=[row_id_colname])

    # One-hot encode string columns; numeric columns remain unchanged.
    dummies = pd.get_dummies(tmp_pdf, prefix_sep="___", drop_first=False)

    # Compute the correlation matrix.
    corr = dummies.corr()

    correlated_groups = []
    visited = set()
    epsilon = 1e-12  # tolerance for float comparison
    for col in corr.columns:
        if col in visited:
            continue
        group = {col}
        for other in corr.columns:
            if other != col and other not in visited and abs(corr.loc[col, other] - 1.0) < epsilon:
                group.add(other)
        if len(group) > 1:
            visited.update(group)
            # Map dummy column names back to original column names.
            original_group = set()
            for col_name in group:
                orig = col_name.split("___")[0] if "___" in col_name else col_name
                original_group.add(orig)
            if len(original_group) > 1:
                correlated_groups.append(original_group)
    
    # Remove duplicate sets.
    unique_groups = list({frozenset(group) for group in correlated_groups})
    unique_groups = [set(g) for g in unique_groups]

    # Iteratively merge groups that overlap.
    merged = True
    while merged:
        merged = False
        new_groups = []
        while unique_groups:
            first = unique_groups.pop(0)
            merge_set = first.copy()
            indices_to_remove = []
            for i, grp in enumerate(unique_groups):
                if merge_set & grp:  # if there is any overlap
                    merge_set |= grp  # union the groups
                    indices_to_remove.append(i)
                    merged = True
            # Remove merged groups by eliminating indices_to_remove
            unique_groups = [grp for j, grp in enumerate(unique_groups) if j not in indices_to_remove]
            new_groups.append(merge_set)
        unique_groups = new_groups
    
    return unique_groups

def prune_fully_correlated_columns(
    df: pl.DataFrame, 
    row_id_colname: str = ''
) -> pl.DataFrame:
    """Prune fully correlated columns."""
    if groups := identify_fully_correlated_columns(df, row_id_colname):
        for group in groups:
            print(f"Found fully correlated columns: {group}")
            if drop_cols_input := input(
                "Enter comma separated column names from the above group that you wish to drop: "
            ).strip():
                drop_cols = [col.strip() for col in drop_cols_input.split(",") if col.strip() in group]
                if invalid_cols := [
                    col.strip()
                    for col in drop_cols_input.split(",")
                    if col.strip() not in group
                ]:
                    print(f"Warning: {invalid_cols} not in group, ignoring them.")
                if drop_cols:
                    print(f"Dropping columns: {drop_cols}")
                    # df is a Polars DataFrame; use its drop method.
                    df = df.drop(drop_cols)
                else:
                    print("No valid columns provided, no columns dropped.")
            else:
                print("No columns specified, no columns dropped.")
    return df


# --- function to prepare data ---
def prepare_data(df: pl.DataFrame, row_id_colname: str) -> pd.DataFrame:
    """Prepare data for analysis with auto-detection of column types."""
    # Handle numeric floating-point columns
    for col in df.columns:
        if df[col].dtype in [pl.Float32, pl.Float64]:
            df = df.with_columns(pl.col(col).round(0).cast(pl.Int64).alias(col))
    
    # Drop constant columns
    if constant_cols := [col for col in df.columns if len(df[col].unique()) == 1]:
        print(f"Dropping constant columns: {constant_cols}")
        df = df.drop(constant_cols)
    
    # Drop columns with all unique values (excluding row_id_colname)
    if unique_cols := [
        col for col in df.columns
        if col != row_id_colname and len(df[col].unique()) == df.height
    ]:
        print(f"Dropping columns with all unique values: {unique_cols}")
        df = df.drop(unique_cols)

    polars_df = prune_fully_correlated_columns(df, row_id_colname)
    
    return polars_df.to_pandas()


def calculate_similarity_matrix(pruned_itemsets: pd.DataFrame, row_id_colname: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the similarity matrix and linkage matrix for visualization purposes.
    Reuses the same similarity calculation logic as cluster_itemsets_by_similarity.
    
    Parameters:
    -----------
    pruned_itemsets : pd.DataFrame
        DataFrame containing itemsets
    row_id_colname : str
        Name of the column containing row IDs
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Distance matrix and linkage matrix
    """
    n_itemsets = len(pruned_itemsets)
    if n_itemsets <= 1:
        return np.zeros(0), np.zeros((0, 4))
    
    similarity_matrix = np.zeros((n_itemsets * (n_itemsets - 1)) // 2)
    
    idx = 0
    for i in range(n_itemsets):
        for j in range(i+1, n_itemsets):
            itemset_i = pruned_itemsets.iloc[i]['itemsets']
            itemset_j = pruned_itemsets.iloc[j]['itemsets']
            
            # Calculate column similarity (Jaccard)
            cols_i = {extract_column_from_item(item) for item in itemset_i}
            cols_j = {extract_column_from_item(item) for item in itemset_j}
            col_similarity = len(cols_i & cols_j) / len(cols_i | cols_j) if cols_i | cols_j else 0
            
            # Calculate row similarity (Jaccard)
            rows_i = set(pruned_itemsets.iloc[i][row_id_colname])
            rows_j = set(pruned_itemsets.iloc[j][row_id_colname])
            row_similarity = len(rows_i & rows_j) / len(rows_i | rows_j) if rows_i | rows_j else 0
            
            composite_similarity = 0.5 * col_similarity + 0.5 * row_similarity
            
            similarity_matrix[idx] = 1 - composite_similarity 
            idx += 1
    
    # Compute linkage matrix
    Z = linkage(similarity_matrix, method='average')
    
    return similarity_matrix, Z


def plot_dendrogram(pruned_itemsets: pd.DataFrame, row_id_colname: str, 
                   similarity_threshold: float = 0.4):
    """
    Plot a dendrogram of the hierarchical clustering of itemsets.
    
    Parameters:
    -----------
    pruned_itemsets : pd.DataFrame
        DataFrame containing itemsets
    row_id_colname : str
        Name of the column containing row IDs
    similarity_threshold : float
        Threshold for similarity when clustering (used for visualization)
    """
    n_itemsets = len(pruned_itemsets)
    if n_itemsets <= 1:
        print("Not enough itemsets to create a dendrogram")
        return
    
    # Get similarity and linkage matrices
    _, Z = calculate_similarity_matrix(pruned_itemsets, row_id_colname)
    
    # Plot dendrogram
    plt.figure(figsize=(12, 8))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Itemset Index')
    plt.ylabel('Distance (1 - Similarity)')
    
    threshold = 1 - similarity_threshold
    
    dendrogram(
        Z,
        leaf_rotation=90.,
        leaf_font_size=10.,
        color_threshold=threshold,
        above_threshold_color='gray'
    )
    
    plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.7, 
                label=f'Similarity Threshold: {similarity_threshold}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('dendrogram.png', dpi=300)
    plt.close()
    print("Dendrogram saved as 'dendrogram.png'")


def plot_itemset_heatmap(pruned_itemsets: pd.DataFrame, row_id_colname: str):
    """
    Plot a heatmap showing the similarity between different itemsets.
    
    Parameters:
    -----------
    pruned_itemsets : pd.DataFrame
        DataFrame containing itemsets
    row_id_colname : str
        Name of the column containing row IDs
    """
    n_itemsets = len(pruned_itemsets)
    if n_itemsets <= 1:
        print("Not enough itemsets to create a heatmap")
        return
    
    similarity_matrix, _ = calculate_similarity_matrix(pruned_itemsets, row_id_colname)
    
    square_matrix = np.zeros((n_itemsets, n_itemsets))
    square_matrix[np.triu_indices(n_itemsets, k=1)] = similarity_matrix
    square_matrix = square_matrix + square_matrix.T 
    
    similarity_square = 1 - square_matrix
    
    np.fill_diagonal(similarity_square, 1.0)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_square,
                annot=False, 
                cmap="viridis",  
                cbar_kws={'label': 'Similarity Score'},
                xticklabels=[f"{i}" for i in range(n_itemsets)],
                yticklabels=[f"{i}" for i in range(n_itemsets)])
    
    plt.title('Itemset Similarity Heatmap', fontsize=16)
    plt.xlabel('Itemset Index', fontsize=14)
    plt.ylabel('Itemset Index', fontsize=14)
    plt.tight_layout()
    plt.savefig('itemset_similarity_heatmap.png', dpi=300)
    plt.close()
    
    print("Itemset similarity heatmap saved as 'itemset_similarity_heatmap.png'")


def plot_cluster_overlap_matrix(clusters: List[Dict], pruned_itemsets: pd.DataFrame, row_id_colname: str):
    """
    Create a heatmap showing the overlap between different clusters based on row coverage.
    
    Parameters:
    -----------
    clusters : List[Dict]
        List of cluster metadata dictionaries
    pruned_itemsets : pd.DataFrame
        DataFrame containing itemsets
    row_id_colname : str
        Name of the column containing row IDs
    """
    if not clusters or len(clusters) <= 1:
        print("Not enough clusters to create an overlap matrix")
        return
    
    n_clusters = len(clusters)
    overlap_matrix = np.zeros((n_clusters, n_clusters))
    
    # Calculate the Jaccard similarity between clusters based on row coverage
    for i in range(n_clusters):
        cluster_i_rows = set()
        for idx in clusters[i]["itemset_indices"]:
            cluster_i_rows.update(pruned_itemsets.iloc[idx][row_id_colname])
        
        for j in range(n_clusters):
            cluster_j_rows = set()
            for idx in clusters[j]["itemset_indices"]:
                cluster_j_rows.update(pruned_itemsets.iloc[idx][row_id_colname])
            
            if i == j:
                overlap_matrix[i, j] = 1.0
            else:
                intersection = len(cluster_i_rows & cluster_j_rows)
                union = len(cluster_i_rows | cluster_j_rows)
                overlap_matrix[i, j] = intersection / union if union > 0 else 0
    
    # Plot overlap matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(overlap_matrix,
                annot=True,
                fmt=".2f",
                cmap="YlOrRd",
                cbar_kws={'label': 'Jaccard Similarity'},
                xticklabels=[f"Cluster {i}\n({clusters[i]['size']} itemsets)" for i in range(n_clusters)],
                yticklabels=[f"Cluster {i}\n({clusters[i]['size']} itemsets)" for i in range(n_clusters)])
    
    plt.title('Cluster Overlap Matrix', fontsize=16)
    plt.tight_layout()
    plt.savefig('cluster_overlap_matrix.png', dpi=300)
    plt.close()
    print("Cluster overlap matrix saved as 'cluster_overlap_matrix.png'")


def plot_network_graph(clusters: List[Dict], pruned_itemsets: pd.DataFrame):
    """
    Create a network graph visualization showing relationships between clusters.
    
    Parameters:
    -----------
    clusters : List[Dict]
        List of cluster metadata dictionaries
    pruned_itemsets : pd.DataFrame
        DataFrame containing itemsets
    """
    G = nx.Graph()
    
    for i, cluster in enumerate(clusters):
        G.add_node(f"Cluster {i}", 
                  size=cluster["size"], 
                  row_coverage=cluster["row_coverage"],
                  node_type="cluster")
    
    for i, cluster in enumerate(clusters):
        for idx in cluster["itemset_indices"]:
            itemset_id = f"Itemset {idx}"
            if not G.has_node(itemset_id):
                G.add_node(itemset_id, 
                          support=pruned_itemsets.iloc[idx]['support'],
                          node_type="itemset")
            
            G.add_edge(f"Cluster {i}", itemset_id, weight=1.0)
    
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    plt.figure(figsize=(14, 10))
    
    cluster_nodes = [node for node, attrs in G.nodes(data=True) if attrs.get('node_type') == 'cluster']
    cluster_sizes = [G.nodes[node]['size'] * 300 for node in cluster_nodes]
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=cluster_nodes,
                          node_size=cluster_sizes, 
                          node_color='lightblue',
                          alpha=0.8)
    
    itemset_nodes = [node for node, attrs in G.nodes(data=True) if attrs.get('node_type') == 'itemset']
    itemset_sizes = [G.nodes[node].get('support', 0.2) * 500 for node in itemset_nodes]
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=itemset_nodes,
                          node_size=itemset_sizes, 
                          node_color='lightgreen',
                          alpha=0.6)
    
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    
    plt.title('Network Graph of Clusters and Itemsets')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('cluster_network_graph.png', dpi=300)
    plt.close()
    print("Network graph saved as 'cluster_network_graph.png'")


def plot_interactive_sunburst(
    clusters: List[Dict], 
    pruned_itemsets: pd.DataFrame, 
    row_id_colname: str,
    itemset_summaries: str = None,
    very_interesting_text: str = None,
    mildly_interesting_text: str = None,
    uninteresting_text: str = None,
    cluster_reports_dir: str = "cluster_reports"
):
    """
    Create an interactive sunburst chart showing the hierarchical structure of the clusters.
    Includes detailed hover information for each cluster, itemset, including human-readable summaries
    and interest categorization if available.
    """
    executive_summary = ""
    try:
        with open("cluster_analysis_report.md", "r") as f:
            content = f.read()
            
        summary_pattern = r"## EXECUTIVE SUMMARY\s*\n([\s\S]*?)(?=\n\s*## DETAILED CLUSTER ANALYSIS|\Z)"
        match = re.search(summary_pattern, content)
        
        if match:
            executive_summary = match.group(1).strip()
            print(f"Loaded executive summary from cluster_analysis_report.md (length: {len(executive_summary)} characters)")
        else:
            print("Warning: Could not find EXECUTIVE SUMMARY section with specific end marker")
            alt_pattern = r"## EXECUTIVE SUMMARY\s*\n([\s\S]*?)(?=\n\s*##|\Z)"
            alt_match = re.search(alt_pattern, content)
            if alt_match:
                executive_summary = alt_match.group(1).strip()
                print(f"Loaded executive summary using general pattern (length: {len(executive_summary)} characters)")
            else:
                print("Warning: Could not extract executive summary")
                executive_summary = "Executive summary not available."
            
    except FileNotFoundError:
        print("Warning: cluster_analysis_report.md not found. Using default message.")
        executive_summary = "Executive summary not available. Please generate the cluster analysis report first."
    except Exception as e:
        print(f"Error reading cluster_analysis_report.md: {e}")
        executive_summary = "Executive summary could not be loaded."
    
    formatted_summary = executive_summary.replace("\n", "<br>")
    
    # Split the summary into two parts for left and right panels
    split_points = [m.start() for m in re.finditer(r'---', formatted_summary)]
    if split_points and len(split_points) > 1:
        left_part = formatted_summary[:split_points[1]]
        right_part = formatted_summary[split_points[1]:]
    else:
        halfway = len(formatted_summary) // 2
        left_part = formatted_summary[:halfway]
        right_part = formatted_summary[halfway:]
    
    # Prepare data for sunburst chart
    labels = ['Root']  
    parents = ['']     
    values = [100]     
    
    # Add hover text
    hover_data = ['Root']
    
    def itemset_to_dict(itemset):
        result = {}
        for item in itemset:
            parts = item.split("___", 1)
            col = parts[0]
            val = parts[1] if len(parts) > 1 else ""
            result[col] = val
        return result

    def get_itemset_summary(itemset_id):
        if itemset_summaries is None:
            return None
            
        pattern = rf"\*\*Itemset {itemset_id}:\*\*(.*?)(?=\n\n\*\*Itemset|\Z)"
        match = re.search(pattern, itemset_summaries, re.DOTALL)
        return match.group(1).strip() if match else None
    
    def get_itemset_category(itemset_id):
        if very_interesting_text and f"**Itemset {itemset_id}:**" in very_interesting_text:
            return "Very Interesting"
        elif mildly_interesting_text and f"**Itemset {itemset_id}:**" in mildly_interesting_text:
            return "Mildly Interesting"
        elif uninteresting_text and f"**Itemset {itemset_id}:**" in uninteresting_text:
            return "Less Interesting"
        return None
    
    def get_cluster_summary(cluster_idx):
        cluster_id = f"CLUSTER_{cluster_idx}"
        file_path = os.path.join(cluster_reports_dir, f"{cluster_id}_analysis.md")
        
        if not os.path.exists(file_path):
            return None
            
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            summary_pattern = r"#### Summary(.*?)(?=\n\n#|\Z)"
            match = re.search(summary_pattern, content, re.DOTALL)
            if match:
                summary = match.group(1).strip()
                summary = re.sub(r'\s*\n+\s*', '<br>', summary)
                summary = re.sub(r'\s+', ' ', summary)
                return summary
                
            return content[:200] + "..." if len(content) > 200 else content               
        except Exception as e:
            print(f"Error reading cluster summary: {e}")
            return None

    
    for i, cluster in enumerate(clusters):
        cluster_id = f"Cluster {i}"
        labels.append(cluster_id)
        parents.append('Root')
        values.append(cluster["row_coverage"])
        
        hover_text = f"<b>{cluster_id}</b><br>"
        hover_text += f"Itemsets: {cluster['size']}<br>"
        hover_text += f"Row coverage: {cluster['row_coverage']}<br>"
        hover_text += f"Common columns: {', '.join(cluster['common_columns']) if cluster['common_columns'] else 'None'}"

        cluster_summary = get_cluster_summary(i)
        if cluster_summary:
            hover_text += f"<br><b>Summary:</b> <i>{cluster_summary}</i><br>"

        hover_data.append(hover_text)
        
        for idx in cluster["itemset_indices"]:
            itemset = pruned_itemsets.iloc[idx]['itemsets']
            row_ids = pruned_itemsets.iloc[idx][row_id_colname]
            
            itemset_id = f"Itemset {idx}"
            labels.append(itemset_id)
            parents.append(cluster_id)
            values.append(len(row_ids))  
            
            itemset_dict = itemset_to_dict(itemset)
            
            hover_text = f"<b>{itemset_id}</b><br>"
            
            category = get_itemset_category(idx)
            if category:
                if category == "Very Interesting":
                    hover_text += f"<b style='color: #d62728;'>Category: {category}</b><br>"
                elif category == "Mildly Interesting":
                    hover_text += f"<b style='color: #ff7f0e;'>Category: {category}</b><br>"
                else:
                    hover_text += f"<b style='color: #1f77b4;'>Category: {category}</b><br>"
            
            summary = get_itemset_summary(idx)
            if summary:
                hover_text += f"<i>Summary: {summary}</i><br><br>"
            
            hover_text += f"Matching rows: {len(row_ids)} ({(len(row_ids) / cluster['row_coverage'] * 100):.1f}% of cluster)<br>"
            
            hover_data.append(hover_text)
    
    fig = px.sunburst(
        names=labels,
        parents=parents,
        values=values,
        title="Hierarchical Cluster Structure",
        hover_data=[hover_data],
        custom_data=[hover_data]  
    )
    
    fig.update_traces(
        hovertemplate='%{customdata[0]}<extra></extra>',
        insidetextorientation='radial' 
    )
    
    # Improve layout
    fig.update_layout(
        margin=dict(t=50, l=10, r=10, b=10), 
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial, sans-serif", 
            align="left",                
            namelength=-1                 
        ),
        title_x=0.5, 
        height=700   
    )
    
    left_part = left_part.replace("#### ", "<h3>").replace("\n\n", "</h3>")
    right_part = right_part.replace("#### ", "<h3>").replace("\n\n", "</h3>")
    left_part = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', left_part)
    right_part = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', right_part)
    left_part = re.sub(r'\*(.*?)\*', r'<i>\1</i>', left_part)
    right_part = re.sub(r'\*(.*?)\*', r'<i>\1</i>', right_part)
    
    # Create HTML with executive summary panels on sides
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Hierarchical Cluster Structure</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                margin: 0;
                padding: 0;
                font-family: Arial, sans-serif;
                display: flex;
                flex-direction: column;
                height: 100vh;
            }}
            .header {{
                text-align: center;
                padding: 10px;
                background-color: #f8f9fa;
                border-bottom: 1px solid #ddd;
            }}
            .container {{
                display: flex;
                flex: 1;
                overflow: hidden;
            }}
            .side-panel {{
                width: 20%;
                padding: 20px;
                overflow-y: auto;
                background-color: #f8f9fa;
                border-right: 1px solid #ddd;
            }}
            .right-panel {{
                width: 20%;
                padding: 20px;
                overflow-y: auto;
                background-color: #f8f9fa;
                border-left: 1px solid #ddd;
            }}
            .chart-container {{
                flex: 1;
                position: relative;
                height: 100%;
            }}
            h2 {{
                color: #333;
                border-bottom: 2px solid #007bff;
                padding-bottom: 10px;
                margin-top: 0;
            }}
            h3 {{
                color: #555;
                margin-top: 20px;
                font-size: 16px;
            }}
            .summary-text {{
                line-height: 1.6;
                font-size: 14px;
            }}
            .key-point {{
                margin: 10px 0;
                padding: 10px;
                background-color: #e9f7fe;
                border-left: 4px solid #007bff;
                border-radius: 4px;
            }}
            .highlight {{
                font-weight: bold;
                color: #d62728;
            }}
            /* Custom hover styling */
            .js-plotly-plot .plotly .hoverlabel {{
                max-width: 800px !important;
                box-sizing: border-box;
            }}
            .js-plotly-plot .plotly .hoverlabel .hoverlabel-text-container {{
                max-width: 780px !important;
                max-height: 600px !important;
                overflow-y: auto !important;
                white-space: normal !important;
                overflow-wrap: break-word !important;
                word-wrap: break-word !important;
                padding: 8px;
            }}
            .cluster-refs {{
                background-color: #f0f0f0;
                border-radius: 3px;
                padding: 2px 4px;
                font-family: monospace;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="side-panel">
                <h2>Executive Summary</h2>
                <div class="summary-text">
                    {left_part}
                </div>
            </div>
            <div class="chart-container" id="chart"></div>
            <div class="right-panel">
                <h2>Key Insights</h2>
                <div class="summary-text">
                    {right_part}
                </div>
            </div>
        </div>
        <script>
            // Get the figure data from the Plotly figure
            var figureData = {json.dumps(fig.to_dict(), cls=plotly.utils.PlotlyJSONEncoder)};
            
            // Create the plot
            Plotly.newPlot('chart', figureData.data, figureData.layout, {{
                responsive: true,
                displayModeBar: true
            }}).then(function() {{
                // Override hover behavior
                var gd = document.getElementById('chart');
                
                // Create a mutation observer to watch for hover elements
                var observer = new MutationObserver(function(mutations) {{
                    mutations.forEach(function(mutation) {{
                        if (mutation.addedNodes.length) {{
                            // Look for hover elements being added
                            var hovers = document.querySelectorAll('.hoverlabel-text-container');
                            hovers.forEach(function(hover) {{
                                hover.style.maxWidth = '780px';
                                hover.style.maxHeight = '600px';
                                hover.style.overflowY = 'auto';
                                hover.style.whiteSpace = 'normal';
                                hover.style.overflowWrap = 'break-word';
                            }});
                        }}
                    }});
                }});
                
                // Start observing
                observer.observe(document.body, {{ childList: true, subtree: true }});
                
                // Enhance the text display
                document.querySelectorAll('.summary-text').forEach(function(element) {{
                    // Enhance cluster references
                    element.innerHTML = element.innerHTML.replace(/CLUSTER_(\d+)/g, '<span class="cluster-refs">CLUSTER_$1</span>');
                    element.innerHTML = element.innerHTML.replace(/Cluster (\d+)/g, '<span class="cluster-refs">Cluster $1</span>');
                    
                    // Add special styling to key metrics
                    element.innerHTML = element.innerHTML.replace(/(\d+%)(?!;)/g, '<b style="color:#007bff">$1</b>');
                }});
            }});
        </script>
    </body>
    </html>
    """
    
    with open("cluster_sunburst_with_summary2.html", "w") as f:
        f.write(html_content)
    
    fig.write_html(
        "cluster_sunburst_standard2.html",
        include_plotlyjs=True,
        full_html=True
    )
    
    print("Enhanced interactive sunburst chart with executive summary saved as 'cluster_sunburst_with_summary2.html'")
    print("A standard version is also available as 'cluster_sunburst_standard2.html")
    
    return fig



def visualize_all(pruned_itemsets: pd.DataFrame, clusters: List[Dict], 
                 row_id_colname: str, similarity_threshold: float = 0.4,
                 itemset_summaries: str=None, very_interesting_text: str = None,
                 mildly_interesting_text: str = None, uninteresting_text: str = None,
                 cluster_reports_dir: str = "cluster_reports"):
    """
    Generate all visualizations for the clustering results.
    
    Parameters:
    -----------
    pruned_itemsets : pd.DataFrame
        DataFrame containing itemsets
    clusters : List[Dict]
        List of cluster metadata dictionaries
    row_id_colname : str
        Name of the column containing row IDs
    similarity_threshold : float
        Threshold for similarity when clustering
    """
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80 + "\n")
    
    plot_dendrogram(pruned_itemsets, row_id_colname, similarity_threshold)
    plot_itemset_heatmap(pruned_itemsets, row_id_colname)
    plot_cluster_overlap_matrix(clusters, pruned_itemsets, row_id_colname)
    plot_network_graph(clusters, pruned_itemsets)
    plot_interactive_sunburst(clusters, pruned_itemsets, row_id_colname, itemset_summaries, very_interesting_text, mildly_interesting_text, uninteresting_text, cluster_reports_dir)
    print("\nAll visualizations have been generated.")

def print_filtered_details_list(
    filtered_details_list: List[Dict[str, str]], 
    pruned_itemsets: pd.DataFrame, 
    row_id_colname: str
) -> None:
    """
    Print the itemsets from filtered_details_list with their indices.
    """
    print("\n" + "="*80)
    print("ITEMSETS FOR RANKING")
    print("="*80)
    
    for idx, itemset in enumerate(filtered_details_list):
        row_ids = []
        if idx < len(pruned_itemsets) and row_id_colname in pruned_itemsets.iloc[idx]:
            row_ids = pruned_itemsets.iloc[idx][row_id_colname]
        
        columns = sorted(itemset.keys())
        pattern = "|".join(columns)
        print(f"\nPattern: {pattern}")
        
        print(f"  Itemset {idx}:")
        print(f"    Common characteristics: {itemset}")
        print(f"    Number of matching rows: {len(row_ids)}")
        if row_ids:
            matching_ids = ", ".join(str(id) for id in list(row_ids)[:])
            print(f"    Matching {row_id_colname} values: {matching_ids}")
    
    print("\n" + "="*80)
    print(f"TOTAL: {len(filtered_details_list)} itemsets")
    print("="*80 + "\n")


# --- Training function ---
def train(
    TD: pd.DataFrame, 
    row_id_colname: str
) -> None:
    """Train the model using provided data."""

    config_file = CONFIG_FILE
    if os.path.exists(config_file):
        try:
            with open(config_file, "r") as f:
                config = json.load(f)
                
            weights = config.get("weights", {})
            min_support = config.get("min_support", 0.1)
            max_collection = config.get("max_collection", -1)
            gamma = config.get("gamma", 0.7)
            lr_weights = config.get("lr_weights", 0.12)
            lr_gamma = config.get("lr_gamma", 0.8)

            print("Loaded configuration from", config_file)
            
            columns = [col for col in TD.columns if col != row_id_colname]
            if missing_cols := [col for col in columns if col not in weights]:
                print(f"Adding {len(missing_cols)} missing columns to weights dictionary with default weights.")
                for col in missing_cols:
                    weights[col] = 1.0/len(columns)

        except Exception as e:
            print(f"Error loading configuration from {config_file}: {e}")
            print("Using default configuration instead.")

            columns = [col for col in TD.columns if col != row_id_colname]
            num_columns = len(columns)
            weights = {col: 1/num_columns for col in columns}
            min_support = 0.1
            max_collection = -1
            gamma = 0.7
            lr_weights = 0.12
            lr_gamma = 0.8
    
    else:
        # Initialize default configuration for training.
        columns = [col for col in TD.columns if col != row_id_colname]
        num_columns = len(columns)
        weights = {col: 1/num_columns for col in columns}
        min_support = 0.1
        max_collection = -1
        gamma = 0.7
        lr_weights = 0.12
        lr_gamma = 0.8
        print("Initialized default configuration for training.")
    
    # Obtain initial pruned maximal frequent itemsets.
    pruned_itemsets = rank_maximal_frequent_itemsets(TD, weights, min_support, max_collection, gamma, row_id_colname)
    filtered_details_list = remove_columns_with_values_common_to_all_itemsets(pruned_itemsets)

    print("Please review the following itemsets and provide your ranking.")
    print_filtered_details_list(filtered_details_list, pruned_itemsets, row_id_colname)
    
    # Collect ranking feedback.
    print("\nProvide your ranking using the itemset indices shown above.")
    print("Enter the indices in order from most preferred to least preferred, separated by commas.")
    ranking_order = collect_ranking_feedback(pruned_itemsets)
    if ranking_order is not None:
        weights, gamma = update_weights_with_ranking(pruned_itemsets, ranking_order, weights, gamma, lr_weights, lr_gamma, row_id_colname)
        pruned_itemsets = rank_maximal_frequent_itemsets(TD, weights, min_support, max_collection, gamma, row_id_colname)
        filtered_details_list = remove_columns_with_values_common_to_all_itemsets(pruned_itemsets)

        print("\nClustering results after weight updates:")
        print_filtered_details_list(filtered_details_list, pruned_itemsets, row_id_colname)
    
        # Group itemsets by columns
        very_interesting_itemsets, mildly_interesting_itemsets, uninteresting_itemsets = (
            group_itemsets_by_columns(filtered_details_list, pruned_itemsets, row_id_colname)
        )

        # Calculate unassigned items
        all_assigned_items = set()
        for _, row in pruned_itemsets.iterrows():
            all_assigned_items.update(row[row_id_colname])
            
        all_items = set(TD[row_id_colname]) if row_id_colname in TD.columns else set(TD.index)
        unassigned_items = list(all_items - all_assigned_items)
        
        print_itemset_details(
            very_interesting_itemsets, 
            mildly_interesting_itemsets, 
            uninteresting_itemsets, 
            unassigned_items, 
            row_id_colname
        )

    config_file = CONFIG_FILE
    final_config = {
        "weights": weights,
        "min_support": min_support,
        "max_collection": max_collection,
        "gamma": gamma,
        "lr_weights": lr_weights,
        "lr_gamma": lr_gamma
    }
    with open(config_file, "w") as f:
        json.dump(final_config, f, indent=4)
    print(f"Saved configuration to {config_file}.")


def generate_column_descriptions(df, pruned_itemsets, constant_columns=None, row_id_colname=None):
    """
    Generate descriptions for columns based on data values and patterns, so that LLM gets a better context of the dataset.

    Returns:
    --------
    Dict[str, str]
        Dictionary mapping column names to descriptions
    """
    def _get_example_values(unique_values):
        example_values = [str(v) for v in unique_values[:5]]
        examples_str = ", ".join(example_values)
        if len(unique_values) > 5:
            examples_str += ", etc."
        return examples_str

    def _is_categorical(col_type, n_unique):
        return(
            col_type not in ['object', 'string', 'category']
            and ('int' in col_type or 'float' in col_type)
            and n_unique <= 10
            and n_unique > 0
        ) or col_type in {'object', 'string', 'category'}

    def _get_col_type(col):
        return str(df[col].dtype)

    column_descriptions = {}
    constant_cols = constant_columns or {}

    all_itemset_columns = set()
    for _, row in pruned_itemsets.iterrows():
        itemset = row['itemsets']
        for item in itemset:
            col = item.split("___")[0]
            all_itemset_columns.add(col)

    for col in df.columns:
        if col not in all_itemset_columns:
            continue
        if col == row_id_colname:
            continue
        if col not in df.columns:
            continue

        unique_values = df[col].dropna().unique()
        n_unique = len(unique_values)
        examples_str = _get_example_values(unique_values)
        col_type = _get_col_type(col)
        is_categorical = _is_categorical(col_type, n_unique)

        if col in constant_cols:
            desc = f"Column with constant value '{constant_cols[col]}' across all clusters"
        elif is_categorical:
            desc = f"Categorical column with {n_unique} unique values. Examples: {examples_str}"
        elif 'int' in col_type:
            min_val = df[col].min()
            max_val = df[col].max()
            desc = f"Integer column with values ranging from {min_val} to {max_val}"
        elif 'float' in col_type:
            min_val = df[col].min()
            max_val = df[col].max()
            desc = f"Numeric column with values ranging from {min_val:.2f} to {max_val:.2f}"
        elif 'date' in col_type.lower() or 'time' in col_type.lower():
            desc = f"Date/time column with values ranging from {df[col].min()} to {df[col].max()}"
        elif 'bool' in col_type.lower():
            desc = f"Boolean column with values: {examples_str}"
        else:
            desc = f"Column with {n_unique} unique values. Examples: {examples_str}"

        column_descriptions[col] = desc

    return column_descriptions


class ClusterAnalysisReport:
    def __init__(self, api_key=None, model=None, base_url=None, column_descriptions=None):
        self.api_key = api_key or DEEPSEEK_API_KEY
        self.model = model or DEEPSEEK_MODEL
        self.base_url = base_url or DEEPSEEK_BASE_URL
        
        if not self.api_key:
            raise ValueError("DeepSeek API key is required. Set DEEPSEEK_API_KEY environment variable or pass api_key parameter.")
            
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.report = {
            "top_level_analysis": "",
            "detailed_cluster_analysis": {},
            "comparitive_analysis": "",
            "executive_summary": ""
        }
        self.column_descriptions = column_descriptions or {}

    def format_column_descriptions(self):
        """Format column descriptions for inclusion in prompts"""
        if not self.column_descriptions:
            return "No column descriptions provided."
            
        formatted = "COLUMN DESCRIPTIONS:\n"
        for col, desc in self.column_descriptions.items():
            formatted += f"- {col}: {desc}\n"
        return formatted

    def generate_top_level_analysis(self, clusters_data):
        # sourcery skip: class-extract-method
        """Generate a high-level analysis of the clustering results."""
        column_info = self.format_column_descriptions()

        prompt = f"""
        Analyze the following hierarchical clustering results and provide a high-level summary:

        You can refer to this column info below for a better understanding of the dataset.
        {column_info}

        {json.dumps(clusters_data, indent=2)}

        Focus on:
        1. Overall patterns across clusters
        2. Key characteristics of the major clusters
        3. Distribution of observations across clusters
        4. Common attributes across multiple clusters
        5. Unique or unexpected clusters

        Provide a well-structered analysis with key insights.
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a data science expert specializing in cluster analysis and pattern recognition."},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )

        self.report["top_level_analysis"] = response.choices[0].message.content
        return self.report["top_level_analysis"]
    
    
    def generate_detailed_cluster_analysis(self, clusters_data):
        """Generate detailed analysis for each individual cluster"""
        detailed_analyses = {}
        column_info = self.format_column_descriptions()
        
        for cluster_id, cluster_info in clusters_data.items():
            prompt = f"""
            Provide a detailed analysis of the following cluster:

            You can refer to this column info below for a better understanding of the dataset.
            {column_info}
            
            {json.dumps(cluster_info, indent=2)}
            
            Focus on:
            1. The significance of common columns and their values
            2. Detailed analysis of each itemset in this cluster
            3. Unique aspects that distinguish this cluster
            4. Provide a summary about the cluster's characteristics in just 2-3 bullet points and each point should be short sentences (Nothing after this). **Start this section with a markdown heading: `#### Summary`**
            
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a data science expert specializing in cluster analysis and pattern recognition."},
                    {"role": "user", "content": prompt},
                ],
                stream=False
            )
            
            detailed_analyses[cluster_id] = response.choices[0].message.content
        
        self.report["detailed_cluster_analysis"] = detailed_analyses
        return detailed_analyses
    
    
    def generate_comparative_analysis(self, clusters_data, top_level_analysis):
        """Generate comparative analysis between clusters"""
        prompt = f"""
        Based on the following cluster data and top-level analysis, provide a comparative analysis between clusters:
        
        CLUSTER DATA:
        {json.dumps(clusters_data, indent=2)}
        
        TOP-LEVEL ANALYSIS:
        {top_level_analysis}
        
        Focus on:
        1. Key similarities and differences between clusters
        2. Relationships or hierarchies between clusters
        3. Cross-cluster patterns that might not be apparent when looking at clusters individually
        
        Provide a comparative analysis that reveals insights across the clustering structure.
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a data science expert specializing in cluster analysis, pattern recognition, and comparative analytics."},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        
        self.report["comparative_analysis"] = response.choices[0].message.content
        return self.report["comparative_analysis"]
    
    
    def generate_executive_summary(self):
        """Generate the most interesting insights"""
        prompt = f"""
        Based on the following analyses, highlight the most interesting and actionable insights:
        
        TOP-LEVEL ANALYSIS:
        {self.report["top_level_analysis"]}
        
        COMPARATIVE ANALYSIS:
        {self.report["comparative_analysis"]}
        
        DETAILED CLUSTER ANALYSES (HIGHLIGHTS):
        {json.dumps({k: v[:500] + "..." for k, v in self.report["detailed_cluster_analysis"].items()}, indent=2)}
        
        Create an executive summary that:
        1. Highlights the 3-5 most important insights from the clustering
        2. Identifies patterns or segments that are most actionable or interesting which are not apparent
        3. Provides specific recommendations based on the cluster analysis
        4. Identifies potential areas for further investigation
        5. Summarizes the business or research implications of these findings
        
        Make the summary concise, impactful, and focused on the most valuable insights.
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a data science expert who specializes in translating complex analytical findings into clear, actionable business insights."},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        
        self.report["executive_summary"] = response.choices[0].message.content
        return self.report["executive_summary"]
    
    
    def format_full_report(self):
        """Format the full report with all sections"""

        full_report = f"""
        # HIERARCHICAL CLUSTERING ANALYSIS REPORT

        ## TOP-LEVEL ANALYSIS
        {self.report["top_level_analysis"]}

        ## COMPARATIVE ANALYSIS
        {self.report["comparative_analysis"]}

        ## EXECUTIVE SUMMARY
        {self.report["executive_summary"]}
        """

        full_report += """
        ## DETAILED CLUSTER ANALYSIS

        Detailed analyses for each individual cluster are available in the 'cluster_reports' directory.
        Each cluster has its own dedicated report file for more focused examination.
        """

        return full_report
    
    def save_individual_cluster_reports(self, output_dir="cluster_reports"):
        """Save individual markdown files for each cluster analysis"""
        os.makedirs(output_dir, exist_ok=True)

        for cluster_id, analysis in self.report["detailed_cluster_analysis"].items():
            safe_cluster_id = cluster_id.replace("/", "_").replace("\\", "_")
            filename = os.path.join(output_dir, f"{safe_cluster_id}_analysis.md")

            with open(filename, "w") as f:
                f.write(f"# {cluster_id} Analysis\n\n")
                f.write(analysis)

        print(f"Individual cluster reports saved to {output_dir}/ directory")

    
    def generate_full_report(self, clusters_data):
        """Generate the complete analysis report"""

        print("Generating top-level analysis...")
        self.generate_top_level_analysis(clusters_data)
        
        print("Generating detailed cluster analysis...")
        self.generate_detailed_cluster_analysis(clusters_data)
        
        print("Generating comparative analysis...")
        self.generate_comparative_analysis(clusters_data, self.report["top_level_analysis"])
        
        print("Generating executive summary...")
        self.generate_executive_summary()
        
        return self.format_full_report()
    
    def save_report(self, filename="cluster_analysis_report.md"):
        """Save the report to a markdown file"""
        with open(filename, "w") as f:
            f.write(self.format_full_report())
        print(f"Report saved to {filename}")


def parse_hierarchical_clustering_results(clusters, constant_columns, pruned_itemsets, row_id_colname):
    """Parse the hierarchical clustering results into a structured format"""
    clusters_data = {"constant_columns": constant_columns}
    
    # Process each cluster
    for i, cluster in enumerate(clusters):
        cluster_id = f"CLUSTER_{i}"
        
        cluster_data = {
            "size": cluster["size"],
            "row_coverage": cluster["row_coverage"],
            "row_coverage_percent": cluster["row_coverage_percent"] * 100,
            "common_columns": cluster["common_columns"],
            "all_columns": cluster["all_columns"],
            "value_distributions": {str(col): {str(val): count for val, count in counter.items()} 
                                   for col, counter in cluster["value_distributions"].items()},
            "example_itemsets": []
        }
        
        for idx in cluster["itemset_indices"]:
            itemset = pruned_itemsets.iloc[idx]['itemsets']
            row_ids = pruned_itemsets.iloc[idx][row_id_colname]
            
            itemset_dict = {k: v for k, v in itemset_to_column_dict(itemset).items() if k not in constant_columns}
            
            cluster_data["example_itemsets"].append({
                "itemset_id": idx,
                "columns": itemset_dict,
                "matching_rows_count": len(row_ids),
                "matching_rows_percent": (len(row_ids) / cluster["row_coverage"] * 100) if cluster["row_coverage"] > 0 else 0,
                "example_row_ids": list(row_ids)[:5]
            })
        
        clusters_data[cluster_id] = cluster_data
    
    return clusters_data


# --- function to generate advanced analysis report ---
def generate_advanced_analysis(clusters, constant_columns, pruned_itemsets, row_id_colname, api_key, TD=None):
    """Generate an advanced analysis report using DeepSeek API"""

    clusters_data = parse_hierarchical_clustering_results(clusters, constant_columns, pruned_itemsets, row_id_colname)

    column_descriptions = {}
    if TD is not None:
        print("Generating column descriptions...")
        column_descriptions = generate_column_descriptions(TD, pruned_itemsets, constant_columns, row_id_colname)
    
    print("Initializing cluster analysis report...")
    analyzer = ClusterAnalysisReport(api_key=api_key, column_descriptions=column_descriptions)
    
    print("Generating comprehensive cluster analysis report...")
    report = analyzer.generate_full_report(clusters_data)
    
    report_filename = "cluster_analysis_report.md"
    analyzer.save_report(report_filename)

    analyzer.save_individual_cluster_reports()
    
    print(f"Advanced analysis complete. Full report has been saved to {report_filename}")
    print("Individual cluster reports saved to cluster_reports/ directory")
    
    return report


def generate_itemset_summaries(filtered_details_list, pruned_itemsets, row_id_colname, api_key, TD=None):
    """
    Generate human-readable summaries for all itemsets.
        
    Returns:
        Markdown formatted summaries of all itemsets
    """
    api_key = api_key or DEEPSEEK_API_KEY
    model = DEEPSEEK_MODEL
    base_url = DEEPSEEK_BASE_URL
    
    if not api_key:
        raise ValueError("DeepSeek API key is required. Set DEEPSEEK_API_KEY environment variable or pass api_key parameter.")

    column_descriptions = {}
    if TD is not None:
        column_descriptions = generate_column_descriptions(TD, pruned_itemsets, row_id_colname=row_id_colname)

    column_info = "COLUMN DESCRIPTIONS:\n"
    for col, desc in column_descriptions.items():
        column_info += f"- {col}: {desc}\n"

    itemsets_data = []
    
    for idx, itemset in enumerate(filtered_details_list):
        # Get the row IDs for this itemset
        row_ids = []
        if idx < len(pruned_itemsets) and row_id_colname in pruned_itemsets.iloc[idx]:
            row_ids = pruned_itemsets.iloc[idx][row_id_colname]
        
        itemset_data = {
            "itemset_id": idx,
            "columns": itemset,
            "matching_rows_count": len(row_ids),
            "row_ids": list(row_ids),
            "total_rows": len(row_ids)
        }
        
        itemsets_data.append(itemset_data)

    prompt = f"""
    # Task: Convert Technical Itemsets to Human-Readable Summaries
    You are a data analyst who needs to translate technical itemsets into clear, human-readable summaries. Each itemset represents a group of records that share specific characteristics.
    
    ## Column Information
    Use this information to understand what each column represents:
    {column_info}
    
    ## Instructions
    For each itemset:
    1. Create a concise 1-3 sentence summary that captures the key characteristics
    2. Write in natural language, not bullet points
    3. Highlight meaningful patterns and relationships in the data
    4. Mention the number of matching records
    6. Title each summary with "**Itemset X:**" where X is the itemset number
    7. Be brief but informative.
    
    ## Input Data
    The following data shows {len(itemsets_data)} itemsets with their characteristics:
    {json.dumps(itemsets_data, indent=2)}
    
    ## Output Format
    Provide your response as markdown text with a separate paragraph for each itemset summary.
    Do not include any explanations about your approach - just provide the final summaries.
    """
    
    # Make the API call
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a data science expert specializing in data analysis and communication."},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )
    
    summaries = response.choices[0].message.content
    
    with open("itemset_summaries.md", "w") as f:
        f.write("# Itemset Summaries\n\n")
        f.write(summaries)
    
    print("Itemset summaries saved to itemset_summaries.md")
    
    return summaries
    
def categorize_itemsets_by_interest_level(
    summaries: str,
    api_key: str
) -> Tuple[str, str, str]:  # sourcery skip: extract-method
    """
    Analyze human-readable itemset summaries and categorize them directly by interest level.
        
    Returns:
        A markdown file containing the categorization of itemsets into very interesting, mildly interestiing and less interesting categories.
    """
    api_key = api_key or DEEPSEEK_API_KEY
    model = DEEPSEEK_MODEL
    base_url = DEEPSEEK_BASE_URL
    
    if not api_key:
        raise ValueError("DeepSeek API key is required. Set DEEPSEEK_API_KEY environment variable or pass api_key parameter.")

    prompt = f"""
    # Task: Categorize Itemsets by Interest Level
    
    You are a data analyst reviewing itemset summaries. Categorize each itemset into one of three levels of interest:
    1. Very Interesting: Unique, significant patterns with high analytical value
    2. Mildly Interesting: Patterns with moderate significance
    3. Uninteresting: Common or less significant patterns
    
    ## Human-Readable Summaries:
    {summaries}
    
    ## Instructions:
    For each itemset:
    1. Evaluate its significance, uniqueness, and analytical value
    2. Assign it to one of the three categories
    3. Keep the original summary text intact, including the "**Itemset X:**" format
    
    ## Output Format:
    Provide a markdown document with three sections:
    
    ## Very Interesting Itemsets
    [Insert all very interesting itemset summaries here with their original titles]
    
    ## Mildly Interesting Itemsets
    [Insert all mildly interesting itemset summaries here with their original titles]
    
    ## Less Interesting Itemsets
    [Insert all less interesting itemset summaries here with their original titles]
    
    Important: Every itemset must be included in exactly one category. Keep the original "**Itemset X:**" format.
    """
    
    # Make the API call
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a data science expert specializing in pattern analysis and interest classification."},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )

    categorized_text = response.choices[0].message.content

    very_interesting_section = ""
    mildly_interesting_section = ""
    uninteresting_section = ""

    current_section = None
    for line in categorized_text.split('\n'):
        if '## Very Interesting Itemsets' in line:
            current_section = 'very'
            continue
        elif '## Mildly Interesting Itemsets' in line:
            current_section = 'mild'
            continue
        elif '## Less Interesting Itemsets' in line:
            current_section = 'less'
            continue
            
        if current_section == 'very':
            very_interesting_section += line + '\n'
        elif current_section == 'mild':
            mildly_interesting_section += line + '\n'
        elif current_section == 'less':
            uninteresting_section += line + '\n'
    
    very_count = very_interesting_section.count('**Itemset')
    mild_count = mildly_interesting_section.count('**Itemset')
    less_count = uninteresting_section.count('**Itemset')
    
    print(f"\nItemsets categorization complete:")
    print(f"- Very interesting itemsets: {very_count}")
    print(f"- Mildly interesting itemsets: {mild_count}")
    print(f"- Less interesting itemsets: {less_count}")

    with open("itemset_categorization.md", "w") as f:
        f.write("# Itemset Categorization by Interest Level\n\n")
        
        f.write("## Very Interesting Itemsets\n")
        f.write(very_interesting_section)
        f.write("\n")
        
        f.write("## Mildly Interesting Itemsets\n")
        f.write(mildly_interesting_section)
        f.write("\n")
        
        f.write("## Less Interesting Itemsets\n")
        f.write(uninteresting_section)
    
    print("Itemset categorization saved to itemset_categorization.md")
    
    return very_interesting_section, mildly_interesting_section, uninteresting_section

def get_similar_itemsets(
    pruned_itemsets: pd.DataFrame, 
    itemset_id: int, 
    row_id_colname: str,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Get itemsets most similar to a specific itemset, ranked by similarity.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the similar itemsets with similarity scores
    """
    n_itemsets = len(pruned_itemsets)
    if n_itemsets <= 1:
        print("Not enough itemsets to calculate similarity")
        return pd.DataFrame()
    
    if itemset_id >= n_itemsets or itemset_id < 0:
        print(f"Itemset ID {itemset_id} is out of range (0-{n_itemsets-1})")
        return pd.DataFrame()

    itemset_target = pruned_itemsets.iloc[itemset_id]['itemsets']
    rows_target = set(pruned_itemsets.iloc[itemset_id][row_id_colname])

    target_col_vals = {}
    for item in itemset_target:
        col = extract_column_from_item(item)
        val = extract_value_from_item(item)
        target_col_vals[col] = val

    similarities = []
    for i in range(n_itemsets):
        if i == itemset_id:  
            continue
            
        itemset_i = pruned_itemsets.iloc[i]['itemsets']
        rows_i = set(pruned_itemsets.iloc[i][row_id_colname])

        comp_col_vals = {}
        for item in itemset_i:
            col = extract_column_from_item(item)
            val = extract_value_from_item(item)
            comp_col_vals[col] = val
        
        all_cols = set(target_col_vals.keys()).union(set(comp_col_vals.keys()))

        matching_pairs = 0
        for col in all_cols:
            if col in target_col_vals and col in comp_col_vals and target_col_vals[col] == comp_col_vals[col]:
                matching_pairs += 1

        col_val_similarity = matching_pairs / len(all_cols) if all_cols else 0

        row_similarity = len(rows_target & rows_i) / len(rows_target | rows_i) if rows_target | rows_i else 0

        composite_similarity = 0.5 * col_val_similarity + 0.5 * row_similarity

        similarities.append({
            'itemset_id': i,
            'similarity_score': composite_similarity,
            'column_value_similarity': col_val_similarity,
            'row_similarity': row_similarity
        })

    similarities_df = pd.DataFrame(similarities)
    similarities_df = similarities_df.sort_values('similarity_score', ascending=False)

    def get_itemset_details(idx):
        itemset = pruned_itemsets.iloc[idx]['itemsets']
        details = {}
        for item in itemset:
            col = extract_column_from_item(item)
            val = extract_value_from_item(item)
            details[col] = val
        return details
    
    similarities_df['itemset_details'] = similarities_df['itemset_id'].apply(get_itemset_details)
    similarities_df['matching_rows_count'] = similarities_df['itemset_id'].apply(
        lambda idx: len(pruned_itemsets.iloc[idx][row_id_colname])
    )

    return similarities_df.head(top_n)

def print_similar_itemsets(
    pruned_itemsets: pd.DataFrame,
    similarities_df: pd.DataFrame,
    target_itemset_id: int,
    row_id_colname: str
):
    """
    Print the similar itemsets in a readable format.
    
    Parameters:
    -----------
    pruned_itemsets : pd.DataFrame
        DataFrame containing the original itemsets
    similarities_df : pd.DataFrame
        DataFrame with similar itemsets (output from get_similar_itemsets)
    target_itemset_id : int
        ID of the target itemset
    row_id_colname : str
        Name of the column containing row IDs
    """
    target_itemset = pruned_itemsets.iloc[target_itemset_id]['itemsets']
    target_rows = pruned_itemsets.iloc[target_itemset_id][row_id_colname]

    target_details = {}
    for item in target_itemset:
        col = extract_column_from_item(item)
        val = extract_value_from_item(item)
        target_details[col] = val
    
    # Print target itemset info
    print("\n" + "="*80)
    print(f"TARGET ITEMSET: {target_itemset_id}")
    print("="*80)
    print(f"Characteristics: {target_details}")
    print(f"Total rows in this itemset: {len(target_rows)}")
    if len(target_rows) > 0:
        print(f"Example row IDs: {list(target_rows)[:5]}" + 
             (f" and {len(target_rows)-5} more" if len(target_rows) > 5 else ""))
    
    # Print similar itemsets
    print("\n" + "="*80)
    print("SIMILAR ITEMSETS (in descending order of similarity)")
    print("="*80)
    
    for i, (_, row) in enumerate(similarities_df.iterrows()):
        itemset_id = row['itemset_id']
        similarity = row['similarity_score']
        col_val_sim = row['column_value_similarity']
        row_sim = row['row_similarity']
        itemset_details = row['itemset_details']
        rows_count = row['matching_rows_count']

        rows = pruned_itemsets.iloc[itemset_id][row_id_colname]

        target_rows_set = set(target_rows)
        current_rows_set = set(rows)
        overlap = len(target_rows_set.intersection(current_rows_set))
        
        print(f"\n{'-'*80}")
        print(f"Rank {i+1}. ITEMSET {itemset_id} - Similarity: {similarity:.4f}")
        print(f"{'-'*80}")
        print(f"Column similarity: {col_val_sim:.4f}")
        print(f"Row similarity: {row_sim:.4f}")
        print(f"Characteristics: {itemset_details}")
        print(f"Total rows in this itemset: {rows_count}")
        print(f"Row overlap with target: {overlap} ({overlap/len(target_rows)*100:.1f}% of target, {overlap/len(rows)*100:.1f}% of this itemset)")
        
        if len(rows) > 0:
            print(f"Example row IDs: {list(rows)[:5]}" + 
                 (f" and {len(rows)-5} more" if len(rows) > 5 else ""))
    
    print("\n" + "="*80)


def get_similar_clusters(
    clusters: List[Dict],
    cluster_id: int,
    pruned_itemsets: pd.DataFrame,
    row_id_colname: str,
    top_n: int = 5
) -> pd.DataFrame:
    """
    Find clusters most similar to a specific cluster, ranked by similarity.

    Returns:
    --------
    pd.DataFrame
        DataFrame containing similar clusters with similarity scores
    """
    def _get_cluster_rows(cluster, pruned_itemsets, row_id_colname):
        rows = set()
        for idx in cluster["itemset_indices"]:
            rows.update(pruned_itemsets.iloc[idx][row_id_colname])
        return rows

    def _get_col_val_counts(cluster, pruned_itemsets):
        col_vals = {}
        for idx in cluster["itemset_indices"]:
            itemset = pruned_itemsets.iloc[idx]['itemsets']
            for item in itemset:
                col = extract_column_from_item(item)
                val = extract_value_from_item(item)
                if col not in col_vals:
                    col_vals[col] = Counter()
                col_vals[col][val] += 1
        return col_vals

    def _column_similarity(target_counts, comp_counts):
        all_values = set(target_counts.keys()) | set(comp_counts.keys())
        if len(all_values) == 1:
            single_value = next(iter(all_values))
            target_count = target_counts.get(single_value, 0)
            comp_count = comp_counts.get(single_value, 0)
            min_count = min(target_count, comp_count)
            max_count = max(target_count, comp_count)
            return min_count / max_count if max_count > 0 else 0
        dot_product = sum(target_counts.get(val, 0) * comp_counts.get(val, 0) for val in all_values)
        target_magnitude = math.sqrt(sum(count**2 for count in target_counts.values()))
        comp_magnitude = math.sqrt(sum(count**2 for count in comp_counts.values()))
        if target_magnitude > 0 and comp_magnitude > 0:
            return dot_product / (target_magnitude * comp_magnitude)
        return 0

    if cluster_id < 0 or cluster_id >= len(clusters):
        print(f"Cluster ID {cluster_id} is out of range (0-{len(clusters)-1})")
        return pd.DataFrame()

    target_cluster = clusters[cluster_id]
    target_rows = _get_cluster_rows(target_cluster, pruned_itemsets, row_id_colname)
    target_col_vals = _get_col_val_counts(target_cluster, pruned_itemsets)

    similarities = []

    for i, cluster in enumerate(clusters):
        if i == cluster_id: 
            continue

        comp_rows = _get_cluster_rows(cluster, pruned_itemsets, row_id_colname)
        row_similarity = 0
        if target_rows | comp_rows:
            row_similarity = len(target_rows & comp_rows) / len(target_rows | comp_rows)

        comp_col_vals = _get_col_val_counts(cluster, pruned_itemsets)
        all_cols = set(target_col_vals.keys()) | set(comp_col_vals.keys())

        total_similarity = 0
        for col in all_cols:
            if col not in target_col_vals or col not in comp_col_vals:
                continue
            col_similarity = _column_similarity(target_col_vals[col], comp_col_vals[col])
            total_similarity += col_similarity

        col_val_similarity = total_similarity / len(all_cols) if all_cols else 0
        composite_similarity = (0.5 * col_val_similarity) + (0.5 * row_similarity)

        similarities.append({
            'cluster_id': i,
            'similarity_score': composite_similarity,
            'column_value_similarity': col_val_similarity,
            'row_similarity': row_similarity,
            'size': cluster["size"],
            'common_columns': cluster["common_columns"],
            'row_coverage': cluster["row_coverage"],
            'itemset_indices': cluster["itemset_indices"]
        })

    similarities_df = pd.DataFrame(similarities)
    similarities_df = similarities_df.sort_values('similarity_score', ascending=False)

    return similarities_df.head(top_n)


def print_similar_clusters(
    similarities_df: pd.DataFrame,
    clusters: List[Dict],
    target_cluster_id: int,
    pruned_itemsets: pd.DataFrame,
    row_id_colname: str
):
    """
    Print similar clusters in a readable format.
    """
    target_cluster = clusters[target_cluster_id]

    def print_value_distributions(cluster):
        for col in cluster['common_columns']:
            if col not in cluster["value_distributions"]:
                continue
            values = cluster["value_distributions"][col]
            most_common = values.most_common(3)
            if not most_common:
                continue
            print(f"  {col}: {', '.join(f'{val} ({count})' for val, count in most_common)}")

    print("\n" + "="*80)
    print(f"TARGET CLUSTER: {target_cluster_id}")
    print("="*80)
    print(f"Size: {target_cluster['size']} itemsets")
    print(f"Common columns: {', '.join(target_cluster['common_columns'])}")

    print_value_distributions(target_cluster)

    print(f"Row coverage: {target_cluster['row_coverage']} rows ({target_cluster['row_coverage_percent']*100:.1f}%)")
    print(f"Itemset indices: {target_cluster['itemset_indices']}")

    print("\n" + "="*80)
    print("SIMILAR CLUSTERS (in descending order of similarity)")
    print("="*80)

    target_rows = set()
    for idx in target_cluster["itemset_indices"]:
        target_rows.update(pruned_itemsets.iloc[idx][row_id_colname])

    for i, (_, row) in enumerate(similarities_df.iterrows()):
        cluster_id = int(row['cluster_id'])
        similarity = row['similarity_score']
        col_val_sim = row['column_value_similarity']
        row_sim = row['row_similarity']

        cluster = clusters[cluster_id]

        comp_rows = set()
        for idx in cluster["itemset_indices"]:
            comp_rows.update(pruned_itemsets.iloc[idx][row_id_colname])

        row_overlap = len(target_rows & comp_rows)

        print(f"\n{'-'*80}")
        print(f"Rank {i+1}. CLUSTER {cluster_id} - Similarity: {similarity:.4f}")
        print(f"{'-'*80}")
        print(f"Column-value similarity: {col_val_sim:.4f}")
        print(f"Row similarity: {row_sim:.4f}")
        print(f"Size: {cluster['size']} itemsets")
        print(f"Common columns: {', '.join(cluster['common_columns'])}")

        print_value_distributions(cluster)

        print(f"Row coverage: {cluster['row_coverage']} rows ({cluster['row_coverage_percent']*100:.1f}%)")
        print(f"Row overlap: {row_overlap} rows ({row_overlap/len(target_rows)*100:.1f}% of target, {row_overlap/len(comp_rows)*100:.1f}% of this cluster)")
        print(f"Itemset indices: {cluster['itemset_indices']}")

    print("\n" + "="*80)


# --- Clustering function ---
def get_clusters(
    TD: pd.DataFrame, 
    row_id_colname: str,
    generate_advanced_report: bool = False,
    api_key: str = None,
    generate_visualizations: bool = False,
    find_similar_itemset: int = None,
    top_n_similar: int = 10,
    find_similar_cluster: int = None,
    top_n_similar_clusters: int = 5,
    generate_itemset_summarization_categorization: bool = False
):
    
    config_file = CONFIG_FILE
    if not os.path.exists(config_file):
        raise FileNotFoundError("Configuration file required for clustering not found. Please run training first.")
        
    with open(config_file, "r") as f:
        config = json.load(f)
        
    weights = config.get("weights")
    min_support = config.get("min_support", 0.1)
    max_collection = config.get("max_collection", 5)
    gamma = config.get("gamma", 0.7)
    
    print("Loaded configuration from JSON file for clustering.")

    # Preserved zero-weight column handling
    if zero_weight_cols := [
        col for col, weight in weights.items() if weight == 0
    ]:
        TD = TD.drop(columns=zero_weight_cols)
        for col in zero_weight_cols:
            del weights[col]
        print(f"Dropped columns with zero weights: {zero_weight_cols}")

    # Get pruned itemsets
    pruned_itemsets = rank_maximal_frequent_itemsets(TD, weights, min_support, max_collection, gamma, row_id_colname)
    
    # Get filtered details
    filtered_details_list = remove_columns_with_values_common_to_all_itemsets(pruned_itemsets)

    if find_similar_itemset is not None:
        if find_similar_itemset >= len(pruned_itemsets) or find_similar_itemset < 0:
            print(f"Error: Itemset ID {find_similar_itemset} is out of range (0-{len(pruned_itemsets)-1})")
        else:
            print(f"\nFinding itemsets similar to itemset {find_similar_itemset}...")
            similarities = get_similar_itemsets(
                pruned_itemsets,
                find_similar_itemset,
                row_id_colname,
                top_n=top_n_similar
            )
            print_similar_itemsets(pruned_itemsets, similarities, find_similar_itemset, row_id_colname)

    if generate_itemset_summarization_categorization or generate_visualizations:
        print("\Generating human-readable itemset summaries...")
        itemset_summaries = generate_itemset_summaries(
            filtered_details_list,
            pruned_itemsets,
            row_id_colname,
            api_key,
            TD
        )

        print("\nCategorizing itemsets by interest level...")
        very_interesting_text, mildly_interesting_text, uninteresting_text = categorize_itemsets_by_interest_level(
            itemset_summaries,
            api_key
        )

    # Group itemsets by columns
    very_interesting_itemsets, mildly_interesting_itemsets, uninteresting_itemsets = (
        group_itemsets_by_columns(filtered_details_list, pruned_itemsets, row_id_colname)
    )

    # Calculate unassigned items
    all_assigned_items = set()
    for _, row in pruned_itemsets.iterrows():
        all_assigned_items.update(row[row_id_colname])
        
    all_items = set(TD[row_id_colname]) if row_id_colname in TD.columns else set(TD.index)
    unassigned_items = list(all_items - all_assigned_items)
    
    print("\nClustering results:")
    print_itemset_details(
        very_interesting_itemsets, 
        mildly_interesting_itemsets, 
        uninteresting_itemsets, 
        unassigned_items, 
        row_id_colname
    )


    print("\nPerforming hierarchical clustering of itemsets...")
    clusters, constant_columns = cluster_hierarchically(pruned_itemsets, row_id_colname, similarity_threshold=0.4)
    print_hierarchical_clusters(clusters, constant_columns, pruned_itemsets, row_id_colname)

    if find_similar_cluster is not None:
        if find_similar_cluster >= len(clusters) or find_similar_cluster < 0:
            print(f"Error: Cluster ID {find_similar_cluster} is out of range (0-{len(clusters)-1})")
        else:
            print(f"\nFinding clusters similar to cluster {find_similar_cluster}...")
            cluster_similarities = get_similar_clusters(
                clusters,
                find_similar_cluster,
                pruned_itemsets,
                row_id_colname,
                top_n=top_n_similar_clusters
            )
            print_similar_clusters(cluster_similarities, clusters, find_similar_cluster, pruned_itemsets, row_id_colname)

    if generate_visualizations:
        print("\nGenerating visualizations...")
        visualize_all(pruned_itemsets, clusters, row_id_colname, similarity_threshold=0.4, itemset_summaries=itemset_summaries, very_interesting_text=very_interesting_text, mildly_interesting_text=mildly_interesting_text, uninteresting_text=uninteresting_text, cluster_reports_dir="cluster_reports")       

    if generate_advanced_report:
        if api_key is None:
            print("Warning: API key is required for advanced analysis. Skipping advanced report generation.")
        else:
            generate_advanced_analysis(clusters, constant_columns, pruned_itemsets, row_id_colname, api_key, TD)


def test_learning_rate_combinations(TD, row_id_colname):
    """Test various learning rate combinations and evaluate their performance."""
    # Learning rate ranges to test
    lr_weights_values = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    lr_gamma_values = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Initialize baseline configuration
    columns = [col for col in TD.columns if col != row_id_colname]
    num_columns = len(columns)
    weights = {col: 1/num_columns for col in columns}
    min_support = 0.1
    max_collection = -1
    gamma = 0.7

    # Initialize results table
    results = []

    # Initialize reference itemsets
    baseline_itemsets = rank_maximal_frequent_itemsets(TD, weights, min_support, max_collection, gamma, row_id_colname)

    print("Please provide your ideal ranking of the following itemsets:")
    filtered_details_list = remove_columns_with_values_common_to_all_itemsets(baseline_itemsets)
    print_filtered_details_list(filtered_details_list, baseline_itemsets, row_id_colname)
    ideal_ranking = collect_ranking_feedback(baseline_itemsets)

    if not ideal_ranking:
        print("No ranking provided. Exiting test.")
        return

    # Test each combination
    for lr_w, lr_g in itertools.product(lr_weights_values, lr_gamma_values):
        print(f"\nTesting lr_weights={lr_w}, lr_gamma={lr_g}")

        test_weights = weights.copy()
        test_gamma = gamma

        test_weights, test_gamma = update_weights_with_ranking(
            baseline_itemsets, ideal_ranking, test_weights, test_gamma, lr_w, lr_g, row_id_colname
        )

        # Compute new itemsets
        updated_itemsets = rank_maximal_frequent_itemsets(
            TD, test_weights, min_support, max_collection, test_gamma, row_id_colname
        )

        # Calculate evaluation metrics
        weight_change = sum(abs(test_weights[col] - weights[col]) for col in weights) / len(weights)
        gamma_value = test_gamma
        gamma_change = abs(test_gamma - gamma)

        current_order = list(range(len(updated_itemsets)))
        rank_distance = sum(abs(pos - ideal_ranking.index(idx)) 
                          for idx, pos in enumerate(current_order) if idx in ideal_ranking)

        results.append({
            'lr_weights': lr_w,
            'lr_gamma': lr_g,
            'weight_change': weight_change,
            'gamma_change': gamma_change,
            'rank_distance': rank_distance,
            'gamma_value': gamma_value,
        })

    print("\nLearning Rate Test Results:")
    print("=" * 80)
    print(f"{'lr_weights':<10} {'lr_gamma':<10} {'Weight Δ':<10} {'Gamma Δ':<10} {'Gamma Val':<12}")
    print("-" * 80)
    for r in sorted(results, key=lambda x: x['rank_distance']):
        print(f"{r['lr_weights']:<10.2f} {r['lr_gamma']:<10.2f} {r['weight_change']:<10.4f} "
              f"{r['gamma_change']:<10.4f} {r['gamma_value']:<12.4f}")

    return results

if __name__ == '__main__':
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "cluster"
    print(f"Running in {mode} mode.")

    generate_advanced = "--advanced" in sys.argv
    generate_visuals = "--visualize" in sys.argv
    generate_itemset_summarization_categorization = "--itemset_summarization_categorization" in sys.argv
    api_key = None

    if generate_advanced or generate_itemset_summarization_categorization or generate_visuals:
        api_key = DEEPSEEK_API_KEY
        if not api_key:
            print("Please enter your DeepSeek API key:")
            api_key = input().strip()
        if not api_key:
             print("No API key provided. Advanced analysis will be skipped.")
             generate_advanced = False
             generate_itemset_summarization_categorization = False
             generate_visuals = False
    
    db_path = DB_PATH
    
    db_uri = f"sqlite:///{db_path}"
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    print("Available tables in database:")
    for i, table in enumerate(tables):
        print(f"{i+1}. {table[0]}")
    
    table_index = int(input("Enter the number of the table you want to analyze: ")) - 1
    selected_table = tables[table_index][0]
    
    cursor.execute(f"PRAGMA table_info({selected_table});")
    columns = cursor.fetchall()
    
    print("\nAvailable columns in selected table:")
    for i, col in enumerate(columns):
        print(f"{i+1}. {col[1]} ({col[2]})")
    
    id_col_index = int(input("Enter the number of the column to use as row identifier: ")) - 1
    row_id_colname = columns[id_col_index][1]
    
    
    # Load the data using Polars with read_database_uri
    query = f"SELECT * FROM {selected_table}"
    df = pl.read_database(query=query, connection=cursor, infer_schema_length=None)

    conn.close()
    
    print(f"Loaded {df.height} rows and {df.width} columns from {selected_table}")
    
    TD = prepare_data(df, row_id_colname)
    
    if mode == "train":
        train(TD, row_id_colname)
    elif mode == "tune_lr":
        print("\n" + "="*80)
        print("LEARNING RATE OPTIMIZATION")
        print("="*80)
        print("This mode will test various combinations of learning rates to find the optimal values.")
        print("You'll be asked to provide an ideal ranking of itemsets.")
        
        results = test_learning_rate_combinations(TD, row_id_colname)
    elif mode == "cluster":
        find_similar_itemset = None
        top_n_similar = 10

        find_similar_response = input("\nDo you want to find similar itemsets? (y/n): ").strip().lower()
        if find_similar_response == 'y':
            itemset_id_input = input("Enter the target itemset ID: ").strip()
            if itemset_id_input.isdigit():
                find_similar_itemset = int(itemset_id_input)

                top_n_input = input(f"Enter number of similar itemsets to find (default: {top_n_similar}): ").strip()
                if top_n_input.isdigit():
                    top_n_similar = int(top_n_input)

                print(f"\nWill find top {top_n_similar} itemsets similar to itemset {find_similar_itemset}")


        find_similar_cluster = None
        top_n_similar_clusters = 5

        find_similar_cluster_response = input("\nDo you want to find similar clusters? (y/n): ").strip().lower()
        if find_similar_cluster_response == 'y':
            print("Cluster IDs will be available after hierarchical clustering is performed.")
            print("You'll need to specify a valid cluster ID between 0 and the total number of clusters minus 1.")
            cluster_id_input = input("Enter the target cluster ID: ").strip()

            if cluster_id_input.isdigit():
                find_similar_cluster = int(cluster_id_input)

                top_n_input = input(f"Enter number of similar clusters to find (default: {top_n_similar_clusters}): ").strip()
                if top_n_input.isdigit():
                    top_n_similar_clusters = int(top_n_input)

                print(f"\nWill find top {top_n_similar_clusters} clusters similar to cluster {find_similar_cluster}")

        get_clusters(TD, row_id_colname, generate_advanced, api_key, generate_visuals, find_similar_itemset, top_n_similar, find_similar_cluster, top_n_similar_clusters, generate_itemset_summarization_categorization)
    else:
        print("Invalid mode provided. Use 'train', 'tune_lr' or 'cluster'.")