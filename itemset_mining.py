import pandas as pd
import numpy as np
import hashlib
from mlxtend.frequent_patterns import fpmax
from typing import Dict, List, FrozenSet

from data_preprocessing import handle_high_cardinality_columns

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

    TD_to_encode = handle_high_cardinality_columns(TD_to_encode, threshold=200)

    df_encoded = pd.get_dummies(TD_to_encode, prefix_sep="___", dummy_na=False)

    print(f"One-hot encoded shape: {df_encoded.shape}")
    print(f"Memory usage: {df_encoded.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

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