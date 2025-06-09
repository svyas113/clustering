import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
from scipy.cluster.hierarchy import fcluster, linkage

from itemset_mining import extract_column_from_item, extract_value_from_item

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