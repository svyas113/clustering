import pandas as pd
import math
from typing import Dict, List
from collections import Counter

from itemset_mining import extract_column_from_item, extract_value_from_item

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
