import pandas as pd
from typing import Dict, List, Any

from itemset_mining import itemset_to_column_dict

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