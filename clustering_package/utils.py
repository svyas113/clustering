from typing import Dict, List, Tuple, Optional
import pandas as pd
from typing import Dict, List, Union, TypedDict

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