import sys
import os
import json
import sqlite3
import polars as pl
import pandas as pd
from dotenv import load_dotenv

load_dotenv(dotenv_path="env/.env")
DB_PATH = os.environ.get("DB_PATH")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")

from data_preprocessing import prepare_data
from itemset_mining import rank_maximal_frequent_itemsets, remove_columns_with_values_common_to_all_itemsets
from clustering import cluster_hierarchically
from display_utils import print_filtered_details_list, print_itemset_details, print_hierarchical_clusters
from parameter_optimization import update_weights_with_ranking, test_learning_rate_combinations
from utils import collect_ranking_feedback, group_itemsets_by_columns
from visualization import visualize_all
from llm_analysis import generate_itemset_summaries, categorize_itemsets_by_interest_level, generate_advanced_analysis
from similarity_search import get_similar_itemsets, print_similar_itemsets, get_similar_clusters, print_similar_clusters

def train(
    TD: pd.DataFrame, 
    row_id_colname: str,
    table_name: str = ''
) -> None:
    """Train the model using provided data."""

    config_dir = "ranking_config"
    os.makedirs(config_dir, exist_ok=True)

    config_file = os.path.join(config_dir, f"{table_name}_config.json")
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

    config_file = os.path.join(config_dir, f"{table_name}_config.json")
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
    generate_itemset_summarization_categorization: bool = False,
    table_name: str = ''
):
    
    config_dir = "ranking_config"
    config_file = os.path.join(config_dir, f"{table_name}_config.json")
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

def get_user_preferences() -> dict:
    """Get user preferences for clustering analysis."""
    preferences = {
        "find_similar_itemset": None,
        "top_n_similar": 10,
        "find_similar_cluster": None,
        "top_n_similar_clusters": 5
    }
    
    # Ask about similar itemsets
    find_similar_response = input("\nDo you want to find similar itemsets? (y/n): ").strip().lower()
    if find_similar_response == 'y':
        itemset_id_input = input("Enter the target itemset ID: ").strip()
        if itemset_id_input.isdigit():
            preferences["find_similar_itemset"] = int(itemset_id_input)
            
            top_n_input = input(f"Enter number of similar itemsets to find (default: {preferences['top_n_similar']}): ").strip()
            if top_n_input.isdigit():
                preferences["top_n_similar"] = int(top_n_input)
            
            print(f"\nWill find top {preferences['top_n_similar']} itemsets similar to itemset {preferences['find_similar_itemset']}")
    
    # Ask about similar clusters
    find_similar_cluster_response = input("\nDo you want to find similar clusters? (y/n): ").strip().lower()
    if find_similar_cluster_response == 'y':
        print("Cluster IDs will be available after hierarchical clustering is performed.")
        print("You'll need to specify a valid cluster ID between 0 and the total number of clusters minus 1.")
        cluster_id_input = input("Enter the target cluster ID: ").strip()
        
        if cluster_id_input.isdigit():
            preferences["find_similar_cluster"] = int(cluster_id_input)
            
            top_n_input = input(f"Enter number of similar clusters to find (default: {preferences['top_n_similar_clusters']}): ").strip()
            if top_n_input.isdigit():
                preferences["top_n_similar_clusters"] = int(top_n_input)
            
            print(f"\nWill find top {preferences['top_n_similar_clusters']} clusters similar to cluster {preferences['find_similar_cluster']}")
    
    return preferences

def main():
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
        train(TD, row_id_colname, table_name=selected_table)
    elif mode == "tune_lr":
        tuning_learning_rate(TD, row_id_colname)
    elif mode == "cluster":
        preferences = get_user_preferences()

        get_clusters(
            TD, 
            row_id_colname,
            generate_advanced_report=generate_advanced,
            api_key=api_key,
            generate_visualizations=generate_visuals,
            find_similar_itemset=preferences["find_similar_itemset"],
            top_n_similar=preferences["top_n_similar"],
            find_similar_cluster=preferences["find_similar_cluster"],
            top_n_similar_clusters=preferences["top_n_similar_clusters"],
            generate_itemset_summarization_categorization=generate_itemset_summarization_categorization,
            table_name=selected_table
        )
    else:
        print("Invalid mode provided. Use 'train', 'tune_lr' or 'cluster'.")

def tuning_learning_rate(TD, row_id_colname):
    print("\n" + "="*80)
    print("LEARNING RATE OPTIMIZATION")
    print("="*80)
    print("This mode will test various combinations of learning rates to find the optimal values.")
    print("You'll be asked to provide an ideal ranking of itemsets.")

    results = test_learning_rate_combinations(TD, row_id_colname)

if __name__ == '__main__':
    main()