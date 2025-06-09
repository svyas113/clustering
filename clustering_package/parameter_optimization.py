import pandas as pd
import numpy as np
import itertools
from typing import Dict, List, Tuple

from itemset_mining import extract_columns, rank_maximal_frequent_itemsets, remove_columns_with_values_common_to_all_itemsets
from display_utils import print_filtered_details_list
from utils import collect_ranking_feedback

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
