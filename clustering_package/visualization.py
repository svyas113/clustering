import os
import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import plotly.express as px
import plotly
from typing import Dict, List
from scipy.cluster.hierarchy import dendrogram

from clustering import calculate_similarity_matrix

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