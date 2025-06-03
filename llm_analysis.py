import os
import json
from typing import Tuple
from openai import OpenAI
from dotenv import load_dotenv

from data_preprocessing import generate_column_descriptions
from itemset_mining import itemset_to_column_dict

load_dotenv(dotenv_path="env/.env")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

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
        {json.dumps({k: f"{v[:500]}..." for k, v in self.report["detailed_cluster_analysis"].items()}, indent=2)}
        
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