import os
import sqlite3
import polars as pl
import json
from litellm import completion
from dotenv import load_dotenv

from .data_preprocessing import prepare_data
from .itemset_mining import rank_maximal_frequent_itemsets
from .clustering import cluster_hierarchically
from .llm_analysis import parse_hierarchical_clustering_results

load_dotenv(dotenv_path="env/.env")

class DatabaseInsightReport:
    def __init__(self, api_key, model="deepseek/deepseek-chat"):
        self.api_key = api_key
        self.model = model
        self.report = {
            "top_level_analysis": {},
            "detailed_table_analysis": {},
            "detailed_view_analysis": {},
            "column_dictionaries": {},
            "executive_summary": {}
        }

    def parse_llm_response_to_dict(self, response_text, analysis_type):
        """Parse LLM response into structured dictionary based on analysis type"""
        result = {}
        
        if analysis_type == "table":
            sections_found = False
            
            if "## 1." in response_text or "## Purpose and Role" in response_text:
                sections_found = True
                parts = response_text.split("##")
                for part in parts[1:]:
                    lines = part.strip().split("\n", 1)
                    if lines:
                        header = lines[0].strip()
                        content = lines[1].strip() if len(lines) > 1 else ""
                        
                        # Determine the key based on header content
                        if "purpose" in header.lower() and "role" in header.lower():
                            result["purpose_and_role"] = content
                        elif "key column" in header.lower():
                            result["key_columns"] = content
                        elif "common pattern" in header.lower():
                            result["common_patterns"] = content
                        elif "data quality" in header.lower() or "anomal" in header.lower():
                            result["data_quality_issues"] = content
                        elif "key insight" in header.lower() or "distribution" in header.lower():
                            result["key_insights"] = content
            
            # Try numbered list format
            if not sections_found:
                sections = {
                    "purpose_and_role": ["1. The purpose and role", "1. Purpose and Role", "Purpose and Role:"],
                    "key_columns": ["2. Key columns", "2. Key Columns", "Key Columns:"],
                    "common_patterns": ["3. Common patterns", "3. Common Patterns", "Common Patterns:"],
                    "data_quality_issues": ["4. Potential data quality", "4. Data Quality", "Data Quality Issues:"],
                    "key_insights": ["5. Key insights", "5. Key Insights", "Key Insights:"],
                }
                
                for key, markers in sections.items():
                    for marker in markers:
                        start_idx = response_text.find(marker)
                        if start_idx != -1:
                            end_idx = len(response_text)
                            for other_key, other_markers in sections.items():
                                if other_key != key:
                                    for other_marker in other_markers:
                                        other_idx = response_text.find(other_marker, start_idx + len(marker))
                                        if other_idx != -1 and other_idx < end_idx:
                                            end_idx = other_idx
                            
                            summary_idx = response_text.find("#### Summary", start_idx)
                            if summary_idx != -1 and summary_idx < end_idx:
                                end_idx = summary_idx
                            
                            content = response_text[start_idx + len(marker):end_idx].strip()
                            content = content.lstrip(":").strip()
                            result[key] = content
                            break
            
            summary_markers = ["#### Summary", "## Summary", "Summary:", "**Summary**"]
            for marker in summary_markers:
                summary_idx = response_text.find(marker)
                if summary_idx != -1:
                    summary_content = response_text[summary_idx + len(marker):].strip()
                    dict_idx = summary_content.find("## Data Dictionary")
                    if dict_idx == -1:
                        dict_idx = summary_content.find("Data Dictionary Entry")
                    if dict_idx != -1:
                        summary_content = summary_content[:dict_idx].strip()
                    result["summary"] = summary_content
                    break
            
            # Extract data dictionary entry if present
            dict_markers = ["## Data Dictionary Entry", "Data Dictionary Entry:", "**Data Dictionary Entry**"]
            for marker in dict_markers:
                dict_idx = response_text.find(marker)
                if dict_idx != -1:
                    dict_content = response_text[dict_idx + len(marker):].strip()
                    result["data_dictionary_entry"] = dict_content
                    break
                    
        elif analysis_type == "view":
            sections = {
                "purpose_and_role": ["1. The purpose and role", "1. Purpose and Role", "Purpose:"],
                "data_transformation": ["2. How this view transforms", "2. Data Transformation", "Transformation:"],
                "business_logic": ["3. The business logic", "3. Business Logic", "Business Logic:"],
                "key_columns": ["4. Key columns", "4. Key Columns", "Key Columns:"],
                "use_cases": ["5. Potential use cases", "5. Use Cases", "Use Cases:"],
            }
            
            for key, markers in sections.items():
                for marker in markers:
                    start_idx = response_text.find(marker)
                    if start_idx != -1:
                        end_idx = len(response_text)
                        for other_key, other_markers in sections.items():
                            if other_key != key:
                                for other_marker in other_markers:
                                    other_idx = response_text.find(other_marker, start_idx + len(marker))
                                    if other_idx != -1 and other_idx < end_idx:
                                        end_idx = other_idx
                        
                        summary_idx = response_text.find("#### Summary", start_idx)
                        if summary_idx != -1 and summary_idx < end_idx:
                            end_idx = summary_idx
                        
                        content = response_text[start_idx + len(marker):end_idx].strip()
                        content = content.lstrip(":").strip()
                        result[key] = content
                        break
            
            summary_markers = ["#### Summary", "## Summary", "Summary:"]
            for marker in summary_markers:
                summary_idx = response_text.find(marker)
                if summary_idx != -1:
                    result["summary"] = response_text[summary_idx + len(marker):].strip()
                    break
                    
        elif analysis_type == "top_level":
            sections_map = {
                "## 1. Database Design": "database_design",
                "## 2. Key Tables and Relationships": "key_tables_and_relationships",
                "## 3. Common Patterns": "common_patterns",
                "## 4. Views Usage": "views_usage",
                "## 5. Anomalies": "anomalies",
                "## 6. Cluster Insights": "clusters_reveal"
            }
            
            for header, key in sections_map.items():
                start_idx = response_text.find(header)
                if start_idx != -1:
                    end_idx = len(response_text)
                    for other_header in sections_map.keys():
                        if other_header != header:
                            other_idx = response_text.find(other_header, start_idx + len(header))
                            if other_idx != -1 and other_idx < end_idx:
                                end_idx = other_idx
                    
                    content = response_text[start_idx + len(header):end_idx].strip()
                    result[key] = content
                        
        elif analysis_type == "executive":
            sections_map = {
                "## 1. Important Insights": "important_insights",
                "## 2. Data Patterns": "data_patterns",
                "## 3. Quality Issues": "quality_issues",
                "## 4. Recommendations": "recommendations",
                "## 5. Further Investigation": "further_investigation"
            }
            
            for header, key in sections_map.items():
                start_idx = response_text.find(header)
                if start_idx != -1:
                    end_idx = len(response_text)
                    for other_header in sections_map.keys():
                        if other_header != header:
                            other_idx = response_text.find(other_header, start_idx + len(header))
                            if other_idx != -1 and other_idx < end_idx:
                                end_idx = other_idx
                    
                    content = response_text[start_idx + len(header):end_idx].strip()
                    result[key] = content
        else:
            return self.parse_column_dictionary(response_text)
        
        result["full_response"] = response_text
        
        # If we didn't find any sections, try to parse as a general structure
        if len(result) <= 1:  
            lines = response_text.split("\n")
            current_section = None
            current_content = []
            
            for line in lines:
                if line.strip() and (line.strip()[0].isdigit() or line.startswith("**") or line.startswith("##")):
                    if current_section:
                        result[f"section_{len(result)}"] = "\n".join(current_content).strip()
                    current_section = line.strip()
                    current_content = []
                elif current_section:
                    current_content.append(line)
            
            if current_section and current_content:
                result[f"section_{len(result)}"] = "\n".join(current_content).strip()
        
        return result
    
    def parse_column_dictionary(self, response_text):
        """Parse column dictionary response into structured format"""
        columns = {}
        
        parts = response_text.split("##")
        
        for part in parts[1:]:
            lines = part.strip().split("\n")
            if not lines:
                continue
                
            header = lines[0].strip()
            column_name = header.split("(")[0].strip()
            
            column_info = {
                "header": header,
                "purpose": "",
                "statistics": "",
                "data_quality": "",
                "relationships": "",
                "business_logic": ""
            }
            
            # Parse the sections
            current_section = None
            current_content = []
            
            for line in lines[1:]:
                if line.startswith("- **Purpose**:"):
                    if current_section:
                        column_info[current_section] = "\n".join(current_content).strip()
                    current_section = "purpose"
                    current_content = [line.replace("- **Purpose**:", "").strip()]
                elif line.startswith("- **Statistics**:"):
                    if current_section:
                        column_info[current_section] = "\n".join(current_content).strip()
                    current_section = "statistics"
                    current_content = [line.replace("- **Statistics**:", "").strip()]
                elif line.startswith("- **Data Quality**:"):
                    if current_section:
                        column_info[current_section] = "\n".join(current_content).strip()
                    current_section = "data_quality"
                    current_content = [line.replace("- **Data Quality**:", "").strip()]
                elif line.startswith("- **Relationships**:"):
                    if current_section:
                        column_info[current_section] = "\n".join(current_content).strip()
                    current_section = "relationships"
                    current_content = [line.replace("- **Relationships**:", "").strip()]
                elif line.startswith("- **Business Logic**:"):
                    if current_section:
                        column_info[current_section] = "\n".join(current_content).strip()
                    current_section = "business_logic"
                    current_content = [line.replace("- **Business Logic**:", "").strip()]
                elif current_section:
                    current_content.append(line)
            
            if current_section:
                column_info[current_section] = "\n".join(current_content).strip()
            
            columns[column_name] = column_info
        
        return {
            "columns": columns,
            "full_response": response_text
        }

    def generate_table_analysis(self, table_name, table_metadata, clustering_results):
        """Generate detailed analysis for a table based on clustering results"""
        prompt = f"""
        Provide a detailed analysis of the following database table based on hierarchical clustering results.
        
        IMPORTANT: Structure your response with clear sections using ## headers for each main point.

        TABLE NAME: {table_name}
        
        TABLE STRUCTURE:
        {json.dumps(table_metadata, indent=2)}
        
        CLUSTERING RESULTS:
        {json.dumps(clustering_results, indent=2)}
        
        Please provide your analysis in the following format:
        
        ## 1. Purpose and Role
        [Describe the purpose and role of this table in the database based on discovered patterns]
        
        ## 2. Key Columns
        [Explain key columns and their significance in forming clusters]
        
        ## 3. Common Patterns
        [Describe common patterns and relationships discovered in the data]
        
        ## 4. Data Quality Issues
        [List potential data quality issues or anomalies detected]
        
        ## 5. Key Insights
        [Provide key insights about the data distribution and column relationships]
        
        #### Summary
        [Provide 2-3 bullet points with short sentences summarizing the table's characteristics]
        
        ## Data Dictionary Entry
        [Create a comprehensive data dictionary entry for this table that highlights the discovered patterns]
        """
        
        response = completion(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a database expert specializing in data pattern recognition and database design."},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        
        response_text = response.choices[0].message.content
        return self.parse_llm_response_to_dict(response_text, "table")
    
    def generate_view_analysis(self, view_name, view_metadata, view_definition, sample_data):
        """Generate detailed analysis for a view based on its definition and sample data"""
        prompt = f"""
        Provide a detailed analysis of the following database view.
        
        IMPORTANT: Structure your response with clear sections using ## headers for each main point.

        VIEW NAME: {view_name}
        
        VIEW STRUCTURE:
        {json.dumps(view_metadata, indent=2)}
        
        VIEW DEFINITION:
        {view_definition}
        
        SAMPLE DATA:
        {sample_data}
        
        Please provide your analysis in the following format:
        
        ## 1. Purpose and Role
        [Describe the purpose and role of this view in the database]
        
        ## 2. Data Transformation
        [Explain how this view transforms or presents data from underlying tables]
        
        ## 3. Business Logic
        [Describe the business logic implemented by this view]
        
        ## 4. Key Columns
        [List and explain key columns and their significance]
        
        ## 5. Use Cases
        [Describe potential use cases for this view]
        
        #### Summary
        [Provide 2-3 bullet points with short sentences summarizing the view's characteristics]
        
        ## Data Dictionary Entry
        [Create a comprehensive data dictionary entry for this view]
        """
        
        response = completion(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a database expert specializing in data visualization, SQL, and database design."},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        
        response_text = response.choices[0].message.content
        return self.parse_llm_response_to_dict(response_text, "view")
    
    def generate_top_level_analysis(self, db_metadata, all_clustering_results, view_results):
        """Generate a high-level analysis of the database based on clustering across all tables"""
        prompt = f"""
        Analyze the following database structure and analysis results to provide a high-level summary.
        
        IMPORTANT: Structure your response with clear sections using ## headers for each main point.

        DATABASE STRUCTURE:
        {json.dumps(db_metadata, indent=2)}
        
        CLUSTERING RESULTS ACROSS TABLES:
        {json.dumps(all_clustering_results, indent=2)}
        
        VIEWS ANALYSIS:
        {json.dumps({v: view_results[v]["definition"][:300] + "..." for v in view_results}, indent=2)}

        Please provide your analysis in the following format:
        
        ## 1. Database Design
        [Describe overall database design and organization]
        
        ## 2. Key Tables and Relationships
        [Explain key tables, views, and their relationships as revealed by pattern analysis]
        
        ## 3. Common Patterns
        [Describe common patterns discovered across different tables]
        
        ## 4. Views Usage
        [Explain how views are being used to transform or present the underlying table data]
        
        ## 5. Anomalies
        [List anomalies or interesting patterns that might indicate data issues or business rules]
        
        ## 6. Cluster Insights
        [Explain how the discovered clusters reveal the underlying data model and business logic]
        """

        response = completion(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a database expert specializing in database design, pattern recognition, and data modeling."},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )

        response_text = response.choices[0].message.content
        self.report["top_level_analysis"] = self.parse_llm_response_to_dict(response_text, "top_level")
        return self.report["top_level_analysis"]
    
    def generate_column_dictionary(self, table_name, table_metadata, column_eda, table_analysis):
        """Generate detailed column-level data dictionary for a table"""
        prompt = f"""
        Create a comprehensive column-level data dictionary for the table {table_name}.
        
        TABLE METADATA (SCHEMA):
        {json.dumps(table_metadata, indent=2)}
        
        COLUMN EDA STATISTICS:
        {json.dumps(column_eda, indent=2)}
        
        TABLE SUMMARY AND ANALYSIS:
        {json.dumps(table_analysis, indent=2)}
        
        DATABASE CONTEXT:
        {json.dumps(self.report["top_level_analysis"], indent=2)}

        For each column in the table, based on DATABASE CONTEXT, TABLE SUMMARY AND ANALYSIS, COLUMN EDA and TABLE METADATA, provide:
        1. Column name and data type
        2. Description of the column's purpose and role in the table as well as database
        3. Key statistics from EDA with 5 most frequent values (if available)
        4. Data quality observations (nulls, outliers, distributions)
        5. Potential relationships with other tables or columns
        6. Business logic implications based on observed patterns
        
        Format each column description with these sections:
        - **Purpose**: [1-2 sentence description of the column's purpose]
        - **Statistics**: [Key statistics in bullet points]
        - **Data Quality**: [Observations about data quality]
        - **Relationships**: [Potential relationships with other tables/columns]
        - **Business Logic**: [Business rules or implications evident from the data]
        
        Start each column with a ## header including its name and type.
        Make the descriptions concise but informative, focusing on patterns discovered in the data.
        """
        
        response = completion(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a database expert specializing in data dictionary creation, data profiling, and metadata documentation."},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        
        response_text = response.choices[0].message.content
        return self.parse_column_dictionary(response_text)
    
    def generate_executive_summary(self, db_metadata, all_clustering_results, view_results):
        """Generate the most interesting insights from the database analysis"""
        table_analyses_summary = {}
        for table, analysis in self.report["detailed_table_analysis"].items():
            table_analyses_summary[table] = {
                "purpose": analysis.get("purpose_and_role", "")[:300] + "...",
                "key_insights": analysis.get("key_insights", "")[:300] + "..."
            }
        
        view_analyses_summary = {}
        for view, analysis in self.report["detailed_view_analysis"].items():
            view_analyses_summary[view] = {
                "purpose": analysis.get("purpose_and_role", "")[:300] + "...",
                "business_logic": analysis.get("business_logic", "")[:300] + "..."
            }

        prompt = f"""
        Based on the following database structure and analyses, highlight the most interesting and actionable insights.
        
        IMPORTANT: Structure your response with clear sections using ## headers for each main point.
        
        DATABASE STRUCTURE:
        {json.dumps(db_metadata, indent=2)}
        
        TOP-LEVEL ANALYSIS:
        {json.dumps(self.report["top_level_analysis"], indent=2)}
        
        DETAILED TABLE ANALYSES (HIGHLIGHTS):
        {json.dumps(table_analyses_summary, indent=2)}
        
        DETAILED VIEW ANALYSES (HIGHLIGHTS):
        {json.dumps(view_analyses_summary, indent=2)}
        
        Please provide your executive summary in the following format:
        
        ## 1. Important Insights
        [Highlight the 3-5 most important insights about this database based on the analysis]
        
        ## 2. Data Patterns
        [Identify data patterns that reveal interesting business rules or domain knowledge]
        
        ## 3. Quality Issues
        [Point out potential data quality issues or optimization opportunities]
        
        ## 4. Recommendations
        [Provide specific recommendations based on the discovered patterns]
        
        ## 5. Further Investigation
        [Identify potential areas for further investigation]
        """

        response = completion(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a data science expert who specializes in translating complex data patterns into clear, actionable business insights."},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )

        response_text = response.choices[0].message.content
        self.report["executive_summary"] = self.parse_llm_response_to_dict(response_text, "executive")
        return self.report["executive_summary"]
    
    def save_individual_reports(self):
        """Save individual reports for each table and view in JSON format"""
        # Create directories if they don't exist
        os.makedirs("table_reports", exist_ok=True)
        os.makedirs("view_reports", exist_ok=True)
        os.makedirs("column_dictionaries", exist_ok=True) 
        
        # Save table reports as JSON
        for table_name, analysis in self.report["detailed_table_analysis"].items():
            safe_name = table_name.replace("/", "_").replace("\\", "_")
            with open(f"table_reports/{safe_name}_analysis.json", "w") as f:
                json.dump({
                    "table_name": table_name,
                    "analysis": analysis
                }, f, indent=2)
        
        # Save view reports as JSON
        for view_name, analysis in self.report["detailed_view_analysis"].items():
            safe_name = view_name.replace("/", "_").replace("\\", "_")
            with open(f"view_reports/{safe_name}_analysis.json", "w") as f:
                json.dump({
                    "view_name": view_name,
                    "analysis": analysis
                }, f, indent=2)
        
        # Save column dictionaries as JSON
        for table_name, column_dict in self.report["column_dictionaries"].items():
            safe_name = table_name.replace("/", "_").replace("\\", "_")
            with open(f"column_dictionaries/{safe_name}_columns.json", "w") as f:
                json.dump({
                    "table_name": table_name,
                    "column_dictionary": column_dict
                }, f, indent=2)
    
    def generate_full_report(self, db_metadata, table_results, view_results):
        """Generate the complete analysis report"""
        all_clustering_results = {table: results["clustering"] for table, results in table_results.items()}
        
        print("Generating top-level analysis...")
        self.generate_top_level_analysis(db_metadata, all_clustering_results, view_results)
        
        print("Generating detailed table analyses...")
        for table_name, results in table_results.items():
            print(f"Analyzing table: {table_name}")
            table_analysis = self.generate_table_analysis(
                table_name, 
                results["metadata"], 
                results["clustering"]
            )
            self.report["detailed_table_analysis"][table_name] = table_analysis
        
        print("Generating detailed view analyses...")
        for view_name, results in view_results.items():
            print(f"Analyzing view: {view_name}")
            view_analysis = self.generate_view_analysis(
                view_name,
                results["metadata"],
                results["definition"],
                results["sample_data"]
            )
            self.report["detailed_view_analysis"][view_name] = view_analysis
        
        print("Generating column-level data dictionaries...")
        for table_name, results in table_results.items():
            print(f"Creating column dictionary for: {table_name}")
            column_eda = read_column_eda(table_name)
            
            column_dict = self.generate_column_dictionary(
                table_name,
                results["metadata"],
                column_eda,
                self.report["detailed_table_analysis"][table_name]
            )
            self.report["column_dictionaries"][table_name] = column_dict
        
        print("Generating executive summary...")
        self.generate_executive_summary(db_metadata, all_clustering_results, view_results)
        
        # Save individual reports
        self.save_individual_reports()
        
        return self.report
    
    def save_main_report(self, filename="data_dictionary.json"):
        """Save the complete report to a JSON file"""
        with open(filename, "w") as f:
            json.dump(self.report, f, indent=2)
        print(f"Main report saved to {filename}")
        
        # Also save a summary JSON with just the keys for easy navigation
        summary = {
            "database_metadata": {
                "tables": list(self.report["detailed_table_analysis"].keys()),
                "views": list(self.report["detailed_view_analysis"].keys()),
                "column_dictionaries": list(self.report["column_dictionaries"].keys())
            },
            "report_structure": {
                "top_level_analysis": list(self.report["top_level_analysis"].keys()),
                "table_analysis_keys": list(self.report["detailed_table_analysis"].get(
                    list(self.report["detailed_table_analysis"].keys())[0] if self.report["detailed_table_analysis"] else "", 
                    {}
                ).keys()),
                "view_analysis_keys": list(self.report["detailed_view_analysis"].get(
                    list(self.report["detailed_view_analysis"].keys())[0] if self.report["detailed_view_analysis"] else "", 
                    {}
                ).keys()),
                "executive_summary_keys": list(self.report["executive_summary"].keys())
            }
        }
        
        with open("data_dictionary_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print("Summary structure saved to data_dictionary_summary.json")

def read_column_eda(table_name):
        """Read column EDA from the JSON file for the given table"""
        eda_file_path = f"{table_name}_eda.json"
        if not os.path.exists(eda_file_path):
            print(f"Warning: Column EDA file not found for table {table_name}: {eda_file_path}")
            return {}
        
        try:
            with open(eda_file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading column EDA file for table {table_name}: {str(e)}")
            return {}

def analyze_table_with_clustering(db_path, table_name, row_id_colname):
    """Analyze a single table using hierarchical clustering"""
    conn = sqlite3.connect(db_path)
    
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = cursor.fetchall()
    
    columns_info = []
    for col in columns:
        col_id, name, type_name, not_null, default_val, is_pk = col
        columns_info.append({
            "name": name,
            "type": type_name,
            "not_null": bool(not_null),
            "default_value": default_val,
            "is_primary_key": bool(is_pk)
        })
    
    cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
    row_count = cursor.fetchone()[0]
    
    cursor.execute(f"PRAGMA foreign_key_list({table_name});")
    foreign_keys = cursor.fetchall()
    
    fk_info = []
    for fk in foreign_keys:
        id, seq, table_name_fk, from_col, to_col, on_update, on_delete, match = fk
        fk_info.append({
            "table": table_name_fk,
            "from_column": from_col,
            "to_column": to_col,
            "on_update": on_update,
            "on_delete": on_delete
        })
    
    query = f"SELECT * FROM {table_name}"
    df = pl.read_database(query=query, connection=cursor, infer_schema_length=None)
    
    TD = prepare_data(df, row_id_colname)
    
    columns = [col for col in TD.columns if col != row_id_colname]
    num_columns = len(columns)
    weights = {col: 1/num_columns for col in columns}
    min_support = 0.1
    max_collection = -1  
    gamma = 0.5
    
    pruned_itemsets = rank_maximal_frequent_itemsets(TD, weights, min_support, max_collection, gamma, row_id_colname)
    
    clusters, constant_columns = cluster_hierarchically(pruned_itemsets, row_id_colname, similarity_threshold=0.4)
    
    clusters_data = parse_hierarchical_clustering_results(clusters, constant_columns, pruned_itemsets, row_id_colname)
    
    table_metadata = {
        "columns": columns_info,
        "row_count": row_count,
        "foreign_keys": fk_info
    }
    
    conn.close()
    
    return {
        "metadata": table_metadata, 
        "clustering": clusters_data
    }

def analyze_view(db_path, view_name):
    """Analyze a single view based on its definition and sample data"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='view' AND name='{view_name}';")
    definition = cursor.fetchone()[0]
    
    cursor.execute(f"SELECT * FROM {view_name} LIMIT 1;")
    columns = [description[0] for description in cursor.description]
    
    cursor.execute(f"SELECT COUNT(*) FROM {view_name};")
    row_count = cursor.fetchone()[0]
    
    cursor.execute(f"SELECT * FROM {view_name} LIMIT 10;")
    rows = cursor.fetchall()
    
    sample_data = f"Column names: {', '.join(columns)}\n\n"
    for i, row in enumerate(rows):
        formatted_row = []
        for val in row:
            str_val = str(val)
            if len(str_val) > 50:
                str_val = f"{str_val[:47]}..."
            formatted_row.append(str_val)
        
        sample_data += f"Row {i+1}: {', '.join(formatted_row)}\n"
    
    view_metadata = {
        "columns": columns,
        "row_count": row_count
    }
    
    conn.close()
    
    return {
        "metadata": view_metadata,
        "definition": definition,
        "sample_data": sample_data
    }

def main():
    # Database path
    db_path = "/Users/sanchitsatija/DB_SUMMARY.db"
    
    # Check if the file exists
    if not os.path.exists(db_path):
        print(f"Error: Database file not found at {db_path}")
        return
    
    print(f"Analyzing database: {db_path}")
    
    # Get API key
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("Please enter your DeepSeek API key:")
        api_key = input().strip()
    if not api_key:
        print("No API key provided. Analysis will be skipped.")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [table[0] for table in cursor.fetchall()]
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='view';")
    views = [view[0] for view in cursor.fetchall()]
    
    db_metadata = {
        "database_path": db_path,
        "tables": tables,
        "views": views,
        "table_count": len(tables),
        "view_count": len(views)
    }
    
    conn.close()
    
    # For each table, perform clustering analysis
    table_results = {}
    
    for table in tables:
        print(f"\nAnalyzing table: {table}")
        
        # Ask user for row ID column for this table
        print(f"Specify row ID column for table {table}:")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table});")
        columns = cursor.fetchall()
        conn.close()
        
        print("\nAvailable columns:")
        for i, col in enumerate(columns):
            print(f"{i+1}. {col[1]} ({col[2]})")
        
        id_col_index = int(input("Enter the number of the column to use as row identifier: ")) - 1
        row_id_colname = columns[id_col_index][1]
        
        results = analyze_table_with_clustering(db_path, table, row_id_colname)
        table_results[table] = results
    
    view_results = {}
    
    for view in views:
        print(f"\nAnalyzing view: {view}")
        results = analyze_view(db_path, view)
        view_results[view] = results
    
    analyzer = DatabaseInsightReport(api_key=api_key)
    
    print("\nGenerating comprehensive database analysis report...")
    report = analyzer.generate_full_report(db_metadata, table_results, view_results)
    
    analyzer.save_main_report("data_dictionary.json")
    
    print("\nDatabase pattern analysis complete!")
    print("- Main report saved to data_dictionary.json")
    print("- Summary structure saved to data_dictionary_summary.json")
    print(f"- Individual table analyses saved to table_reports/ directory ({len(table_results)} JSON files)")
    print(f"- Individual view analyses saved to view_reports/ directory ({len(view_results)} JSON files)")
    print(f"- Column-level data dictionaries saved to column_dictionaries/ directory ({len(table_results)} JSON files)")

if __name__ == "__main__":
    main()