import polars as pl
import pandas as pd
import numpy as np
import contextlib
import warnings
from typing import List, Set

# --- function to prepare data ---
def prepare_data(df: pl.DataFrame, row_id_colname: str) -> pd.DataFrame:
    """Prepare data for analysis with auto-detection of column types."""
    # Handle numeric floating-point columns
    for col in df.columns:
        if df[col].dtype in [pl.Float32, pl.Float64]:
            df = df.with_columns(pl.col(col).round(0).cast(pl.Int64).alias(col))
    
    # Drop constant columns
    if constant_cols := [col for col in df.columns if len(df[col].unique()) == 1]:
        print(f"Dropping constant columns: {constant_cols}")
        df = df.drop(constant_cols)
    
    # Drop columns with all unique values (excluding row_id_colname)
    if unique_cols := [
        col for col in df.columns
        if col != row_id_colname and len(df[col].unique()) == df.height
    ]:
        print(f"Dropping columns with all unique values: {unique_cols}")
        df = df.drop(unique_cols)

    # polars_df = prune_fully_correlated_columns(df, row_id_colname)
    
    # return polars_df.to_pandas()
    return df.to_pandas()

def identify_fully_correlated_columns(
    df: pl.DataFrame, 
    row_id_colname: str = ''
) -> List[Set[str]]:
    """Identify fully correlated columns."""
    # Convert to pandas DataFrame (temporary copy for computation)
    tmp_pdf = df.to_pandas()
    
    # Drop the row_id_colname column if provided and present
    if row_id_colname and row_id_colname in tmp_pdf.columns:
        tmp_pdf = tmp_pdf.drop(columns=[row_id_colname])

    # One-hot encode string columns; numeric columns remain unchanged.
    dummies = pd.get_dummies(tmp_pdf, prefix_sep="___", drop_first=False)

    # Compute the correlation matrix.
    corr = dummies.corr()

    correlated_groups = []
    visited = set()
    epsilon = 1e-12  # tolerance for float comparison
    for col in corr.columns:
        if col in visited:
            continue
        group = {col}
        for other in corr.columns:
            if other != col and other not in visited and abs(corr.loc[col, other] - 1.0) < epsilon:
                group.add(other)
        if len(group) > 1:
            visited.update(group)
            # Map dummy column names back to original column names.
            original_group = set()
            for col_name in group:
                orig = col_name.split("___")[0] if "___" in col_name else col_name
                original_group.add(orig)
            if len(original_group) > 1:
                correlated_groups.append(original_group)
    
    # Remove duplicate sets.
    unique_groups = list({frozenset(group) for group in correlated_groups})
    unique_groups = [set(g) for g in unique_groups]

    # Iteratively merge groups that overlap.
    merged = True
    while merged:
        merged = False
        new_groups = []
        while unique_groups:
            first = unique_groups.pop(0)
            merge_set = first.copy()
            indices_to_remove = []
            for i, grp in enumerate(unique_groups):
                if merge_set & grp:  # if there is any overlap
                    merge_set |= grp  # union the groups
                    indices_to_remove.append(i)
                    merged = True
            # Remove merged groups by eliminating indices_to_remove
            unique_groups = [grp for j, grp in enumerate(unique_groups) if j not in indices_to_remove]
            new_groups.append(merge_set)
        unique_groups = new_groups
    
    return unique_groups

def prune_fully_correlated_columns(
    df: pl.DataFrame, 
    row_id_colname: str = ''
) -> pl.DataFrame:
    """Prune fully correlated columns."""
    if groups := identify_fully_correlated_columns(df, row_id_colname):
        for group in groups:
            print(f"Found fully correlated columns: {group}")
            if drop_cols_input := input(
                "Enter comma separated column names from the above group that you wish to drop: "
            ).strip():
                drop_cols = [col.strip() for col in drop_cols_input.split(",") if col.strip() in group]
                if invalid_cols := [
                    col.strip()
                    for col in drop_cols_input.split(",")
                    if col.strip() not in group
                ]:
                    print(f"Warning: {invalid_cols} not in group, ignoring them.")
                if drop_cols:
                    print(f"Dropping columns: {drop_cols}")
                    # df is a Polars DataFrame; use its drop method.
                    df = df.drop(drop_cols)
                else:
                    print("No valid columns provided, no columns dropped.")
            else:
                print("No columns specified, no columns dropped.")
    return df

def detect_datetime_column(series: pd.Series) -> bool:
    """Detect if a column contains datetime values."""
    if series.dtype in ['datetime64[ns]', 'datetime64', 'datetime']:
        return True

    # Try to parse string values as dates
    if series.dtype == 'object' or str(series.dtype).startswith('string'):
        sample = series.dropna().head(100)
        if len(sample) == 0:
            return False

        with contextlib.suppress(Exception):
            for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%d/%m/%Y', 
                       '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S']:
                parsed = pd.to_datetime(sample, format=fmt, errors='coerce')
                if parsed.notna().sum() / len(sample) > 0.8:  
                    return True
        with contextlib.suppress(Exception):
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='Could not infer format')
                parsed = pd.to_datetime(sample, errors='coerce')
                if parsed.notna().sum() / len(sample) > 0.8:
                    return True
    return False

def bucket_datetime_column(series: pd.Series, bucket_type: str) -> pd.Series:
    """Bucket datetime column based on specified type."""
    if not pd.api.types.is_datetime64_any_dtype(series):
        series = pd.to_datetime(series, errors='coerce')
    
    if bucket_type == 'time_of_day':
        def get_time_period(dt):
            if pd.isna(dt):
                return np.nan
            hour = dt.hour
            if 6 <= hour < 12:
                return 'Morning'
            elif 12 <= hour < 18:
                return 'Afternoon'
            elif 18 <= hour < 22:
                return 'Evening'
            else:
                return 'Night'
        return series.apply(get_time_period)
        
    elif bucket_type == 'day_of_week':
        return series.dt.day_name()
        
    elif bucket_type == 'season':
        def get_season(dt):
            if pd.isna(dt):
                return np.nan
            month = dt.month
            if month in [12, 1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Spring'
            elif month in [6, 7, 8]:
                return 'Summer'
            else:
                return 'Autumn'
        return series.apply(get_season)
        
    elif bucket_type == 'year':
        return series.dt.year.astype(str)
        
    else:
        raise ValueError(f"Unknown bucket type: {bucket_type}")
    
def bucket_categorical_column(series: pd.Series, keep_top_n: int = 10) -> pd.Series:
    """Bucket categorical column keeping top N values and grouping others."""
    value_counts = series.value_counts()
    top_values = value_counts.head(keep_top_n).index.tolist()
    
    def bucket_value(val):
        if pd.isna(val):
            return np.nan
        elif val in top_values:
            return str(val)
        else:
            return 'others'
    
    return series.apply(bucket_value)

def handle_high_cardinality_columns(TD_to_encode: pd.DataFrame, threshold: int = 200) -> pd.DataFrame:
    """Detect and handle columns with high cardinality."""
    high_cardinality_cols = []
    datetime_cols = []

    # Detect high cardinality columns
    for col in TD_to_encode.columns:
        unique_count = TD_to_encode[col].nunique()
        if unique_count > threshold:
            high_cardinality_cols.append(col)
            if detect_datetime_column(TD_to_encode[col]):
                datetime_cols.append(col)

    if not high_cardinality_cols:
        return TD_to_encode

    print("\n" + "="*80)
    print("HIGH CARDINALITY COLUMNS DETECTED")
    print("="*80)
    print(f"\nThe following columns have more than {threshold} unique categories:")
    for col in high_cardinality_cols:
        unique_count = TD_to_encode[col].nunique()
        col_type = "datetime" if col in datetime_cols else "categorical"
        print(f"  - {col}: {unique_count} unique values ({col_type})")

    print("\nThese columns may cause memory issues during frequent itemset mining.")
    print("You need to choose how to handle each column.\n")

    def _process_datetime_col(TD_to_encode, col):
        print("\nDatetime bucketing options:")
        print("  1. By time of day (morning, afternoon, evening, night)")
        print("  2. By day of week (Monday, Tuesday, etc.)")
        print("  3. By season (spring, summer, autumn, winter)")
        print("  4. By year")
        print("  5. Drop this column")
        while True:
            choice = input("\nEnter your choice (1-5): ").strip()
            if choice == '1':
                TD_to_encode[col] = bucket_datetime_column(TD_to_encode[col], 'time_of_day')
                print(f"✓ Bucketed {col} by time of day")
                break
            elif choice == '2':
                TD_to_encode[col] = bucket_datetime_column(TD_to_encode[col], 'day_of_week')
                print(f"✓ Bucketed {col} by day of week")
                break
            elif choice == '3':
                TD_to_encode[col] = bucket_datetime_column(TD_to_encode[col], 'season')
                print(f"✓ Bucketed {col} by season")
                break
            elif choice == '4':
                TD_to_encode[col] = bucket_datetime_column(TD_to_encode[col], 'year')
                print(f"✓ Bucketed {col} by year")
                break
            elif choice == '5':
                TD_to_encode = TD_to_encode.drop(columns=[col])
                print(f"✓ Dropped column {col}")
                break
            else:
                print("Invalid choice. Please enter a number between 1 and 5.")
        return TD_to_encode

    def _process_categorical_col(TD_to_encode, col):
        print("\nCategorical bucketing options:")
        print("  1. Keep top 10 values, group others")
        print("  2. Keep top 20 values, group others")
        print("  3. Keep top 50 values, group others")
        print("  4. Keep top 100 values, group others")
        print("  5. Drop this column")
        while True:
            choice = input("\nEnter your choice (1-5): ").strip()
            if choice == '1':
                TD_to_encode[col] = bucket_categorical_column(TD_to_encode[col], 10)
                print(f"✓ Bucketed {col} keeping top 10 values")
                break
            elif choice == '2':
                TD_to_encode[col] = bucket_categorical_column(TD_to_encode[col], 20)
                print(f"✓ Bucketed {col} keeping top 20 values")
                break
            elif choice == '3':
                TD_to_encode[col] = bucket_categorical_column(TD_to_encode[col], 50)
                print(f"✓ Bucketed {col} keeping top 50 values")
                break
            elif choice == '4':
                TD_to_encode[col] = bucket_categorical_column(TD_to_encode[col], 100)
                print(f"✓ Bucketed {col} keeping top 100 values")
                break
            elif choice == '5':
                TD_to_encode = TD_to_encode.drop(columns=[col])
                print(f"✓ Dropped column {col}")
                break
            else:
                print("Invalid choice. Please enter a number between 1 and 5.")
        return TD_to_encode

    # Process each high cardinality column
    for col in high_cardinality_cols:
        print(f"\n{'-'*60}")
        print(f"Column: {col}")
        print(f"Unique values: {TD_to_encode[col].nunique()}")

        sample_values = TD_to_encode[col].dropna().unique()[:5]
        print(f"Sample values: {list(sample_values)}")

        if col in datetime_cols:
            TD_to_encode = _process_datetime_col(TD_to_encode, col)
        else:
            TD_to_encode = _process_categorical_col(TD_to_encode, col)

    print("\n" + "="*80)
    print("High cardinality columns handled successfully!")
    print("="*80 + "\n")

    return TD_to_encode
    
def generate_column_descriptions(df, pruned_itemsets, constant_columns=None, row_id_colname=None):
    """
    Generate descriptions for columns based on data values and patterns, so that LLM gets a better context of the dataset.

    Returns:
    --------
    Dict[str, str]
        Dictionary mapping column names to descriptions
    """
    def _get_example_values(unique_values):
        example_values = [str(v) for v in unique_values[:5]]
        examples_str = ", ".join(example_values)
        if len(unique_values) > 5:
            examples_str += ", etc."
        return examples_str

    def _is_categorical(col_type, n_unique):
        return(
            col_type not in ['object', 'string', 'category']
            and ('int' in col_type or 'float' in col_type)
            and n_unique <= 10
            and n_unique > 0
        ) or col_type in {'object', 'string', 'category'}

    def _get_col_type(col):
        return str(df[col].dtype)

    column_descriptions = {}
    constant_cols = constant_columns or {}

    all_itemset_columns = set()
    for _, row in pruned_itemsets.iterrows():
        itemset = row['itemsets']
        for item in itemset:
            col = item.split("___")[0]
            all_itemset_columns.add(col)

    for col in df.columns:
        if col not in all_itemset_columns:
            continue
        if col == row_id_colname:
            continue
        if col not in df.columns:
            continue

        unique_values = df[col].dropna().unique()
        n_unique = len(unique_values)
        examples_str = _get_example_values(unique_values)
        col_type = _get_col_type(col)
        is_categorical = _is_categorical(col_type, n_unique)

        if col in constant_cols:
            desc = f"Column with constant value '{constant_cols[col]}' across all clusters"
        elif is_categorical:
            desc = f"Categorical column with {n_unique} unique values. Examples: {examples_str}"
        elif 'int' in col_type:
            min_val = df[col].min()
            max_val = df[col].max()
            desc = f"Integer column with values ranging from {min_val} to {max_val}"
        elif 'float' in col_type:
            min_val = df[col].min()
            max_val = df[col].max()
            desc = f"Numeric column with values ranging from {min_val:.2f} to {max_val:.2f}"
        elif 'date' in col_type.lower() or 'time' in col_type.lower():
            desc = f"Date/time column with values ranging from {df[col].min()} to {df[col].max()}"
        elif 'bool' in col_type.lower():
            desc = f"Boolean column with values: {examples_str}"
        else:
            desc = f"Column with {n_unique} unique values. Examples: {examples_str}"

        column_descriptions[col] = desc

    return column_descriptions