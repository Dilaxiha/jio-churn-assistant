import os
import pandas as pd

# Automatically locate the project root directory regardless of where the script is executed
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Look for subscribers.csv in the Data folder, falling back to the script's own folder
csv_path = os.path.join(project_root, 'Data', 'subscribers.csv')
if not os.path.exists(csv_path):
    csv_path = os.path.join(script_dir, 'subscribers.csv')

# >>> CRITICAL ADDITION: Load the dataframe using the resolved path <<<
df = pd.read_csv(csv_path)

# Save the report into the Reports folder regardless of the current working directory
reports_dir = os.path.join(project_root, 'Reports')
os.makedirs(reports_dir, exist_ok=True)
report_path = os.path.join(reports_dir, 'eda_initial_report.txt')

# 2. Open a text file to write the structured report
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("="*60 + "\n")
    f.write("JIO RETENTION - INITIAL EDA & DATA QUALITY REPORT\n")
    f.write("="*60 + "\n\n")
    
    # Dataset Dimensions
    f.write("1. DATASET SHAPE\n")
    f.write(f"   - Total Rows (Subscribers): {df.shape[0]}\n")
    f.write(f"   - Total Columns: {df.shape[1]}\n\n")
    
    # Duplicate Check
    duplicates = df.duplicated().sum()
    f.write("2. DUPLICATE CHECK\n")
    f.write(f"   - Duplicate rows found: {duplicates}\n\n")
    
    # Missing Values Analysis
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    f.write("3. MISSING VALUES PER COLUMN\n")
    if len(missing) == 0:
        f.write("   - No missing values found.\n")
    else:
        for col, count in missing.items():
            pct = (count / len(df)) * 100
            f.write(f"   - {col}: {count} missing ({pct:.2f}%)\n")
    f.write("\n")
    
    # Data Types Summary
    f.write("4. DATA TYPES SUMMARY\n")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        f.write(f"   - {dtype}: {count} columns\n")
    f.write("\n")
    
    # Statistical Summary for Key Numeric Columns
    f.write("5. KEY NUMERIC STATISTICS\n")
    numeric_summary = df[['arpu_last_month_inr', 'tenure_months', 'plan_price_inr', 'days_since_last_recharge', 'data_gb_last_month']].describe()
    f.write(str(numeric_summary))
    f.write("\n\n")
    
    # Target Variable Distribution
    f.write("6. TARGET VARIABLE DISTRIBUTION (churn_flag_30d)\n")
    churn_counts = df['churn_flag_30d'].value_counts(dropna=False)
    churn_pcts = df['churn_flag_30d'].value_counts(normalize=True, dropna=False) * 100
    for val, count in churn_counts.items():
        f.write(f"   - {val}: {count} subscribers ({churn_pcts[val]:.2f}%)\n")

print(f"EDA report successfully created and saved as '{report_path}'.")