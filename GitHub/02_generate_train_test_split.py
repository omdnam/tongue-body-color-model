import pandas as pd
from sklearn.model_selection import train_test_split
import os

def generate_split():

    # 1. Configuration
    input_excel_path = './Analysis/Input excel/Fatigue_tongue_color_DB_198.xlsx'
    test_size = 0.2
    random_state = 90
    
    if not os.path.exists(input_excel_path):
        print(f"Error: Input file not found at '{input_excel_path}'")
        return

    # 2. Load Data
    try:
        df = pd.read_excel(input_excel_path, sheet_name='DB')
        df = df[['Filename', '최종진단']].copy()
        df.rename(columns={'최종진단': 'target'}, inplace=True)
        df.dropna(inplace=True)
        
    except Exception as e:
        print(f"Error reading or processing Excel file: {e}")
        return

    # 3. Perform Stratified Split
    X = df['Filename']
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # 4. Print the resulting lists
    print("# --- Train Set Filenames ---")
    train_filenames = sorted(list(X_train))
    print("trainset_filenames = [")
    for i in range(0, len(train_filenames), 5):
        line = ", ".join([f"'{f}'" for f in train_filenames[i:i+5]])
        print(f"    {line},")
    print("]\n")

    print("# --- Test Set Filenames ---")
    test_filenames = sorted(list(X_test))
    print("testset_filenames = [")
    for i in range(0, len(test_filenames), 5):
        line = ", ".join([f"'{f}'" for f in test_filenames[i:i+5]])
        print(f"    {line},")
    print("]\n")
    
    print(f"Total files: {len(df)}")
    print(f"Training set size: {len(X_train)} ({len(X_train)/len(df):.0%})")
    print(f"Test set size: {len(X_test)} ({len(X_test)/len(df):.0%})")


if __name__ == "__main__":
    generate_split()