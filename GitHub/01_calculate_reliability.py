

import pandas as pd
import numpy as np
import statsmodels.stats.inter_rater as irr
from sklearn.metrics import cohen_kappa_score
import os

def calculate_intra_rater_reliability(df, assessor_name):

    rate1_col = f'{assessor_name}_1'
    rate2_col = f'{assessor_name}_2'


    if rate1_col not in df.columns or rate2_col not in df.columns:
        print(f"Columns for '{assessor_name}' not found. Skipping.")
        return

    rate1 = df[rate1_col].dropna()
    rate2 = df[rate2_col].dropna()


    common_indices = rate1.index.intersection(rate2.index)
    rate1 = rate1.loc[common_indices]
    rate2 = rate2.loc[common_indices]

    if len(rate1) == 0:
        print(f"Assessor '{assessor_name}': No valid data pairs found.")
        return

    print(f"--- {assessor_name} Intra-rater Reliability ---")


    cohens_kappa = cohen_kappa_score(rate1, rate2)
    linear_kappa = cohen_kappa_score(rate1, rate2, weights='linear')
    quadratic_kappa = cohen_kappa_score(rate1, rate2, weights='quadratic')

    print(f"Cohen's Kappa: {cohens_kappa:.3f}")
    print(f"Linear Weighted Kappa: {linear_kappa:.3f}")
    print(f"Quadratic Weighted Kappa: {quadratic_kappa:.3f}")

    n_subjects = len(rate1)
    kappa_values = {
        "Cohen's kappa": cohens_kappa,
        "Linear weighted kappa": linear_kappa,
        "Quadratic weighted kappa": quadratic_kappa
    }

    for name, kappa in kappa_values.items():
        ase = np.sqrt((kappa * (1 - kappa)) / (n_subjects - 1))
        ci_lower = kappa - 1.96 * ase
        ci_upper = kappa + 1.96 * ase
        print(f"{name} - ASE: {ase:.3f}, 95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
    
    print("-" * 50 + "\n")


def calculate_inter_rater_reliability(df):

    print("--- Inter-rater Reliability (Assessors 1-5) ---")

    rater_columns = [f'Assess{i}_1' for i in range(1, 6)]
    ratings_df = df[rater_columns].dropna()
    ratings = np.array(ratings_df)

    if ratings.shape[0] == 0:
        print("No valid data for inter-rater reliability calculation.")
        return

    N = ratings.shape[0]
    n = ratings.shape[1]
    
    categories = np.unique(ratings)
    k = len(categories)


    count_matrix = np.zeros((N, k))
    for i in range(N):
        for j, category in enumerate(categories):
            count_matrix[i, j] = np.sum(ratings[i] == category)


    try:
        kappa = irr.fleiss_kappa(count_matrix, method='fleiss')

        p_j = np.sum(count_matrix, axis=0) / (N * n)
        P_e = np.sum(p_j**2)
        
        if P_e == 1:
            se_kappa = 0
        else:
            p_i = (np.sum(count_matrix**2, axis=1) - n) / (n * (n - 1))
            P_bar = np.mean(p_i)
            
            var_kappa_num = 2 * ( (1-P_e) * (1-P_e - (P_bar - P_e)*(2*n-3)) + (P_bar-P_e)**2 * (n-2) )
            var_kappa_den = N * n * (n-1) * (1-P_e)**2
            se_kappa = np.sqrt(var_kappa_num / var_kappa_den) if var_kappa_den > 0 else 0


        ci_lower = kappa - 1.96 * se_kappa
        ci_upper = kappa + 1.96 * se_kappa

        print(f"Fleiss' Kappa: {kappa:.3f}")
        print(f"Asymptotic Standard Error: {se_kappa:.3f}")
        print(f"Asymptotic 95% Confidence Interval: ({ci_lower:.3f}, {ci_upper:.3f})")

    except Exception as e:
        print(f"Could not calculate Fleiss' Kappa. Error: {e}")

    print("-" * 50 + "\n")


def main():

    # 1. Load Data and Configuration
    file_path = './Analysis/Input excel/Reliability.xlsx'

    if not os.path.exists(file_path):
        print(f"Error: Input file not found at '{file_path}'")
        return

    try:
        df = pd.read_excel(file_path, sheet_name='3분형')
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    # 2. Calculate Intra-rater Reliability
    assessor_names = [f'Assess{i}' for i in range(1, 6)] + ['Maj']
    for name in assessor_names:
        calculate_intra_rater_reliability(df, name)

    # 3. Calculate Inter-rater Reliability
    calculate_inter_rater_reliability(df)


if __name__ == "__main__":
    main()