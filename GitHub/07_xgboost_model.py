import pandas as pd
import numpy as np
import multiprocessing
import pickle
import random
import xgboost as xgb
import warnings
import os

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from xgboost import XGBClassifier
from scipy.stats import uniform, randint

warnings.filterwarnings('ignore')

# 1. Configuration
RANDOM_SEED = 90
NUM_CLASSES = 3

INPUT_CSV = './Analysis/tongue_color_analysis_result.csv'

OUTPUT_MODEL_DIR = './Analysis/Output csv pkl'
OUTPUT_REPORT_DIR = './Analysis/Output csv pkl'

os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_REPORT_DIR, exist_ok=True)

CATEGORY_MAP_LIST = [
    ['second_R03', 0], ['second_R35', 1], ['first_R008', 2], ['first_R026', 2], ['first_R086', 1], ['first_R064', 0], ['second_R29', 2], ['first_R023', 0], ['first_R132', 2], ['second_R37', 1],
    ['first_R050', 0], ['first_R113', 0], ['first_R133', 0], ['second_R56', 1], ['second_R48', 1], ['first_R097', 2], ['first_R044', 1], ['first_R121', 0], ['first_R100', 1], ['first_R079', 1],
    ['second_R49', 2], ['first_R058', 0], ['first_R025', 1], ['first_R061', 2], ['first_R006', 1], ['first_R127', 1], ['second_R32', 1], ['first_R011', 0], ['first_R095', 1], ['first_R088', 0],
    ['second_R40', 1], ['first_R106', 2], ['second_R43', 0], ['second_R53', 2], ['second_R10', 2], ['first_R066', 1], ['first_R082', 1], ['first_R116', 1], ['first_R063', 0], ['second_R25', 2],
    ['first_R099', 1], ['first_R045', 0], ['first_R035', 1], ['first_R117', 1], ['second_R55', 2], ['second_R34', 2], ['first_R021', 1], ['second_R51', 0], ['first_R019', 1], ['second_R16', 2],
    ['second_R08', 1], ['second_R05', 2], ['second_R12', 2], ['first_R123', 1], ['second_R36', 0], ['first_R102', 2], ['second_R60', 2], ['first_R012', 0], ['first_R070', 0], ['first_R091', 1],
    ['second_R23', 1], ['first_R073', 1], ['second_R46', 0], ['first_R009', 1], ['first_R062', 1], ['first_R054', 1], ['first_R053', 0], ['first_R034', 1], ['first_R047', 1], ['first_R119', 0],
    ['first_R087', 1], ['first_R112', 1], ['second_R44', 1], ['first_R125', 1], ['first_R033', 1], ['first_R118', 2], ['first_R101', 1], ['first_R098', 0], ['first_R114', 0], ['first_R094', 1],
    ['first_R104', 1], ['second_R13', 0], ['first_R048', 1], ['first_R037', 0], ['first_R055', 2], ['first_R052', 1], ['first_R065', 1], ['first_R137', 0], ['second_R14', 2], ['second_R41', 0],
    ['first_R018', 0], ['first_R028', 0], ['first_R004', 2], ['first_R085', 2], ['second_R31', 2], ['second_R27', 1], ['second_R17', 1], ['first_R103', 1], ['first_R060', 2], ['first_R046', 1],
    ['second_R39', 0], ['second_R26', 2], ['first_R131', 1], ['first_R043', 0], ['first_R074', 1], ['second_R02', 1], ['first_R020', 2], ['second_R59', 0], ['first_R136', 1], ['first_R077', 1],
    ['first_R078', 2], ['first_R130', 1], ['first_R124', 0], ['first_R067', 1], ['second_R04', 0], ['first_R109', 1], ['first_R135', 1], ['first_R057', 0], ['first_R128', 0], ['first_R051', 1],
    ['second_R19', 0], ['first_R107', 1], ['first_R126', 1], ['first_R041', 0], ['first_R122', 1], ['first_R003', 1], ['first_R032', 1], ['first_R015', 1], ['second_R21', 1], ['first_R076', 1],
    ['first_R071', 1], ['first_R049', 0], ['first_R010', 0], ['first_R027', 1], ['first_R005', 0], ['first_R013', 1], ['second_R50', 0], ['first_R105', 0], ['first_R096', 1], ['second_R01', 1],
    ['first_R001', 1], ['first_R129', 2], ['first_R138', 2], ['second_R52', 1], ['first_R081', 1], ['first_R031', 1], ['second_R42', 2], ['first_R108', 0], ['first_R075', 1], ['second_R09', 0],
    ['first_R038', 0], ['first_R059', 1], ['second_R30', 1], ['first_R068', 1], ['second_R22', 2], ['first_R115', 0], ['second_R20', 1], ['first_R092', 0], ['second_R24', 1], ['first_R022', 1], 
    ['first_R069', 0], ['second_R54', 1], ['first_R040', 0], ['first_R093', 1], ['first_R039', 1], ['second_R07', 1], ['first_R016', 1], ['first_R072', 1], ['first_R084', 0], ['first_R036', 0], 
    ['second_R18', 0], ['first_R042', 2], ['first_R017', 1], ['first_R120', 1], ['first_R083', 2], ['first_R002', 1], ['second_R15', 2], ['first_R007', 2], ['second_R33', 1], ['first_R111', 2], 
    ['first_R134', 2], ['second_R47', 1], ['first_R030', 1], ['second_R28', 2], ['first_R090', 1], ['first_R014', 1], ['second_R06', 1], ['first_R110', 1], ['second_R58', 1], ['first_R029', 0],
    ['second_R45', 1], ['first_R056', 1], ['second_R57', 2], ['first_R080', 0], ['first_R024', 1], ['second_R38', 2], ['first_R089', 1], ['second_R11', 2]
]

TRAINSET_FILENAMES = [item[0] + '.png' for item in CATEGORY_MAP_LIST if 'aug' not in item[0] and item[0] not in [x[0] for x in [
    ['second_R24', 1], ['first_R022', 1], ['first_R069', 0], ['second_R54', 1], ['first_R040', 0], ['first_R093', 1], ['first_R039', 1], ['second_R07', 1], ['first_R016', 1], ['first_R072', 1],
    ['first_R084', 0], ['first_R036', 0], ['second_R18', 0], ['first_R042', 2], ['first_R017', 1], ['first_R120', 1], ['first_R083', 2], ['first_R002', 1], ['second_R15', 2], ['first_R007', 2],
    ['second_R33', 1], ['first_R111', 2], ['first_R134', 2], ['second_R47', 1], ['first_R030', 1], ['second_R28', 2], ['first_R090', 1], ['first_R014', 1], ['second_R06', 1], ['first_R110', 1],
    ['second_R58', 1], ['first_R029', 0], ['second_R45', 1], ['first_R056', 1], ['second_R57', 2], ['first_R080', 0], ['first_R024', 1], ['second_R38', 2], ['first_R089', 1], ['second_R11', 2]
]]]

TESTSET_FILENAMES = [
    'second_R24.png', 'first_R022.png', 'first_R069.png', 'second_R54.png', 'first_R040.png', 'first_R093.png', 'first_R039.png', 'second_R07.png', 'first_R016.png', 'first_R072.png',
    'first_R084.png', 'first_R036.png', 'second_R18.png', 'first_R042.png', 'first_R017.png', 'first_R120.png', 'first_R083.png', 'first_R002.png', 'second_R15.png', 'first_R007.png',
    'second_R33.png', 'first_R111.png', 'first_R134.png', 'second_R47.png', 'first_R030.png', 'second_R28.png', 'first_R090.png', 'first_R014.png', 'second_R06.png', 'first_R110.png',
    'second_R58.png', 'first_R029.png', 'second_R45.png', 'first_R056.png', 'second_R57.png', 'first_R080.png', 'first_R024.png', 'second_R38.png', 'first_R089.png', 'second_R11.png'
]

# 2. Data Loading and Preprocessing

def load_and_preprocess_data(file_path, category_map_list, train_filenames, test_filenames):

    df = pd.read_csv(file_path)

    columns_to_select = ['Filename', 'Whole_L', 'Whole_a', 'Whole_b', 'TB_L', 'TB_a', 'TB_b']
    df = df[columns_to_select]

    df = df.dropna(axis=0)

    category_map = {item[0].replace('.png', ''): item[1] for item in category_map_list}
    
    def get_target(filename):
        base_name = filename.split('.png')[0].replace('_aug', '')
        return category_map.get(base_name, -1)

    df['target'] = df['Filename'].apply(get_target)
    df = df[df['target'] != -1]
    df['target'] = df['target'].astype(int)

    def get_dataset_label(filename):
        base_name = filename.split('.png')[0].replace('_aug', '') + '.png'
        if base_name in train_filenames:
            return 'train'
        elif base_name in test_filenames:
            return 'test'
        return 'none'

    df['dataset'] = df['Filename'].apply(get_dataset_label)

    X = df.drop(columns=['Filename', 'dataset', 'target'])
    y = df['target']
    
    X_train = X[df['dataset'] == 'train']
    y_train = y[df['dataset'] == 'train']
    X_test = X[df['dataset'] == 'test']
    y_test = y[df['dataset'] == 'test']

    return X_train, X_test, y_train, y_test, df

# 3. Model Training and Evaluation

def train_and_evaluate_xgboost(X_train, X_test, y_train, y_test):

    print("\nTraining XGBoost model...")
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', XGBClassifier(objective='multi:softmax', eval_metric='mlogloss', use_label_encoder=False, random_state=RANDOM_SEED))
    ])

    param_distributions = {
        'xgb__n_estimators': randint(20, 400),
        'xgb__max_depth': randint(2, 9),
        'xgb__learning_rate': uniform(0.01, 0.19),
        'xgb__subsample': uniform(0.2, 0.8),
        'xgb__colsample_bytree': uniform(0.2, 0.5),
        'xgb__gamma': uniform(0.0, 0.5),
        'xgb__min_child_weight': randint(1, 5),
        'xgb__reg_alpha': uniform(0.0, 0.5),
        'xgb__reg_lambda': uniform(0.0, 2.0),
        'xgb__scale_pos_weight': uniform(1.0, 3.0)
    }

    random_search = RandomizedSearchCV(
        pipeline, param_distributions=param_distributions, n_iter=100, cv=5, 
        scoring='accuracy', n_jobs=-1, random_state=RANDOM_SEED, verbose=0
    )
    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_
    print(f"Best parameters found: {random_search.best_params_}")

    y_train_pred = best_model.predict(X_train)
    train_report = classification_report(y_train, y_train_pred, output_dict=True)
    print("\nTraining Classification Report:")
    print(classification_report(y_train, y_train_pred))

    y_test_pred = best_model.predict(X_test)
    test_report = classification_report(y_test, y_test_pred, output_dict=True)
    print("\nTest Classification Report:")
    print(classification_report(y_test, y_test_pred))

    feature_importances = best_model.named_steps['xgb'].feature_importances_
    feature_names = X_train.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    print("\nFeature Importances:")
    print(importance_df)

    return best_model, train_report, test_report, importance_df

# 4. Reporting Functions

def save_results(model, train_report, test_report, importance_df):

    model_path = os.path.join(OUTPUT_MODEL_DIR, 'xgboost_TB_color.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nTrained XGBoost model saved to {model_path}")

    train_report_path = os.path.join(OUTPUT_REPORT_DIR, 'TB_Xgb_train_report.csv')
    test_report_path = os.path.join(OUTPUT_REPORT_DIR, 'TB_Xgb_test_report.csv')
    importance_path = os.path.join(OUTPUT_REPORT_DIR, 'TB_Xgb_importance.csv')

    pd.DataFrame(train_report).transpose().to_csv(train_report_path)
    pd.DataFrame(test_report).transpose().to_csv(test_report_path)
    importance_df.to_csv(importance_path, index=False)
    print(f"Evaluation reports and feature importances saved to {OUTPUT_REPORT_DIR}")

if __name__ == '__main__':
    print("Starting XGBoost model training and evaluation")
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    category_map_with_ext = [[item[0] + '.png', item[1]] for item in CATEGORY_MAP_LIST]

    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data(INPUT_CSV, category_map_with_ext, TRAINSET_FILENAMES, TESTSET_FILENAMES)
    trained_model, train_rep, test_rep, importance_df = train_and_evaluate_xgboost(X_train, X_test, y_train, y_test)
    save_results(trained_model, train_rep, test_rep, importance_df)
    print("\nXGBoost model training and evaluation complete!")
