# src/preprocessing.py

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Compute absolute path to data file
THIS_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.abspath(os.path.join(THIS_DIR, '..', 'data', 'adult.data'))

def load_adult_data(path=DATA_PATH, test_size=0.2, random_state=42):
    """
    Loads and preprocesses the UCI Adult dataset:
      - Drops missing values
      - One-hot encodes categorical features
      - Scales numerical features to [0,1]
      - Splits into stratified train/test sets
    Returns:
      X_train, X_test, y_train, y_test
    """
    # Column names as specified by UCI
    column_names = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
    ]

    # Load raw data
    df = pd.read_csv(
        path,
        header=None,
        names=column_names,
        na_values=' ?',
        skipinitialspace=True
    )

    # Drop rows with missing values
    df.dropna(inplace=True)

    # Encode target as 0/1
    df['income'] = (df['income'] == '>50K').astype(int)

    # Separate features and labels
    X = df.drop('income', axis=1)
    y = df['income'].values

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numeric_cols     = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Build transformers (use sparse_output=False in newer sklearn)
    cat_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    num_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])

    # Combine transformers
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numeric_cols),
        ('cat', cat_pipeline, categorical_cols)
    ])

    # Fit and transform
    X_processed = preprocessor.fit_transform(X)

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    # Quick sanity check
    X_train, X_test, y_train, y_test = load_adult_data()
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Training labels distribution: {np.bincount(y_train)}")
    print(f" Testing labels distribution: {np.bincount(y_test)}")
