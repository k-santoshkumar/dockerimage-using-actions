import pandas as pd
from sklearn.model_selection import train_test_split #type:ignore
from sklearn.linear_model import LogisticRegression #type:ignore
from sklearn.tree import DecisionTreeClassifier #type:ignore
from sklearn.ensemble import RandomForestClassifier #type:ignore
import joblib #type:ignore
import os

def train_models(data_path, models_path):
    # Load processed data
    df = pd.read_csv(os.path.join(data_path, 'processed_data.csv'))
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define models
    models = {
        'logistic_regression': LogisticRegression(),
        'decision_tree': DecisionTreeClassifier(),
        'random_forest': RandomForestClassifier()
    }

    # Train and save models
    os.makedirs(models_path, exist_ok=True)
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, os.path.join(models_path, f'{model_name}.joblib'))

    # Save test set for evaluation
    X_test.to_csv(os.path.join(models_path, 'X_test.csv'), index=False)
    y_test.to_csv(os.path.join(models_path, 'y_test.csv'), index=False)

if __name__ == "__main__":
    data_dir = 'artifacts/preprocessors/'
    models_dir = 'artifacts/models/'
    train_models(data_dir, models_dir)
