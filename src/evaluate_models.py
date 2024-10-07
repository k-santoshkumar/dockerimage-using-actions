import pandas as pd
import joblib #type:ignore
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score #type:ignore

def evaluate_models(models_path, metrics_path):
    # Load test data
    X_test = pd.read_csv(os.path.join(models_path, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(models_path, 'y_test.csv'))

    # Load models
    model_files = [f for f in os.listdir(models_path) if f.endswith('.joblib') and 'test' not in f]
    metrics = {}

    for model_file in model_files:
        model = joblib.load(os.path.join(models_path, model_file))
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        metrics[model_file] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    # Save metrics
    os.makedirs(metrics_path, exist_ok=True)
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
    metrics_df.to_csv(os.path.join(metrics_path, 'model_metrics.csv'))

    # Select and save the best model
    best_model_file = metrics_df['f1_score'].idxmax()
    best_model = joblib.load(os.path.join(models_path, best_model_file))
    joblib.dump(best_model, os.path.join(models_path, 'bestmodel.pkl'))

if __name__ == "__main__":
    models_dir = 'artifacts/models/'
    metrics_dir = 'artifacts/metrics/'
    evaluate_models(models_dir, metrics_dir)
