import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler #type:ignore
import joblib #type:ignore
import os

def preprocess_data(input_path, output_path):
    # Load data
    df = pd.read_csv(input_path)

    # Handle missing values
    df = df.dropna()

    # Label Encoding for categorical variables
    le = LabelEncoder()
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = le.fit_transform(df[column])

    # Feature Scaling
    scaler = StandardScaler()
    features = df.drop('Churn', axis=1)
    target = df['Churn']
    scaled_features = scaler.fit_transform(features)

    # Save the preprocessor artifacts
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(scaler, os.path.join(output_path, 'scaler.joblib'))
    joblib.dump(le, os.path.join(output_path, 'label_encoder.joblib'))

    # Save processed data
    processed_df = pd.DataFrame(scaled_features, columns=features.columns)
    processed_df['Churn'] = target.reset_index(drop=True)
    processed_df.to_csv(os.path.join(output_path, 'processed_data.csv'), index=False)

if __name__ == "__main__":
    input_csv = 'data/customer_churn_data.csv'
    output_dir = 'artifacts/preprocessors/'
    preprocess_data(input_csv, output_dir)
