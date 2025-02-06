import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os


class HousePricePipeline:
    def __init__(self, file_path, target_column="Price"):
        """Initialize with dataset file path and target column."""
        self.file_path = file_path
        self.target_column = target_column
        self.scaler = StandardScaler()

    def load_data(self):
        """Load dataset from CSV file."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Dataset file {self.file_path} not found.")

        df = pd.read_csv(self.file_path)
        return df

    def preprocess_data(self, df):
        """Preprocess data: handle missing values and scale numerical features."""
        # Fill missing values with median
        df.fillna(df.median(), inplace=True)

        # Separate features and target
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        # Identify numerical features
        numerical_features = X.select_dtypes(include=['float64', 'int64']).columns

        # Scale numerical features
        X[numerical_features] = self.scaler.fit_transform(X[numerical_features])

        return X, y

    def save_preprocessor(self, file_name="../models/scaler.pkl"):
        """Save the fitted scaler for later use."""
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        joblib.dump(self.scaler, file_name)
        print(f"Scaler saved as {file_name}")


# Example usage
if __name__ == "__main__":
    pipeline = HousePricePipeline(file_path="../data/house_prices.csv")

    # Load data
    df = pipeline.load_data()

    # Preprocess data
    X, y = pipeline.preprocess_data(df)

    # Save the scaler for future use
    pipeline.save_preprocessor()
