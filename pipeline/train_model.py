import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class HousePriceModel:
    def __init__(self, data_path="../data/house_prices.csv", model_path="../models/house_price_model.pkl"):
        self.data_path = data_path
        self.model_path = model_path

    def load_data(self):
        """Load the preprocessed dataset."""
        df = pd.read_csv(self.data_path)

        # Drop rows with missing values (if any remain after preprocessing)
        df.dropna(inplace=True)

        # Separate features and target
        X = df.drop(columns=["Price"])
        y = df["Price"]

        return X, y

    def train_model(self, X, y):
        """Train a regression model on house price data."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize the model
        model = RandomForestRegressor(n_estimators=100, random_state=42)

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, y_pred)

        print(f"Model Performance:\nMAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")

        return model

    def save_model(self, model):
        """Save the trained model to a file."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(model, self.model_path)
        print(f"Model saved as {self.model_path}")


if __name__ == "__main__":
    house_model = HousePriceModel()

    # Load preprocessed data
    X, y = house_model.load_data()

    # Train model
    trained_model = house_model.train_model(X, y)

    # Save model
    house_model.save_model(trained_model)
