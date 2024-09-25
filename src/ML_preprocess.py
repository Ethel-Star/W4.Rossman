import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

class RossmannModel:
    def __init__(self):
        self.pipeline = None

    def preprocess_data(self, train_df, test_df, store_df):
        """Preprocesses the input data by merging, feature extraction, and scaling."""

        # Merge store data into train and test sets
        train_df = pd.merge(train_df, store_df, how='left', on='Store')
        test_df = pd.merge(test_df, store_df, how='left', on='Store')

        # Convert 'Date' column to datetime
        train_df['Date'] = pd.to_datetime(train_df['Date'])
        test_df['Date'] = pd.to_datetime(test_df['Date'])

        # Extract additional features from 'Date'
        def extract_date_features(df):
            df['Year'] = df['Date'].dt.year
            df['Month'] = df['Date'].dt.month
            df['Day'] = df['Date'].dt.day
            df['WeekOfYear'] = df['Date'].dt.isocalendar().week
            df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 6 else 0)
        extract_date_features(train_df)
        extract_date_features(test_df)

        # Convert categorical variables to numeric using one-hot encoding
        train_df = pd.get_dummies(train_df, columns=['StoreType', 'Assortment', 'StateHoliday', 'PromoInterval'], drop_first=True)
        test_df = pd.get_dummies(test_df, columns=['StoreType', 'Assortment', 'StateHoliday', 'PromoInterval'], drop_first=True)

        # Columns to be scaled
        columns_to_scale = ['Customers', 'CompetitionDistance', 'Promo2SinceWeek', 'Promo2SinceYear']

        # Check if all columns exist before scaling
        missing_cols = [col for col in columns_to_scale if col not in train_df.columns]
        if missing_cols:
            print(f"Warning: The following columns are missing from train_df: {missing_cols}")
            columns_to_scale = [col for col in columns_to_scale if col in train_df.columns]

        missing_test_cols = [col for col in columns_to_scale if col not in test_df.columns]
        if missing_test_cols:
            print(f"Warning: The following columns are missing from test_df: {missing_test_cols}")
            columns_to_scale = [col for col in columns_to_scale if col in test_df.columns]

        # Scale numeric features if columns are present
        if columns_to_scale:
            scaler = StandardScaler()
            train_df[columns_to_scale] = scaler.fit_transform(train_df[columns_to_scale])
            test_df[columns_to_scale] = scaler.transform(test_df[columns_to_scale])

        return train_df, test_df

    def train(self, train_df):
        """Train the model using RandomForestRegressor."""
        
        # Split the data into features and target
        X = train_df.drop(columns=['Sales'])
        y = train_df['Sales']
        
        # Train-Test split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Define the pipeline with preprocessing and model
        self.pipeline = Pipeline([
            ('model', RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42))
        ])
        
        # Fit the pipeline
        self.pipeline.fit(X_train, y_train)
        
        # Predict on validation set
        y_pred = self.pipeline.predict(X_val)
        
        # Calculate RMSE for validation set
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        print(f'Validation RMSE: {rmse}')
        
        return X_train, X_val, y_train, y_val, y_pred

    def get_feature_importance(self, X_train):
        """Get and plot feature importance."""
        importances = self.pipeline.named_steps['model'].feature_importances_
        feature_names = X_train.columns
        sorted_indices = np.argsort(importances)[::-1]

        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importance")
        plt.bar(range(X_train.shape[1]), importances[sorted_indices], align="center")
        plt.xticks(range(X_train.shape[1]), feature_names[sorted_indices], rotation=90)
        plt.tight_layout()
        plt.show()

    def predict_with_confidence_intervals(self, X_val, n_samples=100, ci=95):
        """Estimate confidence intervals for predictions using bootstrap sampling."""
        predictions = []

        for _ in range(n_samples):
            sample_indices = np.random.choice(np.arange(X_val.shape[0]), size=X_val.shape[0], replace=True)
            X_sample = X_val.iloc[sample_indices]
            predictions.append(self.pipeline.predict(X_sample))

        predictions = np.array(predictions)
        
        lower_bound = np.percentile(predictions, (100-ci)/2.0, axis=0)
        upper_bound = np.percentile(predictions, 100 - (100-ci)/2.0, axis=0)
        
        return lower_bound, upper_bound
