import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load your data
# Replace this with your actual data loading
# df = pd.read_csv('your_data.csv')
# For demonstration, creating sample data structure
# You should replace this with your actual data
"""
Expected data format:
df = pd.DataFrame({
    'anchor_diameter_mm': [...],
    'embedment_depth_mm': [...],
    'concrete_strength_mpa': [...],
    'bond_strength_mpa': [...],
    'alpha': [...]  # your target variable (confined/unconfined load ratio)
})
"""


class AlphaPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.best_params = None
        
    def create_engineered_features(self, df):
        """
        Create engineered features based on hypothesis that features
        when multiplied/scaled should predict alpha
        """
        df_eng = df.copy()
        
        # Original features
        anchor_d = df['anchor_diameter_mm']
        embed_d = df['embedment_depth_mm']
        concrete_s = df['concrete_strength_mpa']
        bond_s = df['bond_strength_mpa']
        
        # Multiplicative combinations (based on your hypothesis)
        df_eng['anchor_embed_product'] = anchor_d * embed_d
        df_eng['concrete_bond_product'] = concrete_s * bond_s
        df_eng['all_features_product'] = anchor_d * embed_d * concrete_s * bond_s
        
        # Ratios (often important in structural engineering)
        df_eng['embed_anchor_ratio'] = embed_d / (anchor_d + 1e-8)  # avoid division by zero
        df_eng['bond_concrete_ratio'] = bond_s / (concrete_s + 1e-8)
        
        # Power combinations
        df_eng['anchor_squared'] = anchor_d ** 2
        df_eng['embed_sqrt'] = np.sqrt(embed_d)
        df_eng['concrete_log'] = np.log(concrete_s + 1)
        
        # Volume-like features
        df_eng['anchor_area'] = np.pi * (anchor_d/2) ** 2
        df_eng['embed_volume_proxy'] = df_eng['anchor_area'] * embed_d
        
        # Strength-weighted features
        df_eng['weighted_embed'] = embed_d * concrete_s
        df_eng['weighted_anchor'] = anchor_d * bond_s
        
        return df_eng
    
    def remove_outliers(self, df, target_col='alpha', method='iqr', factor=1.5):
        """
        Remove outliers from the dataset
        """
        if method == 'iqr':
            Q1 = df[target_col].quantile(0.25)
            Q3 = df[target_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            mask = (df[target_col] >= lower_bound) & (df[target_col] <= upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(df[target_col]))
            mask = z_scores < factor
            
        print(f"Removed {len(df) - mask.sum()} outliers out of {len(df)} points")
        return df[mask].copy()
    
    def prepare_data(self, df, target_col='alpha', test_size=0.2, remove_outliers=True):
        """
        Prepare data for training
        """
        # Remove outliers if requested
        if remove_outliers:
            df = self.remove_outliers(df, target_col)
        
        # Create engineered features
        df_eng = self.create_engineered_features(df)
        
        # Separate features and target
        feature_cols = [col for col in df_eng.columns if col != target_col]
        X = df_eng[feature_cols]
        y = df_eng[target_col]
        
        # Handle any infinite or NaN values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        # Store feature names
        self.feature_names = feature_cols
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=None
        )
        
        # Scale features (important for noisy data)
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test
    
    def optimize_hyperparameters(self, X_train, y_train):
        """
        Optimize XGBoost hyperparameters using GridSearchCV
        """
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5, 1.0],  # L1 regularization
            'reg_lambda': [0.1, 0.5, 1.0, 2.0]  # L2 regularization
        }
        
        # Create XGBoost regressor
        xgb_reg = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            xgb_reg,
            param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        print("Optimizing hyperparameters...")
        grid_search.fit(X_train, y_train)
        
        self.best_params = grid_search.best_params_
        print(f"Best parameters: {self.best_params}")
        print(f"Best CV score: {-grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def train_model(self, X_train, y_train, optimize_params=True):
        """
        Train the XGBoost model
        """
        if optimize_params:
            self.model = self.optimize_hyperparameters(X_train, y_train)
        else:
            # Use default parameters optimized for noisy data
            self.model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                objective='reg:squarederror'
            )
            self.model.fit(X_train, y_train)
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate model performance
        """
        y_pred = self.model.predict(X_test)
        
        cov_div = y_pred / y_test
        coe_var = np.std(cov_div) / np.mean(cov_div)
        print(f"Coefficient of Variability: {coe_var}")
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print("\nModel Performance:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': y_pred
        }
    
    def plot_results(self, y_test, y_pred):
        """
        Plot actual vs predicted values and residuals
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Actual vs Predicted
        axes[0].scatter(y_test, y_pred, alpha=0.7)
        axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0].set_xlabel('Actual Alpha')
        axes[0].set_ylabel('Predicted Alpha')
        axes[0].set_title('Actual vs Predicted Alpha Values')
        
        # Residuals plot
        residuals = y_test - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.7)
        axes[1].axhline(y=0, color='r', linestyle='--')
        axes[1].set_xlabel('Predicted Alpha')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residuals Plot')
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self):
        """
        Plot feature importance
        """
        if self.model is None:
            print("Model not trained yet!")
            return
            
        importance = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=True)
        
        plt.figure(figsize=(10, 8))
        plt.barh(feature_importance['feature'], feature_importance['importance'])
        plt.xlabel('Feature Importance')
        plt.title('XGBoost Feature Importance')
        plt.tight_layout()
        plt.show()
        
        return feature_importance
    
    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation
        """
        if self.model is None:
            print("Model not trained yet!")
            return
            
        cv_scores = cross_val_score(self.model, X, y, cv=cv,
                                    scoring='neg_mean_squared_error')
        rmse_scores = np.sqrt(-cv_scores)
        
        print(f"\nCross-validation results ({cv}-fold):")
        print(f"RMSE: {rmse_scores.mean():.4f} (+/- {rmse_scores.std() * 2:.4f})")
        
        return rmse_scores


# Example usage:
def main():
    """
    Main function to demonstrate usage
    Replace the sample data creation with your actual data loading
    """
    
    # REPLACE THIS SECTION WITH YOUR DATA LOADING
    # Example: df = pd.read_csv('your_data.csv')

    df = pd.read_excel(
        "./Data_for_chemofast.xlsx",
        sheet_name="Database"
    )

    df = df[[
        "Diameter",
        "Embedment depth",
        "fc",
        "Tau",
        "alpha"
    ]]
    df.columns = [
        "anchor_diameter_mm",
        "embedment_depth_mm",
        "concrete_strength_mpa",
        "bond_strength_mpa",
        "alpha"
    ]
    df = df[1:].astype(float)
    # Creating sample data for demonstration
    np.random.seed(42)
    
    # Initialize predictor
    predictor = AlphaPredictor()
    
    # Prepare data
    X_train, X_test, y_train, y_test, X_train_orig, X_test_orig = predictor.prepare_data(df)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Number of features: {X_train.shape[1]}")
    
    # Train model (set optimize_params=False for faster training during testing)
    predictor.train_model(X_train, y_train, optimize_params=False)
    
    # Evaluate model
    results = predictor.evaluate_model(X_test, y_test)
    
    # Plot results
    predictor.plot_results(y_test, results['predictions'])
    
    # Show feature importance
    feature_importance = predictor.plot_feature_importance()
    print("\nTop 10 Most Important Features:")
    print(feature_importance.tail(10))
    
    # Cross-validation
    predictor.cross_validate(X_train, y_train)
    
    return predictor, results


# Run the analysis
if __name__ == "__main__":
    predictor, results = main()
