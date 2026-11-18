import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler


def log_transform_approach(X, y, test_size=0.2, random_state=42):
    """
    Log-transform approach: log(y) = log(c) + log(x1) + log(x2) + ... + log(xn)
    This converts multiplicative to additive relationship
    """
    print("=== Log Transform Approach ===")
    
    # Handle zero/negative values by adding small constant
    X_log = np.log(np.abs(X) + 1e-10)
    y_log = np.log(np.abs(y) + 1e-10)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_log, y_log, test_size=test_size, random_state=random_state
    )
    
    # Try different models on log-transformed data
    models = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1)
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_log = model.predict(X_test)
        
        # Transform back to original scale
        y_pred = np.exp(y_pred_log)
        y_test_original = np.exp(y_test)
        
        mse = mean_squared_error(y_test_original, y_pred)
        r2 = r2_score(y_test_original, y_pred)
        
        results[name] = {
            'model': model,
            'mse': mse,
            'r2': r2,
            'y_pred': y_pred,
            'y_test': y_test_original
        }
        
        print(f"{name} - MSE: {mse:.4f}, R²: {r2:.4f}")
    
    return results


def nonlinear_least_squares_approach(X, y):
    """
    Nonlinear least squares to directly estimate the constant and any nonlinear parameters
    Assumes y = c * x1^a1 * x2^a2 * ... * xn^an
    """
    from scipy.optimize import curve_fit
    
    def multiplicative_model(X_flat, c, *exponents):
        # Reshape flat input back to original shape
        X_reshaped = X_flat.reshape(-1, X.shape[1])
        result = c * np.ones(X_reshaped.shape[0])
        for i, exp in enumerate(exponents):
            result *= np.power(np.abs(X_reshaped[:, i]) + 1e-10, exp)
        return result
    
    # Flatten X for curve_fit
    X_flat = X.flatten()
    
    # Initial guess: c=1, all exponents=1
    initial_guess = [1.0] + [1.0] * X.shape[1]
    
    try:
        # Fit the model
        popt, pcov = curve_fit(
            multiplicative_model, X_flat, y, 
            p0=initial_guess, maxfev=5000
        )
        
        constant = popt[0]
        exponents = popt[1:]
        
        print("=== Nonlinear Least Squares Approach ===")
        print(f"Estimated constant: {constant:.4f}")
        print("Estimated exponents:")
        for i, exp in enumerate(exponents):
            print(f"  Feature {i}: {exp:.4f}")
        
        # Calculate predictions
        y_pred = multiplicative_model(X_flat, *popt)
        
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        print(f"MSE: {mse:.4f}, R²: {r2:.4f}")
        
        return {'constant': constant, 'exponents': exponents, 'y_pred': y_pred, 'mse': mse, 'r2': r2}
    
    except Exception as e:
        print(f"Nonlinear fitting failed: {e}")
        return None


def feature_engineering_xgboost(X, y, feature_names=None):
    """
    Engineer features for XGBoost to better capture multiplicative relationships
    """
    print("=== Feature Engineering for XGBoost ===")
    
    X_orig = X.copy()
    
    # Add logarithmic features
    X_log = np.log(np.abs(X) + 1e-10)
    
    # Add pairwise products
    n_features = X.shape[1]
    products = []
    product_names = []
    
    for i in range(n_features):
        for j in range(i+1, n_features):
            product = X[:, i] * X[:, j]
            products.append(product.reshape(-1, 1))
            if feature_names:
                product_names.append(f"{feature_names[i]} × {feature_names[j]}")
            else:
                product_names.append(f"Feature_{i} × Feature_{j}")
    
    # Add overall product
    overall_product = np.prod(X, axis=1).reshape(-1, 1)
    products.append(overall_product)
    product_names.append("Overall Product")
    
    # Combine all features
    if products:
        X_products = np.hstack(products)
        X_engineered = np.hstack([X_orig, X_log, X_products])
        
        if feature_names:
            log_names = [f"log({name})" for name in feature_names]
            all_names = feature_names + log_names + product_names
        else:
            log_names = [f"log_Feature_{i}" for i in range(n_features)]
            orig_names = [f"Feature_{i}" for i in range(n_features)]
            all_names = orig_names + log_names + product_names
    else:
        X_engineered = np.hstack([X_orig, X_log])
        if feature_names:
            log_names = [f"log({name})" for name in feature_names]
            all_names = feature_names + log_names
        else:
            orig_names = [f"Feature_{i}" for i in range(n_features)]
            log_names = [f"log_Feature_{i}" for i in range(n_features)]
            all_names = orig_names + log_names
    
    print(f"Original features: {n_features}")
    print(f"Engineered features: {X_engineered.shape[1]}")
    print(f"Added features: {', '.join(product_names) if products else 'log transforms only'}")
    
    return X_engineered, all_names


def multiplicative_aware_xgboost(X_train, y_train, X_test, y_test):
    """
    XGBoost configuration optimized for multiplicative relationships
    """
    print("=== Multiplicative-Aware XGBoost ===")
    
    # Feature engineering
    X_train_eng, feature_names = feature_engineering_xgboost(X_train)
    X_test_eng, _ = feature_engineering_xgboost(X_test)
    
    # XGBoost with configuration for multiplicative relationships
    dtrain = xgb.DMatrix(X_train_eng, label=y_train)
    dtest = xgb.DMatrix(X_test_eng, label=y_test)
    
    params = {
        'objective': 'reg:squarederror',
        'tree_method': 'hist',
        'learning_rate': 0.01,  # Very low learning rate
        'max_depth': 2,  # Shallow trees
        'min_child_weight': 5,  # Conservative splits
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 10,  # Strong L1 regularization
        'reg_lambda': 10,  # Strong L2 regularization
        'random_state': 42
    }
    
    evals_result = {}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=2000,  # Many rounds with low learning rate
        early_stopping_rounds=50,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        evals_result=evals_result,
        verbose_eval=100
    )
    
    y_pred = model.predict(dtest)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Best iteration: {model.best_iteration}")
    print(f"MSE: {mse:.4f}, R²: {r2:.4f}")
    
    # Feature importance
    importance = model.get_score(importance_type='weight')
    print("\nTop 10 Most Important Features:")
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    for feature, score in sorted_importance:
        feature_idx = int(feature[1:])  # Remove 'f' prefix
        if feature_idx < len(feature_names):
            print(f"  {feature_names[feature_idx]}: {score}")
    
    return model, y_pred, mse, r2


def compare_all_approaches(X, y, feature_names=None):
    """
    Compare all approaches for multiplicative relationships
    """
    print("=" * 60)
    print("COMPARING APPROACHES FOR MULTIPLICATIVE RELATIONSHIPS")
    print("=" * 60)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = {}
    
    # 1. Log transform approach
    log_results = log_transform_approach(X, y)
    results['log_transform'] = log_results
    
    # 2. Nonlinear least squares
    nls_result = nonlinear_least_squares_approach(X, y)
    if nls_result:
        results['nonlinear_ls'] = nls_result
    
    # 3. Feature-engineered XGBoost
    xgb_model, xgb_pred, xgb_mse, xgb_r2 = multiplicative_aware_xgboost(X_train, y_train, X_test, y_test)
    results['xgboost_engineered'] = {
        'model': xgb_model,
        'y_pred': xgb_pred,
        'y_test': y_test,
        'mse': xgb_mse,
        'r2': xgb_r2
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY OF RESULTS")
    print("=" * 60)
    
    if 'log_transform' in results:
        best_log = min(results['log_transform'].items(), key=lambda x: x[1]['mse'])
        print(f"Best Log Transform: {best_log[0]} - MSE: {best_log[1]['mse']:.4f}, R²: {best_log[1]['r2']:.4f}")
    
    if 'nonlinear_ls' in results:
        print(f"Nonlinear Least Squares - MSE: {results['nonlinear_ls']['mse']:.4f}, R²: {results['nonlinear_ls']['r2']:.4f}")
    
    print(f"Feature-Engineered XGBoost - MSE: {results['xgboost_engineered']['mse']:.4f}, R²: {results['xgboost_engineered']['r2']:.4f}")
    
    return results


def theoretical_multiplicative_model(X, constant, feature_names=None):
    """
    If you know the theoretical relationship, use this to validate
    Example: y = constant * diameter^a * depth^b * strength^c
    """
    print("=== Theoretical Model Validation ===")
    print("Assuming y = constant × feature1 × feature2 × ... × featureN")
    
    # Simple multiplicative model (all exponents = 1)
    y_theoretical = constant * np.prod(X, axis=1)
    
    if feature_names:
        formula = f"y = {constant:.4f}"
        for name in feature_names:
            formula += f" × {name}"
        print(f"Formula: {formula}")
    
    return y_theoretical


# Example usage and testing
if __name__ == "__main__":
    # Generate synthetic multiplicative data for testing
    np.random.seed(42)
    n_samples = 270
    
    # Simulate structural engineering features
    anchor_diameter = np.random.uniform(10, 25, n_samples)  # mm
    embedment_depth = np.random.uniform(50, 200, n_samples)  # mm
    concrete_strength = np.random.uniform(20, 50, n_samples)  # MPa
    
    X_synthetic = np.column_stack([anchor_diameter, embedment_depth, concrete_strength])
    
    # True multiplicative relationship with some noise
    true_constant = 0.001
    y_true = true_constant * anchor_diameter * embedment_depth * concrete_strength
    noise = np.random.normal(0, 0.1 * np.std(y_true), n_samples)
    y_synthetic = y_true + noise
    
    feature_names = ['Anchor Diameter', 'Embedment Depth', 'Concrete Strength']
    
    print("Testing with synthetic multiplicative data...")
    print(f"True relationship: y = {true_constant} × Diameter × Depth × Strength")
    print(f"Data shape: {X_synthetic.shape}")
    
    # Compare all approaches
    results = compare_all_approaches(X_synthetic, y_synthetic, feature_names)
    
    # Test theoretical model
    y_theoretical = theoretical_multiplicative_model(X_synthetic, true_constant, feature_names)
    theoretical_mse = mean_squared_error(y_synthetic, y_theoretical)
    theoretical_r2 = r2_score(y_synthetic, y_theoretical)
    print(f"\nTheoretical Model - MSE: {theoretical_mse:.4f}, R²: {theoretical_r2:.4f}")
