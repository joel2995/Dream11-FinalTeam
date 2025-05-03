import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from typing import Tuple, List, Dict

class AdvancedEnsemblePredictor:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.model_weights = {}

    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced features for better prediction."""
        # Basic statistics
        df['Batting_SR'] = df['Runs'] / df['Balls Faced'] * 100
        df['Bowling_SR'] = df['Balls Given'] / df['Wickets'] if 'Balls Given' in df.columns else 0
        df['Economy'] = df['Runs Given'] / df['Overs']
        
        # Interaction features
        df['Runs_per_boundary'] = df['Runs'] / (df['Fours'] + df['Sixes'] + 1)
        df['Wickets_per_over'] = df['Wickets'] / df['Overs']
        
        # Form indicators
        df['Recent_Performance'] = df['Total Points'].rolling(window=5, min_periods=1).mean()
        df['Performance_Trend'] = df['Total Points'].rolling(window=5, min_periods=1).std()
        
        return df

    def train_ensemble_model(self, fantasy_data_path: str) -> Tuple[Dict, float]:
        """Train an advanced ensemble model with feature engineering and model tuning."""
        df = pd.read_excel(fantasy_data_path)
        df = self.create_advanced_features(df)

        # Enhanced feature set
        features = [
            'Runs', 'Balls Faced', 'Fours', 'Sixes', 'Strike Rate',
            'Wickets', 'Overs', 'Maidens', 'Runs Given',
            'Catches', 'Stumpings', 'Run Outs (Direct)', 'Run Outs (Assist)',
            'Batting_SR', 'Bowling_SR', 'Economy',
            'Runs_per_boundary', 'Wickets_per_over',
            'Recent_Performance', 'Performance_Trend'
        ]

        X = df[features]
        y = df['Total Points']

        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Initialize models with optimized parameters
        models = {
            'xgb': XGBRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, random_state=42
            ),
            'cat': CatBoostRegressor(
                iterations=200, learning_rate=0.05, depth=6,
                subsample=0.8, random_state=42, verbose=0
            ),
            'gbr': GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=6,
                subsample=0.8, random_state=42
            ),
            'rf': RandomForestRegressor(
                n_estimators=200, max_depth=10, min_samples_split=5,
                random_state=42
            )
        }

        # Train models and calculate weights based on performance
        for name, model in models.items():
            model.fit(X_train, y_train)
            self.models[name] = model
            
            # Calculate cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            self.model_weights[name] = np.mean(cv_scores)

            # Store feature importance
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = dict(zip(features, model.feature_importances_))

        # Normalize weights
        total_weight = sum(self.model_weights.values())
        self.model_weights = {k: v/total_weight for k, v in self.model_weights.items()}

        # Calculate ensemble performance
        ensemble_pred = self.predict_with_ensemble(X_test)
        ensemble_score = r2_score(y_test, ensemble_pred)

        return self.models, ensemble_score

    def predict_with_ensemble(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the weighted ensemble."""
        predictions = np.zeros(X.shape[0])
        
        for name, model in self.models.items():
            pred = model.predict(X)
            predictions += pred * self.model_weights[name]
            
        return predictions

    def get_feature_importance_analysis(self) -> Dict:
        """Get aggregated feature importance across all models."""
        if not self.feature_importance:
            return {}
            
        # Average feature importance across models
        all_features = set()
        for importance in self.feature_importance.values():
            all_features.update(importance.keys())
            
        avg_importance = {}
        for feature in all_features:
            importances = []
            for model_imp in self.feature_importance.values():
                if feature in model_imp:
                    importances.append(model_imp[feature])
            avg_importance[feature] = np.mean(importances)
            
        return avg_importance


def predict_with_ensemble(models, df):
    xgb, cat, gbr = models

    features = ['Runs', 'Balls Faced', 'Fours', 'Sixes', 'Strike Rate',
                'Wickets', 'Overs', 'Maidens', 'Runs Given',
                'Catches', 'Stumpings', 'Run Outs (Direct)', 'Run Outs (Assist)']
    
    xgb_pred = xgb.predict(df[features])
    cat_pred = cat.predict(df[features])
    gbr_pred = gbr.predict(df[features])

    df['Model Score'] = (xgb_pred + cat_pred + gbr_pred) / 3
    return df

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None
try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None

class ModelPredictor:
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.feature_cols = None
        self.feature_importance = {}
        self.model_weights = {}

    def create_advanced_features(self, df):
        """Create advanced features for better prediction."""
        df = df.copy()
        
        # Create ratio features
        if 'Runs' in df.columns and 'Balls Faced' in df.columns and df['Balls Faced'].sum() > 0:
            df['Runs_per_Ball'] = df['Runs'] / df['Balls Faced'].replace(0, 1)
        
        if 'Wickets' in df.columns and 'Overs' in df.columns and df['Overs'].sum() > 0:
            df['Wickets_per_Over'] = df['Wickets'] / df['Overs'].replace(0, 1)
        
        # Create interaction features
        if 'Fours' in df.columns and 'Sixes' in df.columns:
            df['Boundary_Count'] = df['Fours'] + df['Sixes']
        
        # Create role-specific features
        if 'Player Role' in df.columns:
            # One-hot encode player roles
            role_dummies = pd.get_dummies(df['Player Role'], prefix='Role')
            df = pd.concat([df, role_dummies], axis=1)
            
            # Create role-specific interaction features
            if 'Runs' in df.columns and 'Role_BAT' in df.columns:
                df['BAT_Runs'] = df['Runs'] * df['Role_BAT']
            
            if 'Wickets' in df.columns and 'Role_BOWL' in df.columns:
                df['BOWL_Wickets'] = df['Wickets'] * df['Role_BOWL']
        
        return df

    def train(self, df, feature_cols, target_col):
        df = df.copy()
        
        # Create advanced features
        df = self.create_advanced_features(df)
        
        # Ensure all feature columns exist
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        
        # Add advanced features to feature columns
        advanced_features = [col for col in df.columns if col not in feature_cols and 
                            col not in [target_col, 'Player', 'Team', 'Match Number'] and 
                            col.startswith(('Runs_per', 'Wickets_per', 'Boundary', 'BAT_', 'BOWL_', 'Role_'))]
        
        all_features = feature_cols + advanced_features
        
        # Handle missing values
        df[target_col] = np.nan_to_num(df[target_col])
        df[all_features] = df[all_features].fillna(0)
        
        X = df[all_features]
        y = df[target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models with optimized hyperparameters
        self.models['rf'] = RandomForestRegressor(
            n_estimators=200, 
            max_depth=10, 
            min_samples_split=5, 
            random_state=42
        ).fit(X_train, y_train)
        
        self.models['lr'] = LinearRegression().fit(X_train_scaled, y_train)
        
        self.models['knn'] = KNeighborsRegressor(
            n_neighbors=5, 
            weights='distance'
        ).fit(X_train_scaled, y_train)
        
        if XGBRegressor:
            self.models['xgb'] = XGBRegressor(
                n_estimators=200, 
                learning_rate=0.05, 
                max_depth=6, 
                subsample=0.8, 
                colsample_bytree=0.8, 
                random_state=42, 
                verbosity=0
            ).fit(X_train, y_train)
        
        if CatBoostRegressor:
            self.models['cat'] = CatBoostRegressor(
                iterations=200, 
                learning_rate=0.05, 
                depth=6, 
                subsample=0.8, 
                random_state=42, 
                verbose=0
            ).fit(X_train, y_train)
        
        # Calculate model weights based on performance
        for name, model in self.models.items():
            if name == 'lr' or name == 'knn':
                pred = model.predict(X_test_scaled)
            else:
                pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, pred)
            # Convert MSE to weight (lower MSE = higher weight)
            self.model_weights[name] = 1 / (mse + 1e-10)  # Add small epsilon to avoid division by zero
            
            # Store feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = dict(zip(all_features, model.feature_importances_))
        
        # Normalize weights to sum to 1
        total_weight = sum(self.model_weights.values())
        self.model_weights = {k: v/total_weight for k, v in self.model_weights.items()}
        
        self.scaler = scaler
        self.feature_cols = all_features

    def predict(self, df):
        df = df.copy()
        
        # Create advanced features
        df = self.create_advanced_features(df)
        
        # Ensure all feature columns exist
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0
        
        # Handle missing values
        df[self.feature_cols] = df[self.feature_cols].fillna(0)
        
        X = df[self.feature_cols]
        X_scaled = self.scaler.transform(X)
        
        # Make weighted predictions
        weighted_preds = np.zeros(X.shape[0])
        
        for name, model in self.models.items():
            if name == 'lr' or name == 'knn':
                pred = model.predict(X_scaled)
            else:
                pred = model.predict(X)
            
            # Apply model weight
            weighted_preds += pred * self.model_weights[name]
        
        df['selection_score'] = weighted_preds
        return df
        return df
