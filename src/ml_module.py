"""
Machine Learning Module.

Random Forest and Neural Network (MLP) regressors to predict
duration deviation percentages per activity. The ML output serves
as a bias factor that adjusts Monte Carlo sampling.
"""

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class DurationPredictor:
    """
    ML model that predicts duration deviation (bias factor) per activity.

    The bias factor is a multiplier applied to Monte Carlo samples:
    adjusted_duration = sampled_duration * bias_factor

    A bias_factor > 1 means the activity is expected to take longer
    than the three-point estimate suggests; < 1 means shorter.
    """

    def __init__(
        self,
        model_type: str = "random_forest",
        n_estimators: int = 100,
        random_state: int = 42,
    ):
        """
        Parameters
        ----------
        model_type : str
            'random_forest' or 'neural_network'.
        n_estimators : int
            Number of trees for Random Forest.
        random_state : int
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.is_fitted = False

        if model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=10,
                min_samples_split=3,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1,
            )
        elif model_type == "neural_network":
            self.model = MLPRegressor(
                hidden_layer_sizes=(64, 32, 16),
                activation="relu",
                solver="adam",
                max_iter=500,
                random_state=random_state,
                early_stopping=True,
                validation_fraction=0.15,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def prepare_features(
        self,
        activities_df: pd.DataFrame,
        is_training: bool = True,
    ) -> np.ndarray:
        """
        Prepare feature matrix from activity data.

        Expected feature columns (any available subset):
        - activity_type / category (categorical -> encoded)
        - complexity_score (1-5)
        - estimated_duration (m value)
        - range_ratio ((b-a)/m - normalized uncertainty)
        - vendor_lead_time (days)
        - team_experience (1-5)
        - weather_index (0-1)
        - regulatory_risk (0-1)
        - project_capacity (m³/day)
        - n_predecessors (count)

        Parameters
        ----------
        activities_df : pd.DataFrame
        is_training : bool
            If True, fit encoders/scaler. If False, transform only.

        Returns
        -------
        np.ndarray
            Feature matrix.
        """
        df = activities_df.copy()

        # Generate derived features
        if "m" in df.columns and "a" in df.columns and "b" in df.columns:
            df["range_ratio"] = (df["b"] - df["a"]) / df["m"].clip(lower=0.1)
            df["estimated_duration"] = df["m"]

        if "predecessors" in df.columns:
            df["n_predecessors"] = df["predecessors"].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )

        # Define all possible feature columns
        numeric_features = [
            "estimated_duration", "range_ratio", "complexity_score",
            "vendor_lead_time", "team_experience", "weather_index",
            "regulatory_risk", "project_capacity", "n_predecessors",
        ]
        categorical_features = ["category"]

        # During prediction, ensure all trained features exist (fill missing with defaults)
        if not is_training and self.feature_names is not None:
            if "complexity_score" not in df.columns:
                if all(c in df.columns for c in ["a", "m", "b"]):
                    df["complexity_score"] = np.clip(
                        ((df["b"] - df["a"]) / df["m"].clip(lower=0.1) * 2.5), 1, 5
                    ).round(0)
                else:
                    df["complexity_score"] = 3
            equipment_cats = {"electromechanical", "piping", "instrumentation", "electrical"}
            if "vendor_lead_time" not in df.columns and "category" in df.columns:
                df["vendor_lead_time"] = df["category"].apply(
                    lambda c: 30 if c in equipment_cats else 0
                )
            if "team_experience" not in df.columns:
                df["team_experience"] = 3
            outdoor_cats = {"site_preparation", "civil_works", "piping"}
            if "weather_index" not in df.columns and "category" in df.columns:
                df["weather_index"] = df["category"].apply(
                    lambda c: 0.4 if c in outdoor_cats else 0.1
                )
            reg_cats = {"permitting", "design", "commissioning"}
            if "regulatory_risk" not in df.columns and "category" in df.columns:
                df["regulatory_risk"] = df["category"].apply(
                    lambda c: 0.5 if c in reg_cats else 0.1
                )
            if "project_capacity" not in df.columns:
                df["project_capacity"] = 0

        # Select available features
        available_numeric = [f for f in numeric_features if f in df.columns]
        available_categorical = [f for f in categorical_features if f in df.columns]

        # Encode categorical features
        for col in available_categorical:
            if is_training:
                le = LabelEncoder()
                df[col + "_encoded"] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le:
                    df[col + "_encoded"] = df[col].astype(str).map(
                        lambda x, le=le: (
                            le.transform([x])[0] if x in le.classes_
                            else -1
                        )
                    )
                else:
                    df[col + "_encoded"] = 0

            available_numeric.append(col + "_encoded")

        if is_training:
            self.feature_names = available_numeric
        else:
            # Use exact same features as training, in same order
            # Fill any missing with 0
            for feat in self.feature_names:
                if feat not in df.columns:
                    df[feat] = 0
            available_numeric = self.feature_names

        X = df[available_numeric].fillna(0).values

        if is_training:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)

        return X

    def train(
        self,
        training_df: pd.DataFrame,
        target_column: str = "actual_duration",
        test_size: float = 0.2,
    ) -> dict:
        """
        Train the model on historical data.

        The target is the deviation ratio: actual_duration / estimated_duration.
        This becomes the bias factor for Monte Carlo adjustment.

        Parameters
        ----------
        training_df : pd.DataFrame
            Historical data with actual durations.
        target_column : str
            Column containing actual observed durations.
        test_size : float
            Fraction for test split.

        Returns
        -------
        dict with training metrics.
        """
        if target_column not in training_df.columns:
            raise ValueError(f"Target column '{target_column}' not found.")

        df = training_df.dropna(subset=[target_column]).copy()

        if len(df) < 5:
            warnings.warn(
                f"Very small training set ({len(df)} samples). "
                "Model may not generalize well."
            )

        X = self.prepare_features(df, is_training=True)

        # Target: deviation ratio
        y = df[target_column].values / df["m"].clip(lower=0.1).values

        if len(df) >= 10:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state
            )
        else:
            X_train, X_test, y_train, y_test = X, X, y, y

        self.model.fit(X_train, y_train)
        self.is_fitted = True

        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)

        metrics = {
            "n_samples": len(df),
            "n_features": X.shape[1],
            "feature_names": self.feature_names,
            "train_mae": mean_absolute_error(y_train, y_pred_train),
            "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
            "train_r2": r2_score(y_train, y_pred_train),
            "test_mae": mean_absolute_error(y_test, y_pred_test),
            "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
            "test_r2": r2_score(y_test, y_pred_test),
        }

        # Feature importance (Random Forest)
        if self.model_type == "random_forest" and self.feature_names:
            importances = self.model.feature_importances_
            metrics["feature_importance"] = dict(
                zip(self.feature_names, importances.tolist())
            )

        return metrics

    def cross_validate(
        self,
        training_df: pd.DataFrame,
        target_column: str = "actual_duration",
        cv_folds: int = 5,
    ) -> dict:
        """
        Perform k-fold cross-validation.

        Returns
        -------
        dict with CV scores.
        """
        df = training_df.dropna(subset=[target_column]).copy()

        if len(df) < cv_folds * 2:
            warnings.warn(f"Too few samples ({len(df)}) for {cv_folds}-fold CV.")
            cv_folds = max(2, len(df) // 2)

        X = self.prepare_features(df, is_training=True)
        y = df[target_column].values / df["m"].clip(lower=0.1).values

        scores = cross_val_score(
            self.model, X, y,
            cv=cv_folds, scoring="neg_mean_absolute_error",
        )

        return {
            "cv_folds": cv_folds,
            "cv_mae_mean": -float(np.mean(scores)),
            "cv_mae_std": float(np.std(scores)),
            "cv_scores": (-scores).tolist(),
        }

    def predict_bias_factors(
        self, activities_df: pd.DataFrame
    ) -> dict[int, float]:
        """
        Predict bias factors for activities.

        Parameters
        ----------
        activities_df : pd.DataFrame
            Activities to predict bias factors for.

        Returns
        -------
        dict[int, float]
            Mapping of activity_id -> bias_factor.
        """
        if not self.is_fitted:
            warnings.warn("Model not fitted. Returning neutral bias factors (1.0).")
            return {row["id"]: 1.0 for _, row in activities_df.iterrows()}

        X = self.prepare_features(activities_df, is_training=False)
        predictions = self.model.predict(X)

        # Clip extreme predictions
        predictions = np.clip(predictions, 0.5, 2.5)

        return {
            row["id"]: float(predictions[i])
            for i, (_, row) in enumerate(activities_df.iterrows())
        }

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance ranking (Random Forest only)."""
        if self.model_type != "random_forest" or not self.is_fitted:
            return None

        importances = self.model.feature_importances_
        df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False)
        return df


def build_training_data(
    normalized_projects_df: pd.DataFrame,
    actual_durations: Optional[dict[int, float]] = None,
) -> pd.DataFrame:
    """
    Build a training DataFrame from normalized historical projects.

    If actual_durations is provided, it maps activity_id -> actual duration.
    Otherwise, looks for an 'actual_duration' column in the DataFrame.

    Parameters
    ----------
    normalized_projects_df : pd.DataFrame
        Combined, normalized data from multiple projects.
    actual_durations : dict, optional
        {activity_id: actual_duration_days}

    Returns
    -------
    pd.DataFrame
        Training data with required features and target column.
    """
    df = normalized_projects_df.copy()

    if actual_durations:
        df["actual_duration"] = df["id"].map(actual_durations)

    # Add derived features if not present
    if "complexity_score" not in df.columns:
        # Estimate complexity from range ratio
        df["complexity_score"] = np.clip(
            ((df["b"] - df["a"]) / df["m"].clip(lower=0.1) * 2.5), 1, 5
        ).round(0)

    if "vendor_lead_time" not in df.columns:
        # Set 0 for non-equipment activities, estimate for others
        equipment_cats = {"electromechanical", "piping", "instrumentation", "electrical"}
        df["vendor_lead_time"] = df["category"].apply(
            lambda c: 30 if c in equipment_cats else 0
        )

    if "team_experience" not in df.columns:
        df["team_experience"] = 3  # neutral default

    if "weather_index" not in df.columns:
        outdoor_cats = {"site_preparation", "civil_works", "piping"}
        df["weather_index"] = df["category"].apply(
            lambda c: 0.4 if c in outdoor_cats else 0.1
        )

    if "regulatory_risk" not in df.columns:
        reg_cats = {"permitting", "design", "commissioning"}
        df["regulatory_risk"] = df["category"].apply(
            lambda c: 0.5 if c in reg_cats else 0.1
        )

    return df


def create_synthetic_training_data(
    activities_df: pd.DataFrame,
    n_synthetic_projects: int = 20,
    noise_std: float = 0.15,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic training data by adding noise to activity estimates.

    Useful when only a few real projects are available. Each synthetic
    project samples actual durations from a perturbed distribution
    around the most-likely estimate.

    Parameters
    ----------
    activities_df : pd.DataFrame
    n_synthetic_projects : int
    noise_std : float
        Standard deviation of the log-normal noise factor.
    random_state : int

    Returns
    -------
    pd.DataFrame
        Synthetic training data with 'actual_duration' column.
    """
    rng = np.random.default_rng(random_state)
    all_rows = []

    for proj_idx in range(n_synthetic_projects):
        df = activities_df.copy()
        df["project_id"] = f"synthetic_{proj_idx}"

        # Simulate actual durations with log-normal noise
        noise = rng.lognormal(mean=0, sigma=noise_std, size=len(df))
        df["actual_duration"] = (df["m"] * noise).round(1)

        # Ensure actual is within reasonable bounds
        df["actual_duration"] = df["actual_duration"].clip(
            lower=df["a"] * 0.8,
            upper=df["b"] * 1.3,
        )

        # Vary some features
        df["team_experience"] = rng.integers(2, 5, size=len(df))
        df["weather_index"] = rng.uniform(0.05, 0.6, size=len(df)).round(2)

        all_rows.append(df)

    return pd.concat(all_rows, ignore_index=True)
