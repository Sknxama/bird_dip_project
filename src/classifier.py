"""Lightweight Random Forest classifier for bird species prediction."""

import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from config import MODELS_DIR, SEED


def train_classifier(X, y, n_estimators=100):
    """
    Train a Random Forest classifier.
    
    Args:
        X: DataFrame of numeric features
        y: Series of class_id labels
        n_estimators: number of trees (default 100)
    
    Returns:
        model: trained RandomForestClassifier
        accuracy: float
    """
    # Split data: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    
    # Train Random Forest (no need for scaling, trees handle raw values)
    print(f"Training Random Forest with {n_estimators} trees...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=SEED,
        n_jobs=-1,  # Use all CPU cores
        max_depth=20,  # Prevent overfitting
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Training complete.")
    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Accuracy: {accuracy:.4f}")
    
    # Save model
    joblib.dump(model, os.path.join(MODELS_DIR, "rf_model.pkl"))
    print(f"  Model saved to: {MODELS_DIR}")
    
    return model, accuracy


def predict_species(features_dict, model, class_names=None):
    """
    Predict species from a single image's features.
    
    Args:
        features_dict: dict with feature values
        model: trained Random Forest model
        class_names: optional dict {class_id: class_name}
    
    Returns:
        predicted_class_id: int
        predicted_class_name: str
        confidence: float (probability of top class)
    """
    feature_cols = [
        "area", "perimeter", "circularity",
        "mean_h", "mean_s", "mean_v",
        "symmetry", "glcm_contrast", "glcm_energy", "fractal_dim"
    ]
    
    X = np.array([[features_dict[col] for col in feature_cols]])
    
    pred_id = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    confidence = float(np.max(proba))
    
    if class_names and pred_id in class_names:
        pred_name = class_names[pred_id]
    else:
        pred_name = str(pred_id)
    
    return int(pred_id), pred_name, confidence