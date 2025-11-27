from typing import Optional, Any
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone, is_classifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from .core import StructuralFeatureMap, Array
from .tilt import PosExpoReweighter

class PosExpoClassifier(BaseEstimator, ClassifierMixin):
    """
    Classifier that uses POS-Expo reweighting to correct for selection bias.
    """
    
    def __init__(
        self,
        base_estimator: Any,
        feature_map: StructuralFeatureMap,
        reweighter: Optional[PosExpoReweighter] = None,
        alpha_reg: float = 1e-3,
    ) -> None:
        self.base_estimator = base_estimator
        self.feature_map = feature_map
        self.reweighter = reweighter
        self.alpha_reg = alpha_reg
        
    def fit(self, X, y, X_ref=None, iw_star=None):
        """
        Fit the model.
        
        1. Fit feature_map.
        2. Fit reweighter (using X_ref or iw_star).
        3. Compute weights.
        4. Fit base_estimator with weights.
        """
        X, y = check_X_y(X, y)
        
        # Initialize reweighter if not provided
        if self.reweighter is None:
            self.reweighter_ = PosExpoReweighter(
                feature_map=self.feature_map,
                alpha_reg=self.alpha_reg
            )
        else:
            self.reweighter_ = self.reweighter

        # 1 & 2. Fit reweighter (which fits feature_map internally)
        self.reweighter_.fit(X, y, X_ref=X_ref, iw_star=iw_star)
        
        # 3. Compute weights
        weights = self.reweighter_.compute_weights(X, y)
        
        # 4. Fit base estimator
        self.estimator_ = clone(self.base_estimator)
        
        # Handle sample_weight support
        fit_params = {}
        # We assume base_estimator supports sample_weight. 
        # Ideally check via inspection but most sklearn models do.
        fit_params["sample_weight"] = weights
        
        self.estimator_.fit(X, y, **fit_params)
        
        self.classes_ = self.estimator_.classes_
        
        return self

    def predict(self, X):
        check_is_fitted(self, ["estimator_", "reweighter_"])
        X = check_array(X)
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self, ["estimator_", "reweighter_"])
        X = check_array(X)
        return self.estimator_.predict_proba(X)


class PosExpoRegressor(BaseEstimator, RegressorMixin):
    """
    Regressor that uses POS-Expo reweighting.
    """
    
    def __init__(
        self,
        base_estimator: Any,
        feature_map: StructuralFeatureMap,
        reweighter: Optional[PosExpoReweighter] = None,
        alpha_reg: float = 1e-3,
    ) -> None:
        self.base_estimator = base_estimator
        self.feature_map = feature_map
        self.reweighter = reweighter
        self.alpha_reg = alpha_reg

    def fit(self, X, y, X_ref=None, iw_star=None):
        X, y = check_X_y(X, y)
        
        if self.reweighter is None:
            self.reweighter_ = PosExpoReweighter(
                feature_map=self.feature_map,
                alpha_reg=self.alpha_reg
            )
        else:
            self.reweighter_ = self.reweighter

        self.reweighter_.fit(X, y, X_ref=X_ref, iw_star=iw_star)
        weights = self.reweighter_.compute_weights(X, y)
        
        self.estimator_ = clone(self.base_estimator)
        self.estimator_.fit(X, y, sample_weight=weights)
        
        return self

    def predict(self, X):
        check_is_fitted(self, ["estimator_", "reweighter_"])
        X = check_array(X)
        return self.estimator_.predict(X)

