from typing import Optional, Protocol, List, Sequence
import numpy as np

Array = np.ndarray

class StructuralFeatureMap(Protocol):
    """
    Protocol for structural feature maps phi: (X, y, model) -> R^k.
    """

    def fit(
        self,
        X: Array,
        y: Optional[Array] = None,
        base_model: Optional[object] = None,
    ) -> "StructuralFeatureMap":
        """
        Fit the feature map to the data.
        """
        ...

    def transform(
        self,
        X: Array,
        y: Optional[Array] = None,
        base_model: Optional[object] = None,
    ) -> Array:
        """
        Transform data into structural features.
        Returns array of shape (n_samples, k).
        """
        ...

    def fit_transform(
        self,
        X: Array,
        y: Optional[Array] = None,
        base_model: Optional[object] = None,
    ) -> Array:
        """
        Fit to data, then transform it.
        """
        ...


class CompositeFeatureMap:
    """
    Combines multiple StructuralFeatureMaps into a single map by stacking outputs.
    phi(z) = [phi_1(z), ..., phi_m(z)].
    """

    def __init__(self, feature_maps: Sequence[StructuralFeatureMap]) -> None:
        self.feature_maps = feature_maps

    def fit(
        self,
        X: Array,
        y: Optional[Array] = None,
        base_model: Optional[object] = None,
    ) -> "CompositeFeatureMap":
        for fm in self.feature_maps:
            fm.fit(X, y, base_model)
        return self

    def transform(
        self,
        X: Array,
        y: Optional[Array] = None,
        base_model: Optional[object] = None,
    ) -> Array:
        outputs = []
        for fm in self.feature_maps:
            out = fm.transform(X, y, base_model)
            # Ensure output is 2D (n_samples, k_j)
            if out.ndim == 1:
                out = out.reshape(-1, 1)
            outputs.append(out)
        
        if not outputs:
            return np.empty((X.shape[0], 0))
            
        return np.hstack(outputs)

    def fit_transform(
        self,
        X: Array,
        y: Optional[Array] = None,
        base_model: Optional[object] = None,
    ) -> Array:
        outputs = []
        for fm in self.feature_maps:
            # Check if fit_transform is implemented, else fit then transform
            if hasattr(fm, "fit_transform"):
                out = fm.fit_transform(X, y, base_model)
            else:
                fm.fit(X, y, base_model)
                out = fm.transform(X, y, base_model)
            
            if out.ndim == 1:
                out = out.reshape(-1, 1)
            outputs.append(out)

        if not outputs:
            return np.empty((X.shape[0], 0))

        return np.hstack(outputs)

