import torch

from jaxtyping import Float
from torch import Tensor
from typing import Optional


class PCA:
    """Perform Principal Component Analysis (PCA).

    Args:
        n_components (int): Number of components to keep.

    Attributes:
        components (Float[Tensor, "n_features, n_features"]):
        Principal axes in feature space, representing the directions of
        maximum variance in the data.
        components_ (Float[Tensor, "n_components, n_features"]):
        `n_components` number of components
    """

    def __init__(self, n_components: Optional[int] = None):
        self.n_components = n_components

    def __call__(self, X: Float[Tensor, "n_samples n_features"]):
        """Perform PCA on the given data.

        Args:
            X (Float[Tensor, "n_samples n_features"]): The data on which PCA is performed.
        """
        return X @ self.components.T

    def fit(self, X: Float[Tensor, "n_samples n_features"]):
        """Fit PCA on the given data.

        Args:
            X (Float[Tensor, "n_samples n_features"]): The data on which PCA is to be fitted.
        """

        # 1. Find Covariance of X
        cov = X.T.cov()

        # 2. Find eigen values, and eigen vectors of X
        eigen_values, eigen_vectors = torch.linalg.eigh(cov)

        # 3. Sort the eigen vectors in decreasing order of explained variance
        desc_idxs: Float[Tensor, " n_features"] = eigen_values.argsort().flip((0,))
        desc_eigen_vectors = eigen_vectors.T[desc_idxs]

        # 4. Set self.components_
        self.components_ = desc_eigen_vectors

        # 5. Set self.components
        n_components = self.n_components if self.n_components else len(desc_eigen_vectors)
        self.components = desc_eigen_vectors[:n_components, :]
