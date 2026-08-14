"""
Linear Identifiability Probes and Ground-Truth State Alignment Metrics.

Theoretical Foundation:
Klindt, LeCun, Balestriero (2026): "When Does LeJEPA Learn a World Model?"
Theorem 1 & 2: Under SIGReg, the learned representation h(s) recovers ground-truth
latent state variables s up to an orthogonal transformation Q in O(n): h(s) = Q s.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import scipy.linalg
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error


class LinearIdentifiabilityProbe:
    """
    Linear Probe measuring how well learned latent representations z recover
    the underlying ground-truth physical state s:
        s_hat = W * z + b

    Attributes:
        alpha (float): L2 regularization parameter for Ridge regression.
    """

    def __init__(self, alpha: float = 1e-3) -> None:
        self.alpha = alpha
        self.probe = Ridge(alpha=alpha, fit_intercept=True)
        self.is_fitted = False

    def fit(self, z_train: np.ndarray, s_train: np.ndarray) -> "LinearIdentifiabilityProbe":
        """
        Fits linear probe on training embeddings and ground-truth states.

        Args:
            z_train (np.ndarray): Learned latents of shape (N_train, d_z).
            s_train (np.ndarray): True physical states of shape (N_train, d_s).
        """
        self.probe.fit(z_train, s_train)
        self.is_fitted = True
        return self

    def evaluate(self, z_test: np.ndarray, s_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluates linear probe on held-out test data.

        Returns:
            Dict[str, float]:
                - r2_score: Mean coefficient of determination across state dimensions.
                - mse: Mean squared state reconstruction error.
                - r2_per_dim: R2 score for each physical state variable.
        """
        if not self.is_fitted:
            raise RuntimeError("LinearIdentifiabilityProbe must be fitted before evaluation.")

        s_pred = self.probe.predict(z_test)
        r2 = r2_score(s_test, s_pred, multioutput="uniform_average")
        mse = mean_squared_error(s_test, s_pred)
        r2_dims = r2_score(s_test, s_pred, multioutput="raw_values").tolist()

        return {
            "r2_score": float(r2),
            "mse": float(mse),
            "r2_per_dim": r2_dims,
        }


class OrthogonalProcrustesAlignment:
    """
    Computes optimal orthogonal transformation Q in O(n) aligning normalized
    latents Z to ground-truth states S:
        min_Q || Z Q - S ||_F^2  s.t.  Q^T Q = I

    Used to directly test the Linear Identifiability Theorem (h(s) = Q s).
    """

    @staticmethod
    def align(z: np.ndarray, s: np.ndarray) -> Tuple[float, np.ndarray, Dict[str, float]]:
        """
        Args:
            z (np.ndarray): Latents of shape (N, d_z).
            s (np.ndarray): True states of shape (N, d_s).

        Returns:
            Tuple[float, np.ndarray, Dict[str, float]]:
                - normalized_discrepancy: Relative Frobenius norm error in [0, 1].
                - Q: Optimal orthogonal rotation matrix (d_z, d_s).
                - metrics: Summary statistics.
        """
        # Center and normalize both representations
        z_c = z - z.mean(axis=0, keepdims=True)
        s_c = s - s.mean(axis=0, keepdims=True)

        z_norm = z_c / (np.linalg.norm(z_c, "fro") + 1e-8)
        s_norm = s_c / (np.linalg.norm(s_c, "fro") + 1e-8)

        # Truncate or pad dimensions to match
        d_z, d_s = z.shape[1], s.shape[1]
        if d_z > d_s:
            # PCA projection of z down to d_s
            u, _, _ = scipy.linalg.svd(z_norm, full_matrices=False)
            z_proj = u[:, :d_s]
        else:
            z_proj = z_norm

        # Solve Orthogonal Procrustes: Q = U V^T where U Sigma V^T = SVD(z_proj^T s_norm)
        R, scale = scipy.linalg.orthogonal_procrustes(z_proj, s_norm)
        aligned_z = z_proj @ R

        discrepancy = np.linalg.norm(aligned_z - s_norm, "fro")
        cosine_sim = np.sum(aligned_z * s_norm)

        metrics = {
            "procrustes_discrepancy": float(discrepancy),
            "cosine_similarity": float(cosine_sim),
            "identifiability_score": float(max(0.0, 1.0 - discrepancy)),
        }

        return float(discrepancy), R, metrics


class CanonicalCorrelationMetrics:
    """Computes Canonical Correlation Analysis (CCA) between latents and states."""

    @staticmethod
    def compute(z: np.ndarray, s: np.ndarray) -> Dict[str, float]:
        """
        Computes canonical correlation coefficients between z and s.

        Returns:
            Dict[str, float]: Mean canonical correlation and top canonical correlations.
        """
        z_c = z - z.mean(axis=0, keepdims=True)
        s_c = s - s.mean(axis=0, keepdims=True)

        # Covariance matrices with ridge stabilization
        N = z.shape[0]
        eps = 1e-5 * np.eye(z.shape[1])
        eps_s = 1e-5 * np.eye(s.shape[1])

        C_zz = (z_c.T @ z_c) / (N - 1) + eps
        C_ss = (s_c.T @ s_c) / (N - 1) + eps_s
        C_zs = (z_c.T @ s_c) / (N - 1)

        # Invert square roots
        inv_sqrt_zz = scipy.linalg.inv(scipy.linalg.sqrtm(C_zz).real)
        inv_sqrt_ss = scipy.linalg.inv(scipy.linalg.sqrtm(C_ss).real)

        # T = inv(C_zz^(1/2)) * C_zs * inv(C_ss^(1/2))
        T = inv_sqrt_zz @ C_zs @ inv_sqrt_ss

        # Singular values of T are the canonical correlations rho_i
        _, s_vals, _ = scipy.linalg.svd(T)
        canonical_corrs = np.clip(s_vals, 0.0, 1.0)

        return {
            "mean_cca": float(np.mean(canonical_corrs)),
            "top1_cca": float(canonical_corrs[0]) if len(canonical_corrs) > 0 else 0.0,
            "min_cca": float(canonical_corrs[-1]) if len(canonical_corrs) > 0 else 0.0,
        }
