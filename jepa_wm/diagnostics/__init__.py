"""Latent space diagnostics, linear identifiability metrics, and spectral probes."""

from jepa_wm.diagnostics.identifiability import (
    LinearIdentifiabilityProbe,
    OrthogonalProcrustesAlignment,
    CanonicalCorrelationMetrics,
)
from jepa_wm.diagnostics.spectral import (
    compute_effective_rank,
    compute_spectral_diagnostics,
)
from jepa_wm.diagnostics.energy import (
    compute_noise_rejection_ratio,
    compute_latent_energy_surface,
)

__all__ = [
    "LinearIdentifiabilityProbe",
    "OrthogonalProcrustesAlignment",
    "CanonicalCorrelationMetrics",
    "compute_effective_rank",
    "compute_spectral_diagnostics",
    "compute_noise_rejection_ratio",
    "compute_latent_energy_surface",
]
