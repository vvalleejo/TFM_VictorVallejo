"""Loss functions and anti-collapse regularizers for JEPA World Models."""

from jepa_wm.losses.sigreg import SIGReg
from jepa_wm.losses.vicreg import VICRegLoss
from jepa_wm.losses.info_nce import InfoNCELoss
from jepa_wm.losses.rollout_loss import MultiStepPredictiveLoss, LeWMLoss
from jepa_wm.losses.variational import VariationalJEPALoss

__all__ = [
    "SIGReg",
    "VICRegLoss",
    "InfoNCELoss",
    "MultiStepPredictiveLoss",
    "LeWMLoss",
    "VariationalJEPALoss",
]
