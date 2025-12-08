from ml4cv_assignment.training.losses.dml_loss import DMLLoss
from ml4cv_assignment.training.losses.objectosphere_loss import ObjectosphereLoss
from ml4cv_assignment.training.losses.ood_bce_loss import OoDBCELoss
from ml4cv_assignment.training.losses.ow_loss import OWLoss
from ml4cv_assignment.training.losses.prototypical_global_local_triplet_loss import (
    PrototypicalGlobalLocalTripletLoss,
)
from ml4cv_assignment.training.losses.rejected_by_all_loss import RejectedByAllLoss
from ml4cv_assignment.training.losses.weighted_loss import WeightedLoss

__all__ = [
    "RejectedByAllLoss",
    "ObjectosphereLoss",
    "WeightedLoss",
    "OWLoss",
    "PrototypicalGlobalLocalTripletLoss",
    "DMLLoss",
    "OoDBCELoss",
]
