import albumentations as A
import timm
import torch.nn as nn
from kornia.losses import DiceLoss, FocalLoss
from torch.nn import CrossEntropyLoss

from ml4cv_assignment.config.registry import Registry
from ml4cv_assignment.models import FaPNDecoder, Mask2Former, ProtoSegNet
from ml4cv_assignment.training.callbacks import (
    Callback,
    EarlyStoppingCallback,
    ModelSavingCallback,
    ModelSummaryCallback,
    PixelEmbeddingsCallback,
    VisualizeSegmentationResultsCallback,
)
from ml4cv_assignment.training.losses import (
    ObjectosphereLoss,
    OWLoss,
    PrototypicalGlobalLocalTripletLoss,
    RejectedByAllLoss,
    WeightedLoss,
)
from ml4cv_assignment.training.metrics import MeanIoU, Metric, OoDAUPR

# ------------------------
# Registry losses
# ------------------------
loss_registry = Registry[nn.Module]()
loss_registry.register("WeightedLoss", WeightedLoss)
loss_registry.register("FocalLoss", FocalLoss)
loss_registry.register("DiceLoss", DiceLoss)
loss_registry.register("CrossEntropyLoss", CrossEntropyLoss)
loss_registry.register("OWLoss", OWLoss)
loss_registry.register(
    "PrototypicalGlobalLocalTripletLoss", PrototypicalGlobalLocalTripletLoss
)
loss_registry.register("ObjectosphereLoss", ObjectosphereLoss)
loss_registry.register("RejectedByAllLoss", RejectedByAllLoss)

# ------------------------
# Registry transformations
# ------------------------
transform_registry = Registry[A.BasicTransform]()
# --- Resize & geometric
transform_registry.register("PadIfNeeded", A.PadIfNeeded)
transform_registry.register("Resize", A.Resize)
transform_registry.register("SmallestMaxSize", A.SmallestMaxSize)
transform_registry.register("LongestMaxSize", A.LongestMaxSize)
transform_registry.register("RandomCrop", A.RandomCrop)
transform_registry.register("RandomResizedCrop", A.RandomResizedCrop)
transform_registry.register("HorizontalFlip", A.HorizontalFlip)
# --- Color & brightness ---
transform_registry.register("ColorJitter", A.ColorJitter)
transform_registry.register("RGBShift", A.RGBShift)
transform_registry.register("RandomBrightnessContrast", A.RandomBrightnessContrast)
transform_registry.register("HueSaturationValue", A.HueSaturationValue)
transform_registry.register("RandomGamma", A.RandomGamma)
# --- Occlusion ---
transform_registry.register("CoarseDropout", A.CoarseDropout)
transform_registry.register("GridDropout", A.GridDropout)
transform_registry.register("Erasing", A.Erasing)
# --- Environmental artefacts ---
transform_registry.register("RandomSunFlare", A.RandomSunFlare)
transform_registry.register("RandomShadow", A.RandomShadow)
transform_registry.register("RandomRain", A.RandomRain)
transform_registry.register("Normalize", A.Normalize)
transform_registry.register("ToTensorV2", A.ToTensorV2)
# --- Compositions ---
transform_registry.register("OneOf", A.OneOf)
transform_registry.register("Compose", A.Compose)

# ------------------------
# Registry metrics
# ------------------------
metric_registry = Registry[Metric]()
metric_registry.register("MeanIoU", MeanIoU)
metric_registry.register("AUPR", OoDAUPR)

# ------------------------
# Registry callbacks
# ------------------------
callback_registry = Registry[Callback]()
callback_registry.register("EarlyStoppingCallback", EarlyStoppingCallback)
callback_registry.register("ModelSavingCallback", ModelSavingCallback)
callback_registry.register(
    "VisualizeSegmentationResultsCallback", VisualizeSegmentationResultsCallback
)
callback_registry.register("ModelSummaryCallback", ModelSummaryCallback)
callback_registry.register("PixelEmbeddingsCallback", PixelEmbeddingsCallback)

# ------------------------
# Registry models
# ------------------------
model_registry = Registry[nn.Module]()
model_registry.register("Mask2Former", Mask2Former)
model_registry.register("FaPNDecoder", FaPNDecoder)
model_registry.register("ProtoSegNet", ProtoSegNet)
model_registry.register("timm", timm.create_model)  # type: ignore
