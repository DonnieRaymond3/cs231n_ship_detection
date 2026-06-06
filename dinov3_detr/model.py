import torch
from torch import nn
from transformers import AutoBackbone, AutoConfig, DetrConfig, DetrForObjectDetection

class DetrBackboneAdapter(nn.Module):

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, pixel_values, pixel_mask):
        outputs = self.backbone(pixel_values)
        feature_maps = getattr(outputs, "feature_maps", None)
        if not feature_maps:
            raise ValueError(
                "Backbone did not return feature_maps. Use an AutoBackbone "
                "checkpoint that supports spatial feature extraction."
            )

        features = []
        position_embeddings = []
        for feature_map in feature_maps:

            if isinstance(feature_map, (tuple, list)):
                if len(feature_map) != 1:
                    raise ValueError(f"Unexpected nested feature map with {len(feature_map)} values.")
                feature_map = feature_map[0]

            mask = torch.nn.functional.interpolate(
                pixel_mask[:, None].float(),
                size=feature_map.shape[-2:],
                mode="nearest",
            ).to(torch.bool)[:, 0]
            features.append((feature_map, mask))

        return tuple(features)

def load_compatible_detr_weights(model, checkpoint_name: str):
    source = DetrForObjectDetection.from_pretrained(checkpoint_name)
    source_state = source.state_dict()
    target_state = model.state_dict()

    skip_prefixes = (
        "model.backbone.",
        "model.input_projection.",
        "class_labels_classifier.",
    )
    compatible = {}
    skipped_shape = []
    for key, value in source_state.items():
        if key.startswith(skip_prefixes):
            continue
        if key in target_state and target_state[key].shape == value.shape:
            compatible[key] = value
        elif key in target_state:
            skipped_shape.append(key)

    missing, unexpected = model.load_state_dict(compatible, strict=False)
    print(
        f"Initialized {len(compatible)} tensors from {checkpoint_name}; "
        f"left {len(missing)} target tensors at their DINO/task-specific init; "
        f"skipped {len(skipped_shape)} shape-mismatched tensors.",
        flush=True,
    )
    if unexpected:
        print(f"Unexpected pretrained keys while loading DETR init: {unexpected}", flush=True)
    return model

def build_model(
    backbone_name: str,
    num_labels: int = 1,
    freeze_backbone: bool = True,
    pretrained_detr_name: str = "facebook/detr-resnet-50",
    id2label=None,
    label2id=None,
):
    if id2label is None:
        id2label = {0: "ship"}
    if label2id is None:
        label2id = {"ship": 0}

    backbone_config = AutoConfig.from_pretrained(backbone_name)
    backbone = AutoBackbone.from_pretrained(backbone_name)

    config = DetrConfig(
        backbone_config=backbone_config,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        use_timm_backbone=False,
        use_pretrained_backbone=False,
    )
    model = DetrForObjectDetection(config)

    if pretrained_detr_name:
        model = load_compatible_detr_weights(model, pretrained_detr_name)

    model.model.backbone = DetrBackboneAdapter(backbone)

    if freeze_backbone:

        for param in model.model.backbone.backbone.parameters():
            param.requires_grad = False

    return model
