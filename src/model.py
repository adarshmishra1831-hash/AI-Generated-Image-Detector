import torch
import torch.nn as nn
import timm


class AIImageDetector(nn.Module):
    """
    EfficientNet-B3 backbone with custom classifier for
    Real vs AI-generated image classification.
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.4, pretrained: bool = True):
        super().__init__()

        # Backbone (feature extractor)
        self.backbone = timm.create_model(
            "efficientnet_b3",
            pretrained=pretrained,
            num_classes=0,       # remove default classifier
            global_pool="avg"    # global average pooling
        )

        feature_dim = self.backbone.num_features  # 1536

        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),

            nn.Linear(128, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output

    def get_features(self, x):
        return self.backbone.forward_features(x)