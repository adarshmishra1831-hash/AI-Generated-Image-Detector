import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


IMG_SIZE = 224


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).
    Produces a heatmap highlighting image regions most influential
    to the model's prediction.

    Paper: [arxiv.org](https://arxiv.org/abs/1610.02391)
    """

    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor,
                 class_idx: int = None) -> np.ndarray:
        """
        Generates a Grad-CAM heatmap for the given input.

        Args:
            input_tensor : Preprocessed image tensor (1, C, H, W)
            class_idx    : Target class (None = predicted class)

        Returns:
            cam : Heatmap as np.ndarray (H, W) in [0, 1]
        """
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        # Pool gradients over spatial dimensions
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted combination of forward activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)                       # Keep only positive influence
        cam = F.interpolate(cam,
                            size=(IMG_SIZE, IMG_SIZE),
                            mode="bilinear",
                            align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def overlay_heatmap(original_img: np.ndarray,
                    cam: np.ndarray,
                    alpha: float = 0.45) -> np.ndarray:
    """
    Overlays Grad-CAM heatmap on the original image.

    Args:
        original_img : BGR image (H, W, 3) as np.ndarray
        cam          : Heatmap from GradCAM.generate()
        alpha        : Heatmap transparency

    Returns:
        Blended image as np.ndarray (RGB)
    """
    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap,
                         (original_img.shape[1], original_img.shape[0]))

    original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB) \
        if len(original_img.shape) == 3 else original_img

    blended = (alpha * heatmap + (1 - alpha) * original_rgb).astype(np.uint8)
    return blended


def preprocess_image(pil_img: Image.Image) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return transform(pil_img).unsqueeze(0)
