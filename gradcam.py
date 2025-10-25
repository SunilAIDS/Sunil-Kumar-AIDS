import torch
import torch.nn.functional as F
import cv2
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_image, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_image)
        if class_idx is None:
            class_idx = output.argmax().item()
        target = output[0, class_idx]
        target.backward()
        # Compute weights
        weights = self.gradients.mean(dim=(2,3), keepdim=True)
        cam = (weights * self.activations).sum(1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        # Resize cam to match input image
        cam = cv2.resize(cam, (input_image.shape[3], input_image.shape[2]))  # Width, Height
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam
