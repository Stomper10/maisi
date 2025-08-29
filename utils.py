import numpy as np
import torch
import torch.nn as nn
from monai.networks.nets.resnet import resnet10
from scipy.linalg import sqrtm

def count_parameters(model):
    """
    Ultimate accurate count of the total, trainable, and non-trainable parameters in a model,
    ensuring shared parameters are not double-counted and handling any complex nested submodules.
    
    Args:
        model (torch.nn.Module): The model to count parameters for.
    
    Returns:
        dict: Contains 'total', 'trainable', 'non_trainable' counts of parameters.
    """
    param_count = {
        "total": 0,
        "trainable": 0,
        "non_trainable": 0,
        "by_layer_type": {},  # Added for detailed reporting by layer type
    }

    # Use a set to track shared parameters and avoid double-counting
    seen_params = set()

    for name, param in model.named_parameters():
        param_id = id(param)
        if param_id not in seen_params:
            # Add unique parameter ID to avoid double-counting shared parameters
            seen_params.add(param_id)

            # Count number of elements in the parameter
            num_params = param.numel()

            # Update the total and type-specific counts
            param_count['total'] += num_params
            if param.requires_grad:
                param_count['trainable'] += num_params
            else:
                param_count['non_trainable'] += num_params

            # Track parameters by layer type for detailed reporting
            layer_type = type(param).__name__
            if layer_type not in param_count['by_layer_type']:
                param_count['by_layer_type'][layer_type] = 0
            param_count['by_layer_type'][layer_type] += num_params

            # Optional: print layer-specific details
            #print(f"Layer {name}: {num_params} parameters (Trainable: {param.requires_grad})")

    return param_count

class PretrainedMedicalModel(nn.Module):
    def __init__(self, pretrained_path=None, selected_layers=("layer1", "layer2"), normalize=True):
        super().__init__()
        self.encoder = resnet10(spatial_dims=3, n_input_channels=1, num_classes=1)
        self.selected_layers = selected_layers
        self.normalize = normalize
        self._register_hooks()

        if pretrained_path:
            state_dict = torch.load(pretrained_path, map_location="cpu")
            self.encoder.load_state_dict(state_dict, strict=False)

        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

    def _register_hooks(self):
        self.features = {}

        def save_hook(name):
            def fn(_, __, output):
                self.features[name] = output
            return fn

        for name, module in self.encoder.named_children():
            if name in self.selected_layers:
                module.register_forward_hook(save_hook(name))

    def forward(self, input):
        self.features = {}
        _ = self.encoder(input)
        feats_input = [self.features[name] for name in self.selected_layers]
        return feats_input
    
def compute_mse(original, generated):
    return np.mean((original - generated) ** 2)

def compute_psnr(original, generated, max_intensity=None):
    mse_value = compute_mse(original, generated)
    if mse_value == 0:
        return float('inf')  
    max_intensity = max_intensity or max(original.max(), generated.max())
    return 20 * np.log10(max_intensity / np.sqrt(mse_value))

def compute_fid(mu1, sigma1, mu2, sigma2):
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        if not np.allclose(np.imag(covmean), 0, atol=1e-3):
            print("Warning: Significant imaginary component in sqrtm.")
        covmean = np.real(covmean)
    return np.sum((mu1 - mu2)**2) + np.trace(sigma1 + sigma2 - 2 * covmean)

def clip_and_normalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Clip tensor to [0,1] and ensure it's within expected range for metrics.

    Args:
        tensor: torch.Tensor of shape (B, C, D, H, W)

    Returns:
        Tensor with values in [0,1]
    """
    return torch.clamp(tensor, 0.0, 1.0)