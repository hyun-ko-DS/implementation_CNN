import torch
import torch.nn as nn
import numpy as np
from PIL import Image

class Conv2DGray(nn.Module):
    def __init__(self, kernel):
        super(Conv2DGray, self).__init__()
        self.conv = nn.Conv2d(in_channels = 1, out_channels = 1,
                              kernel_size = kernel.shape, bias = False)
        
        with torch.no_grad():
            self.conv.weight = nn.Parameter(torch.tensor(kernel, dtype = torch.float32).unsqueeze(0).unsqueeze(0))

    def forward(self, image: Image.Image):
        # pre-processing on an input image: convert to tensor
        image_tensor = self.pre_proc(image)
        output_tensor = self.conv(image_tensor)

        # post-processing on an output tensor: convert to image
        output_image = self.post_proc(output_tensor)

        return output_image
    
    def pre_proc(self, gray_image: Image.Image) -> torch.Tensor:
        # convert to numpy array
        image_np = np.array(gray_image).astype(np.float32)

        # convert numpy array to tensor
        image_tensor = torch.tensor(image_np).unsqueeze(0).unsqueeze(0)

        return image_tensor

    def post_proc(self, image_tensor: torch.Tensor) -> Image.Image:
        output_image = image_tensor.squeeze(0).squeeze(0).detach().numpy()
        return Image.fromarray(output_image.astype(np.uint8))