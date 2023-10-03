from custom_inpainting.utils import BBox, open_coco
import torch

from PIL import Image
from pathlib import Path
from typing import Optional

from diffusers import AutoPipelineForInpainting


class Inpainter: # TODO: Put weights in a docker image

    def __init__(
        self,
        num_inference_steps: int = 20,  # steps between 15 and 30 work well for us
        input_width: int = 512,
    ):
        self.num_inference_steps = num_inference_steps

        self.pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16").to("cuda")
        self.generator = torch.Generator(device="cuda").manual_seed(0)
        self.input_width = input_width
        
        self.input_stride = 32 # Only used for checking the resolution of the input image

        assert self.input_width % self.input_stride == 0, f"Specified input width {self.input_width} is not a multiple of the stride {self.input_stride}"
    
    def inpaint(
            self,
            image: Image.Image,
            mask: Image.Image,
            prompt: str,
        ) -> Image.Image:
        return self.pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            guidance_scale=8.0,
            num_inference_steps=self.num_inference_steps,  # steps between 15 and 30 work well for us
            strength=0.99,  # make sure to use `strength` below 1.0
            generator=self.generator,
        ).images[0]





