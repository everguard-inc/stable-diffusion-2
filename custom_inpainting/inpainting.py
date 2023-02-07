from custom_inpainting.utils import BBox, open_coco
import torch

from PIL import Image
from pathlib import Path
from typing import Optional, Union

from stable_diffusion2_inpainter import StableDiffusion2Inpainter
from diffusers import StableDiffusionInpaintPipeline


class Inpainter:

    def __init__(
        self,
        weights_path: Union[str, Path],
        config_path: Optional[Union[str, Path]] = None,
        half_model: Optional[bool] = False,
        num_inference_steps: Optional[int] = 60,
        guidance_scale: Optional[int] = 7.5,
        input_width: Optional[int] = 512,
    ):
        """
        Args:
            weights_path (str): Path to the .ckpt or directory with weights of the model. If path is .ckpt, then the config_path must be specified.
            config_path (str): Path to the .yaml config file. If path is .ckpt, then the config_path must be specified.
            half_model (bool): Whether to use half precision for the model.
            num_inference_steps (int): Number of inference steps to run.
            guidance_scale (int): Scale of how much generator follow prompt.
            input_width (int): size of the inference image.
        """
        assert not (Path(weights_path).is_file() and config_path is None), "If weights_path is a .ckpt file, then config_path must be specified"
        
        self.num_inference_steps = num_inference_steps
        self.weights_path = Path(weights_path)
        device = torch.device(f"cuda:0" if torch.cuda.is_available() else 'cpu')
        if self.weights_path.is_file():
            self.pipe = StableDiffusion2Inpainter(
                config_path=config_path,
                ckpt_path=self.weights_path,
                device=device,
                half_model=half_model,
                scale=guidance_scale,
            )
        else:
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                self.weights_path,
                # safety_checker=None,    # TODO: check how properly disable safety checker
                guidance_scale=guidance_scale,
            ).to(device)

        self.input_width = input_width
        self.input_stride = 32 # Only used for checking the resolution of the input image
        assert self.input_width % self.input_stride == 0, f"Specified input width {self.input_width} is not a multiple of the stride {self.input_stride}"


    
    def inpaint(
            self,
            image: Image.Image,
            mask: Image.Image,
            prompt: str,
        ) -> Image.Image:
        if self.weights_path.is_file():
            return self.pipe(
                    input_image=image,
                    input_mask=mask,
                    prompt=prompt,
                    num_inference_steps=self.num_inference_steps,
                )[0]
        else:
            return self.pipe(
                    prompt=prompt, 
                    image=image, 
                    mask_image=mask, 
                    height=self.input_width, 
                    width=self.input_width,
                    num_inference_steps=self.num_inference_steps,
                ).images[0]




