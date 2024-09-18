# Prediction interface for Cog ⚙️
# https://cog.run/python

import mimetypes
import subprocess
import time
import torch
import os
from controlnet_flux import FluxControlNetModel
from transformer_flux import FluxTransformer2DModel
from pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline
from cog import BasePredictor, Input, Path
from typing import Iterator
from PIL import Image
from typing import List, Union

mimetypes.add_type("image/webp", ".webp")


MODEL_CACHE = "model_cache"
BASE_URL = f"https://weights.replicate.delivery/default/flux-controlnet-inpainting/{MODEL_CACHE}/"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE


def download_weights(url: str, dest: str) -> None:
    start = time.time()
    print("[!] Initiating download from URL: ", url)
    print("[~] Destination path: ", dest)
    if ".tar" in dest:
        dest = os.path.dirname(dest)
    command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
        raise
    print("[+] Download completed in: ", time.time() - start, "seconds")


def make_multiple_of_16(n):
    """Rounds up to the next multiple of 16."""
    return ((n + 15) // 16) * 16


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient."""
        # Ensure models are downloaded properly by specifying cache directories
        if not os.path.exists(MODEL_CACHE):
            os.makedirs(MODEL_CACHE)
        model_files = [
            "models--alimama-creative--FLUX.1-dev-Controlnet-Inpainting-Alpha.tar",
            "models--black-forest-labs--FLUX.1-dev.tar",
        ]
        for model_file in model_files:
            url = BASE_URL + model_file
            filename = url.split("/")[-1]
            dest_path = os.path.join(MODEL_CACHE, filename)
            if not os.path.exists(dest_path.replace(".tar", "")):
                download_weights(url, dest_path)

        os.makedirs(MODEL_CACHE, exist_ok=True)

        controlnet_model_name = (
            "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha"
        )
        transformer_model_name = "black-forest-labs/FLUX.1-dev"
        base_model_name = "black-forest-labs/FLUX.1-dev"

        # Load ControlNet model
        self.controlnet = FluxControlNetModel.from_pretrained(
            controlnet_model_name, torch_dtype=torch.bfloat16, cache_dir=MODEL_CACHE
        )
        # Load Transformer model
        self.transformer = FluxTransformer2DModel.from_pretrained(
            transformer_model_name,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            cache_dir=MODEL_CACHE,
        )
        # Load the pipeline
        self.pipe = FluxControlNetInpaintingPipeline.from_pretrained(
            base_model_name,
            controlnet=self.controlnet,
            transformer=self.transformer,
            torch_dtype=torch.bfloat16,
            cache_dir=MODEL_CACHE,
        ).to("cuda")
        self.pipe.transformer.to(torch.bfloat16)
        self.pipe.controlnet.to(torch.bfloat16)

    def predict(
        self,
        image: Path = Input(
            description="Upload an image for inpainting. This will be the base image that will be partially modified.",
        ),
        mask: Path = Input(
            description=(
                "Upload a mask image for inpainting. White areas (255) indicate regions to be inpainted, "
                "while black areas (0) will be preserved from the original image."
            ),
        ),
        prompt: str = Input(
            description="Enter a text description to guide the image generation process.",
            default="A colorful hot air balloon floating over a serene landscape.",
        ),
        negative_prompt: str = Input(
            description="Negative text prompt. Used to reduce or avoid certain aspects in the generated image.",
            default="",
        ),
        controlnet_conditioning_scale: float = Input(
            description=(
                "ControlNet conditioning scale."
            ),
            default=0.9,
            ge=0.0,
            le=2.0,
        ),
        num_inference_steps: int = Input(
            description="Set the number of denoising steps. More steps generally result in higher quality but slower generation.",
            ge=1,
            le=100,
            default=28,
        ),
        guidance_scale: float = Input(
            description=(
                "Guidance scale for classifier-free guidance. Higher values encourage the model to generate images "
                "that are closer to the text prompt."
            ),
            ge=0,
            le=20,
            default=3.5,
        ),
        true_guidance_scale: float = Input(
            description="True guidance scale for the transformer model.",
            ge=0,
            le=20,
            default=3.5,
        ),
        num_outputs: int = Input(
            description="Number of images to generate.",
            ge=1,
            le=10,
            default=1,
        ),
        seed: int = Input(
            description="Set a seed for reproducible generation. Leave as None for random results.", 
            default=None
        ),
        output_format: str = Input(
            description="Choose the file format for the output images.",
            choices=["webp", "jpg", "png"],
            default="webp",
        ),
        output_quality: int = Input(
            description=(
                "Quality of the output images (applicable for 'jpg' and 'webp'). Value between 1 (lowest quality) "
                "and 100 (highest quality). Ignored for 'png'."
            ),
            default=80,
            ge=1,
            le=100,
        ),
    ) -> List[Path]:
        """Run batch predictions on the model and return a list of generated images."""

        # Generate a list of seeds
        if seed is None or seed < 0:
            seeds = [int.from_bytes(os.urandom(4), "big") for _ in range(num_outputs)]
        else:
            seeds = [seed + i for i in range(num_outputs)]
        generators = [torch.Generator(device="cuda").manual_seed(s) for s in seeds]
        print(f"Using seeds: {seeds}")

        # Load and prepare images
        image = Image.open(str(image)).convert("RGB")
        mask = Image.open(str(mask)).convert("RGB")

        # Ensure dimensions are multiples of 16
        width = make_multiple_of_16(max(image.width, mask.width))
        height = make_multiple_of_16(max(image.height, mask.height))
        size = (width, height)

        if image.size != size:
            print(f"Resizing input image from {image.size} to {size}")
            image = image.resize(size)
        if mask.size != size:
            print(f"Resizing mask image from {mask.size} to {size}")
            mask = mask.resize(size)

        # Generate images
        result = self.pipe(
            prompt=[prompt] * num_outputs,
            negative_prompt=[negative_prompt] * num_outputs,
            height=height,
            width=width,
            control_image=image,
            control_mask=mask,
            num_inference_steps=num_inference_steps,
            generator=generators,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            guidance_scale=guidance_scale,
            true_guidance_scale=true_guidance_scale,
        )

        # Save the results
        output_paths = []
        for idx, img in enumerate(result.images):
            extension = output_format.lower()
            extension = "jpeg" if extension == "jpg" else extension
            output_path = f"/tmp/output_{idx}.{extension}"

            print(f"[~] Saving to {output_path}...")
            print(f"[~] Output format: {extension.upper()}")
            if output_format != "png":
                print(f"[~] Output quality: {output_quality}")

            save_params = {"format": extension.upper()}
            if output_format != "png":
                save_params["quality"] = output_quality
                save_params["optimize"] = True

            img.save(output_path, **save_params)
            output_paths.append(Path(output_path))

        return output_paths
