import gc

import PIL.Image
import torch

from stable_diffusion_xl_partedit import PartEditPipeline, DotDictExtra, Binarization, PaddingStrategy, EmptyControl
from diffusers import AutoencoderKL
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor

from huggingface_hub import hf_hub_download

available_pts = [
    "pt/torso_custom.pt", # this is human torso only
    "pt/chair_custom.pt", # this is seat of the chair only
    "pt/carhood_custom.pt", 
    "pt/partimage_biped_head.pt", # this is essentially monkeys
    "pt/partimage_carbody.pt", # this is everything except the wheels
    "pt/partimage_human_hair.pt", 
    "pt/partimage_human_head.pt", # this is essentially faces
    "pt/partimage_human_torso.pt", # use custom on in favour of this one
    "pt/partimage_quadruped_head.pt", # this is essentially animals on 4 legs
]

def download_part(index):
    return hf_hub_download(
        repo_id="Aleksandar/PartEdit-extra",
        repo_type="dataset",
        filename=available_pts[index]
    )

PART_TOKENS = {
    "human_head": download_part(6),
    "human_hair": download_part(5),
    "human_torso_custom": download_part(0), # custom one
    "chair_custom": download_part(1),
    "carhood_custom": download_part(2),
    "carbody": download_part(4),
    "biped_head": download_part(8),
    "quadruped_head": download_part(3),
    "human_torso": download_part(7), # based on partimage
}


class PartEditSDXLModel:
    MAX_NUM_INFERENCE_STEPS = 50

    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
            self.sd_pipe, self.partedit_pipe = PartEditPipeline.default_pipeline(self.device)
        else:
            self.pipe = None

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: int = 0,
        eta: float = 0,
    ) -> PIL.Image.Image:

        if not torch.cuda.is_available():
            raise RuntimeError("This demo does not work on CPU!")

        out = self.sd_pipe(
            prompt=prompt,
            # negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            eta=eta,
            generator=torch.Generator().manual_seed(seed),
        ).images[0]

        gc.collect()
        torch.cuda.empty_cache()
        return out

    def edit(
        self,
        prompt: str,
        subject: str,
        part: str,
        edit: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: int = 0,
        eta: int = 0,
        t_e: int = 50,
    ) -> PIL.Image.Image:

        # Sanity Checks
        if not torch.cuda.is_available():
            raise RuntimeError("This demo does not work on CPU!")

        if part in PART_TOKENS:
            token_path = PART_TOKENS[part]
        else:
            raise ValueError(f"Part `{part}` is not supported!")

        if subject not in prompt:
            raise ValueError(f"The subject `{subject}` does not exist in the original prompt!")

        prompts = [
            prompt,
            prompt.replace(subject, edit),
        ]

        # PartEdit Parameters
        cross_attention_kwargs = {
            "edit_type": "replace",
            "n_self_replace": 0.0,
            "n_cross_replace": {"default_": 1.0, edit: 0.4},
        }
        extra_params = DotDictExtra()
        extra_params.update({"omega": 1.5, "edit_steps": t_e})

        out = self.partedit_pipe(
            prompt=prompts,
            # negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            eta=eta,
            generator=torch.Generator().manual_seed(seed),
            cross_attention_kwargs=cross_attention_kwargs,
            extra_kwargs=extra_params,
            embedding_opt=token_path,
        ).images[:2][::-1]

        mask = self.partedit_pipe.visualize_map_across_time()
        gc.collect()
        torch.cuda.empty_cache()
        return out, mask
