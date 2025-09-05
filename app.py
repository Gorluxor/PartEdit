#!/usr/bin/env python

import os
import random
from typing import Optional, Tuple, Union, List

import numpy as np
import PIL.Image
import gradio as gr
import torch
import spaces  # 👈 ZeroGPU support

from model import PartEditSDXLModel, PART_TOKENS
from datasets import load_dataset

import base64
from io import BytesIO
import tempfile
import uuid

MAX_SEED = np.iinfo(np.int32).max
CACHE_EXAMPLES = os.environ.get("CACHE_EXAMPLES") == "1"
AVAILABLE_TOKENS = list(PART_TOKENS.keys())

# Download examples directly from the huggingface PartEdit-Bench
# Login using e.g. `huggingface-cli login` or `hf login` if needed.
bench = load_dataset("Aleksandar/PartEdit-Bench", revision="v1.1", split="synth")

use_examples = None  # all with None
logo = "assets/partedit.png"
loaded_logo = PIL.Image.open(logo).convert("RGB")
# base encoded logo 

logo_encoded = None
with open(logo, "rb") as f:
    logo_encoded = base64.b64encode(f.read()).decode()


def _save_image_for_download(edited: Union[PIL.Image.Image, np.ndarray, str, List]) -> str:
    """Save the first edited image to a temp file and return its filepath."""
    # clone to be sure we don't modify the input
    edited = edited.copy() 
    img = edited[0] if isinstance(edited, list) else edited
    if isinstance(img, str):
        # path on disk already
        return img
    if isinstance(img, np.ndarray):
        img = PIL.Image.fromarray(img)
    assert isinstance(img, PIL.Image.Image), "Edited output must be PIL, ndarray, str path, or list of these."
    out_path = os.path.join(tempfile.gettempdir(), f"partedit_{uuid.uuid4().hex}.png")
    img.save(out_path)
    return out_path



def get_example(idx, bench):
    # [prompt_original, subject, token_cls, edit, "", 50, 7.5, seed, 50]
    example = bench[idx]
    return [
        example["prompt_original"],
        example["subject"],
        example["token_cls"],
        example["edit"],
        "",
        50,
        7.5,
        example["seed"],
        50,
    ]

examples = [get_example(idx, bench) for idx in (use_examples if use_examples is not None else range(len(bench)))]
first_ex = examples[0] if len(examples) else ["", "", AVAILABLE_TOKENS[0], "", "", 50, 7.5, 0, 50]

title = f"""
<div style="display: flex; align-items: center;">
    <img src="data:image/png;base64,{logo_encoded}" alt="PartEdit Logo">
    <div style="margin-left: 10px;">
      <h1 style="margin: 0;">PartEdit with SDXL</h1>
      <p style="margin: 2px 0 0 0;">Official demo for the PartEdit paper.</p>
      <h2 style="margin: 6px 0 0 0;">PartEdit: Fine-Grained Image Editing using Pre-Trained Diffusion Models</h2>
      <p style="margin: 6px 0 0 0; font-size: 14px;">
        It <b>simultaneously predicts the part-localization mask and edits the original trajectory</b>.
        Supports <b>Hugging Face ZeroGPU</b> and one-click <b>Duplicate</b> for private use.
      </p>
    </div>
</div>
"""


def _as_gallery(edited: Union[PIL.Image.Image, np.ndarray, str, List]) -> List:
    """Ensure the output fits a Gallery component."""
    if isinstance(edited, list):
        return edited
    return [edited]


def edit_demo(model: PartEditSDXLModel) -> gr.Blocks:
    @spaces.GPU(duration=120)  # 👈 request a ZeroGPU allocation during this call
    def run(
        prompt: str,
        subject: str,
        part: str,
        edit: str,
        negative_prompt: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: int = 0,
        t_e: int = 50,
        progress=gr.Progress(track_tqdm=True),
    ) -> Tuple[List, Optional[PIL.Image.Image]]:
        if seed == -1:
            seed = random.randint(0, MAX_SEED)

        out = model.edit(
            prompt=prompt,
            subject=subject,
            part=part,
            edit=edit,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            t_e=t_e,
        )

        # Accept either (image, mask) or just image from model.edit
        if isinstance(out, tuple) and len(out) == 2:
            edited, mask_img = out
        else:
            edited, mask_img = out, None

        download_path = _save_image_for_download(edited)
        return _as_gallery(edited), mask_img, gr.update(value=download_path, visible=True)

    

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Group():
                    prompt = gr.Textbox(
                        first_ex[0],  # <- was "a closeup of a man full-body"
                        placeholder="Prompt",
                        label="Original Prompt",
                        show_label=True,
                        max_lines=1,
                    )
                    with gr.Row():
                        subject = gr.Textbox(value=first_ex[1], label="Subject", show_label=True, max_lines=1)
                        edit = gr.Textbox(value=first_ex[3], label="Edit", show_label=True, max_lines=1)
                        part = gr.Dropdown(label="Part Name", choices=AVAILABLE_TOKENS, value=first_ex[2])

                    seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=int(first_ex[7]))
                    run_button = gr.Button("Apply Edit")

                with gr.Accordion("Advanced options", open=False):
                    negative_prompt = gr.Textbox(label="Negative prompt", value=first_ex[4])
                    num_inference_steps = gr.Slider(
                        label="Number of steps",
                        minimum=1,
                        maximum=PartEditSDXLModel.MAX_NUM_INFERENCE_STEPS,
                        step=1,
                        value=int(first_ex[5]),
                    )
                    guidance_scale = gr.Slider(
                        label="Guidance scale",
                        minimum=0.1,
                        maximum=30.0,
                        step=0.1,
                        value=float(first_ex[6]),
                    )
                    t_e = gr.Slider(
                        label="Editing steps",
                        minimum=1,
                        maximum=PartEditSDXLModel.MAX_NUM_INFERENCE_STEPS,
                        step=1,
                        value=int(first_ex[8]),
                    )
                with gr.Accordion('Citation', open=True):
                    gr.Markdown(citation)


            with gr.Column(scale=3):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=120):
                        mask = gr.Image(label="Editing Mask", width=100, height=100, show_label=True)
                    with gr.Column(scale=7):
                        result = gr.Gallery(
                            label="Edited Image",
                            height=700,
                            object_fit="fill",
                            preview=True,
                            selected_index=0,
                            show_label=True,
                        )
                download_btn = gr.File(
                    label="Download full-resolution",
                    type="filepath",
                    file_count="single",   # <-- keeps it to one file
                    interactive=False,
                    height=48,             # <-- compact
                    visible=False          # <-- hide until we have a file
                )

        inputs = [prompt, subject, part, edit, negative_prompt, num_inference_steps, guidance_scale, seed, t_e]

        gr.Examples(
            examples=examples,
            inputs=inputs,
            outputs=[result, mask, download_btn],
            fn=run,
            cache_examples=CACHE_EXAMPLES,
        )

        run_button.click(fn=run, inputs=inputs, outputs=[result, mask, download_btn], api_name="run")

    return demo


badges_text = r"""
<div style="text-align: center; display: flex; justify-content: center; gap: 5px; flex-wrap: wrap;">
  <a href="https://gorluxor.github.io/part-edit/">
    <img alt="Project Page" src="https://img.shields.io/badge/%F0%9F%8C%90%20Project%20Page-PartEdit-blue">
  </a>
  <a href="https://arxiv.org/abs/2502.04050">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2502.04050-b31b1b.svg">
  </a>
  <a href="https://huggingface.co/datasets/Aleksandar/PartEdit-Bench">
    <img alt="HF Dataset: PartEdit-Bench" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-PartEdit--Bench-blue">
  </a>
  <a href="https://huggingface.co/datasets/Aleksandar/PartEdit-extra">
    <img alt="HF Dataset: PartEdit-extra" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-PartEdit--extra-blue">
  </a>
  <a href="https://s2025.siggraph.org/">
    <img alt="SIGGRAPH 2025" src="https://img.shields.io/badge/%F0%9F%8E%A8%20Accepted-SIGGRAPH%202025-blueviolet">
  </a>
  <a href="https://github.com/Gorluxor/partedit/blob/main/LICENSE">
    <img alt="Code License" src="https://img.shields.io/badge/license-MIT-blue.svg">
  </a>
</div>
""".strip()

citation = r"""
If you use this demo, please cite the following paper:
```
@inproceedings{cvejic2025partedit,
  title={PartEdit: Fine-Grained Image Editing using Pre-Trained Diffusion Models},
  author={Cvejic, Aleksandar and Eldesokey, Abdelrahman and Wonka, Peter},
  booktitle={Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers},
  pages={1--11},
  year={2025}
}
```
"""

DESCRIPTION = title + badges_text

if not torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CPU 🥶 This demo does not work on CPU. On ZeroGPU Spaces, a GPU will be requested when you click <b>Apply Edit</b>.</p>"

def running_in_hf_space() -> bool:
    # Common env vars present on Hugging Face Spaces
    return (
        os.getenv("SYSTEM") == "spaces" or
        any(os.getenv(k) for k in (
            "SPACE_ID", "HF_SPACE_ID", "SPACE_REPO_ID",
            "SPACE_REPO_NAME", "SPACE_AUTHOR_NAME", "SPACE_TITLE"
        ))
    )

if __name__ == "__main__":
    model = PartEditSDXLModel()

    with gr.Blocks(css="style.css") as demo:
        gr.Markdown(DESCRIPTION)

        # Always show Duplicate button on Spaces
        gr.DuplicateButton(
            value="Duplicate Space for private use",
            elem_id="duplicate-button",
            variant="huggingface",
            size="lg",
            visible=running_in_hf_space(),
        )

        # Single tab: PartEdit only
        with gr.Tabs():
            with gr.Tab(label="PartEdit", id="edit"):
                edit_demo(model)

        demo.queue(max_size=20).launch()
