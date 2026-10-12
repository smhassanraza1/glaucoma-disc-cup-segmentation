"""Upload a fundus photo, see the outlined optic disc and cup and the vertical cup-to-disc ratio.

    uv run --extra demo python app/demo.py

Needs trained checkpoints in ``checkpoints/`` (``locator.pt`` and ``segmenter.pt``). This is a research demo,
not a medical device: the numbers must not be used to screen, diagnose or treat anyone.
"""

import os
from pathlib import Path

import gradio as gr
import numpy as np

from glaucoma.pipeline import load_checkpoint, pick_device, predict_masks
from glaucoma.visualize import describe, overlay

CHECKPOINTS = Path(os.environ.get("CHECKPOINTS", "checkpoints"))
NOTICE = (
    "**Research demo, not a medical tool.** It was trained on 275 photos from one dataset and has not been "
    "validated for clinical use. Do not use it to screen, diagnose or treat anyone."
)

device = pick_device()
locator = load_checkpoint(CHECKPOINTS / "locator.pt", device)
segmenter = load_checkpoint(CHECKPOINTS / "segmenter.pt", device)


def analyse(image: np.ndarray) -> tuple[np.ndarray, str]:
    disc, cup = predict_masks(image, locator, segmenter, device)
    return overlay(image, disc, cup), describe(disc, cup)


demo = gr.Interface(
    fn=analyse,
    inputs=gr.Image(type="numpy", label="Fundus photo (optic disc visible)"),
    outputs=[gr.Image(label="Disc (green outline) and cup (blue outline)"), gr.Textbox(label="Measurement")],
    title="Optic disc and cup segmentation",
    description=NOTICE,
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"))
