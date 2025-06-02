import gradio as gr
import spaces, torch
from samv2_handler import load_sam_image_model, run_sam_im_inference
from PIL import Image
from typing import Union


@spaces.GPU
def load_im_model(variant, auto_mask_gen: bool = False):
    return load_sam_image_model(
        variant=variant, device="cuda", auto_mask_gen=auto_mask_gen
    )


@spaces.GPU
def detect_image(
    im: Image.Image,
    variant: str,
    bboxes: Union[list, str] = None,
    points: Union[list, str] = None,
    point_labels: Union[list, str] = None,
):
    """
    SAM2 Image Segmentation

    Args:
        im: Pillow Image
        object_name: the object you would like to detect
        mode: point or object_detection
    Returns:
        list: a list of masks
    """
    bboxes = json.loads(bboxes) if isinstance(bboxes, str) else bboxes
    model = load_im_model(variant=variant)
    return run_sam_im_inference(
        model, image=im, bboxes=bboxes, get_pil_mask=False, b64_encode_mask=True
    )


with gr.Blocks() as demo:
    with gr.Tab("Images"):
        gr.Interface(
            fn=detect_image,
            inputs=[
                gr.Image(label="Input Image", type="pil"),
                gr.Dropdown(
                    label="Model Variant",
                    choices=["tiny", "small", "base_plus", "large"],
                ),
                gr.JSON(
                    label='Bounding Boxes (JSON list of dicts: [{"x0":..., "y0":..., "x1":..., "y1":...}, ...])',
                    optional=True,
                ),
            ],
            outputs=gr.JSON(label="Output JSON"),
            title="SAM2 for Images",
        )
demo.launch(
    mcp_server=True, app_kwargs={"docs_url": "/docs"}  # add FastAPI Swagger API Docs
)
