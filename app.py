import gradio as gr
import spaces, torch, os, requests, json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from typing import Union
import numpy as np
from samv2_handler import (
    load_sam_image_model,
    run_sam_im_inference,
    load_sam_video_model,
    run_sam_video_inference,
    logger,
)
from toolbox.mask_encoding import b64_mask_decode

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def download_checkpoints():
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    # Read URLs from the file
    with open(checkpoint_dir / "sam2_checkpoints_url.txt", "r") as f:
        urls = [url.strip() for url in f.readlines() if url.strip()]

    for url in urls:
        filename = url.split("/")[-1]
        output_path = checkpoint_dir / filename

        if output_path.exists():
            print(f"Checkpoint {filename} already exists, skipping...")
            continue

        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))

        with open(output_path, "wb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        print(f"Downloaded {filename} successfully!")


@spaces.GPU
def load_im_model(variant, auto_mask_gen: bool = False):
    return load_sam_image_model(
        variant=variant, device="cuda", auto_mask_gen=auto_mask_gen
    )


@spaces.GPU
def load_vid_model(variant):
    return load_sam_video_model(variant=variant, device="cuda")


@spaces.GPU
@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def process_image(
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
        variant: SAM2 model variant
        bboxes: bounding boxes of objects to segment, expressed as a list of dicts: [{"x0":..., "y0":..., "x1":..., "y1":...}, ...]
        points: points of objects to segment, expressed as a list of dicts [{"x":..., "y":...}, ...]
        point_labels: list of integar
    Returns:
        list: a list of masks in the form of bit64 encoded strings
    """
    # input validation
    has_bboxes = type(bboxes) != type(None) and bboxes != ""
    has_points = type(points) != type(None) and points != ""
    has_point_labels = type(point_labels) != type(None) and point_labels != ""
    assert has_bboxes or has_points, f"either bboxes or points must be provided."
    if has_points:
        assert has_point_labels, f"point_labels is required if points are provided."

    bboxes = json.loads(bboxes) if isinstance(bboxes, str) and has_bboxes else bboxes
    points = json.loads(points) if isinstance(points, str) and has_points else points
    point_labels = (
        json.loads(point_labels)
        if isinstance(point_labels, str) and has_point_labels
        else point_labels
    )
    if has_points:
        assert len(points) == len(
            point_labels
        ), f"{len(points)} points provided but there are {len(point_labels)} labels."

    model = load_im_model(variant=variant)
    return run_sam_im_inference(
        model,
        image=im,
        bboxes=bboxes,
        points=points,
        point_labels=point_labels,
        get_pil_mask=False,
        b64_encode_mask=True,
    )


@spaces.GPU
@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def process_video(video_path: str, variant: str, masks: Union[list, str]):
    """
    SAM2 Video Segmentation

    Args:
        video_path: path to video object
        variant: SAMv2's model variant
        masks: a list of b64 encoded masks for the first frame of the video, indicating the objects to be tracked
    Returns:
        list: a list of masks
    """
    model = load_vid_model(variant=variant)
    masks = json.loads(masks) if isinstance(masks, str) else masks
    logger.debug(f"masks---\n{masks}")
    masks = [
        m[2:-1].encode() if m.startswith("b'") and m.endswith("'") else m for m in masks
    ]  # expect the b'' literal to be included
    masks = np.array([b64_mask_decode(m).astype(np.uint8) for m in masks])
    logger.debug(f"masks---\n{masks}")
    return run_sam_video_inference(
        model,
        video_path=video_path,
        masks=masks,
        device="cuda",
        do_tidy_up=True,
        drop_mask=False,
    )


with gr.Blocks() as demo:
    with gr.Tab("Images"):
        gr.Interface(
            fn=process_image,
            inputs=[
                gr.Image(label="Input Image", type="pil"),
                gr.Dropdown(
                    label="Model Variant",
                    choices=["tiny", "small", "base_plus", "large"],
                ),
                gr.Textbox(
                    label="Bounding Boxes",
                    value=None,
                    lines=5,
                    placeholder='JSON list of dicts: [{"x0":..., "y0":..., "x1":..., "y1":...}, ...]',
                ),
                gr.Textbox(
                    label="Points",
                    lines=3,
                    placeholder='JSON list of dicts: [{"x":..., "y":...}, ...]',
                ),
                gr.Textbox(label="Points' Labels", placeholder="JSON List of Integars"),
            ],
            outputs=gr.JSON(label="Output JSON"),
            title="SAM2 for Images",
            api_name="process_image",
        )
    with gr.Tab("Videos"):
        gr.Interface(
            fn=process_video,
            inputs=[
                gr.Video(label="Input Video"),
                gr.Dropdown(
                    label="Model Variant",
                    choices=["tiny", "small", "base_plus", "large"],
                ),
                gr.Textbox(
                    label='Masks for Objects of Interest in the First Frame (JSON list of dicts: [{"x0":..., "y0":..., "x1":..., "y1":...}, ...])',
                    value=None,
                    lines=5,
                    placeholder='JSON list of dicts: [{"x0":..., "y0":..., "x1":..., "y1":...}, ...]',
                ),
            ],
            outputs=gr.JSON(label="Output JSON"),
            title="SAM2 for Videos",
            api_name="process_video",
        )

# Download checkpoints before launching the app
download_checkpoints()
demo.launch(
    mcp_server=True, app_kwargs={"docs_url": "/docs"}  # add FastAPI Swagger API Docs
)
