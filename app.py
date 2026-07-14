import gradio as gr
import spaces, torch, os, json
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


@spaces.GPU
def load_im_model(variant):
    return load_sam_image_model(variant=variant, device="cuda")


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
) -> list[str]:
    """SAM2 image segmentation: segment objects in an image from box and/or point prompts and return their masks.

    Provide prompts as bounding boxes and/or points (points require point_labels). Returns a JSON list of segmentation masks, one mask per prompt -- one mask per bounding box, or a single mask for a points-only prompt -- in the same order as the input prompts. Each mask is a base64-encoded 1-bit (binary) PNG string; when decoded the PNG has the same width and height as the original input image, with pixel value 1 inside the object and 0 elsewhere.

    Args:
        im: The RGB image to segment, as a Pillow Image.
        variant: SAM2 model variant to load; one of "tiny", "small", "base_plus", or "large".
        bboxes: Object bounding boxes to segment, as a JSON list of dicts [{"x0":..., "y0":..., "x1":..., "y1":...}, ...] in ABSOLUTE PIXEL coordinates of the input image (top-left corner x0/y0, bottom-right corner x1/y1) -- the albumentations "pascal_voc" format; either bboxes or points must be provided.
        points: Object points to segment, as a JSON list of dicts [{"x":..., "y":...}, ...] in ABSOLUTE PIXEL coordinates of the input image; requires point_labels.
        point_labels: JSON list of integers, one per point, where 1 marks a foreground (include) point and 0 a background (exclude) point.
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
        b64_encode_mask=True,
    )


@spaces.GPU(
    duration=120
)  # user must have 2-minute of inference time left at the time of calling
@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def process_video(
    video_path: str,
    variant: str,
    masks: Union[list, str],
    drop_masks: bool = False,
    ref_frame_idx: int = 0,
    async_frame_load: bool = True,
) -> list[dict]:
    """SAM2 video segmentation: track objects through a video given their segmentation masks on a reference frame.

    Seed the objects to track by supplying their masks on the reference frame (ref_frame_idx); SAM2 then propagates each object forward -- and, when ref_frame_idx is nonzero, also backward -- through every frame. Returns a JSON list of per-frame detections, one entry per tracked object per frame in which it appears (frames where an object is absent are skipped). Each detection is a dict with: "frame" (integer frame index); "track_id" (integer object id, stable across frames); "x", "y", "w", "h" (the object's bounding box as top-left-x, top-left-y, width, height, each NORMALIZED to 0.0-1.0 by dividing by the frame width or height -- this is the albumentations "coco" layout [x_min, y_min, width, height] but normalized to 0-1 rather than absolute pixels, and it is NOT [x_min, y_min, x_max, y_max]; box formats documented at https://albumentations.ai/docs/3-basic-usage/bounding-boxes-augmentations/#bounding-box-formats ); "conf" (always 1); and, unless drop_masks is true, "mask_b64" (a base64-encoded 1-bit PNG string the same width and height as the video frame, 1 inside the object and 0 elsewhere).

    Args:
        video_path: Filesystem path to the input video file.
        variant: SAM2 model variant to load; one of "tiny", "small", "base_plus", or "large".
        masks: JSON list of base64-encoded 1-bit PNG masks for the reference frame, one per object to track, e.g. ["b'iVBORw0KGgo...'", ...]; the b'...' literal wrapper is accepted and stripped.
        drop_masks: When true, omit the "mask_b64" field from every detection so only bounding-box information is returned.
        ref_frame_idx: Frame index the provided masks correspond to; a nonzero value triggers bidirectional tracking (forward and backward from this frame).
        async_frame_load: When true, load video frames in parallel with propagation to reduce inference time.
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
        drop_mask=drop_masks,
        async_frame_load=async_frame_load,
        ref_frame_idx=ref_frame_idx,
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
                    label="Masks for Objects of Interest in the First Frame",
                    value=None,
                    lines=5,
                    placeholder="""
                    JSON list of base64 encoded masks, e.g.: ["b'iVBORw0KGgoAAAANSUhEUgAABDgAAAeAAQAAAAADGtqnAAAXz...'",...]
                    """,
                ),
                gr.Checkbox(
                    label="Drop Masks",
                    info="remove base64 encoded masks from result JSON",
                    value=True,
                ),
                gr.Number(
                    label="Reference Frame Index",
                    info="frame index for the provided object masks",
                    value=0,
                    precision=0,
                ),
                gr.Checkbox(
                    label="async frame load",
                    info="start inference in parallel to frame loading",
                ),
            ],
            outputs=gr.JSON(label="Output JSON"),
            title="SAM2 for Videos",
            api_name="process_video",
        )

# Model weights are auto-downloaded from the HuggingFace Hub on first use.
demo.launch(
    mcp_server=True, app_kwargs={"docs_url": "/docs"}  # add FastAPI Swagger API Docs
)
