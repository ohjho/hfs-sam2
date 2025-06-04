import os, shutil
import numpy as np
from PIL import Image
from typing import Literal, Any, Union, Generic, List
from pydantic import BaseModel
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.utils.misc import variant_to_config_mapping
from sam2.utils.visualization import show_masks
from ffmpeg_extractor import extract_frames, logger
from toolbox.vid_utils import VidInfo
from toolbox.mask_encoding import b64_mask_encode

variant_checkpoints_mapping = {
    "tiny": "checkpoints/sam2_hiera_tiny.pt",
    "small": "checkpoints/sam2_hiera_small.pt",
    "base_plus": "checkpoints/sam2_hiera_base_plus.pt",
    "large": "checkpoints/sam2_hiera_large.pt",
}


class bbox_xyxy(BaseModel):
    x0: Union[int, float]
    y0: Union[int, float]
    x1: Union[int, float]
    y1: Union[int, float]


class point_xy(BaseModel):
    x: Union[int, float]
    y: Union[int, float]


def mask_to_xyxy(mask: np.ndarray) -> tuple:
    """Convert a binary mask of shape (h, w) to
    xyxy bounding box format (top-left and bottom-right coordinates).
    """
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        logger.warning("mask_to_xyxy: No object found in the mask")
        return None
    x_min = np.min(xs)
    y_min = np.min(ys)
    x_max = np.max(xs)
    y_max = np.max(ys)
    xyxy = (x_min, y_min, x_max, y_max)
    xyxy = tuple([int(i) for i in xyxy])
    return xyxy


def load_sam_image_model(
    # variant: Literal[*variant_checkpoints_mapping.keys()],
    variant: Literal["tiny", "small", "base_plus", "large"],
    device: str = "cpu",
    auto_mask_gen: bool = False,
) -> SAM2ImagePredictor:
    model = build_sam2(
        config_file=variant_to_config_mapping[variant],
        ckpt_path=variant_checkpoints_mapping[variant],
        device=device,
    )
    return (
        SAM2AutomaticMaskGenerator(model)
        if auto_mask_gen
        else SAM2ImagePredictor(sam_model=model)
    )


def load_sam_video_model(
    variant: Literal["tiny", "small", "base_plus", "large"] = "small",
    device: str = "cpu",
) -> Any:
    return build_sam2_video_predictor(
        config_file=variant_to_config_mapping[variant],
        ckpt_path=variant_checkpoints_mapping[variant],
        device=device,
    )


def run_sam_im_inference(
    model: Any,
    image: Image.Image,
    points: Union[List[point_xy], List[dict]] = [],
    point_labels: List[int] = [],
    bboxes: Union[List[bbox_xyxy], List[dict]] = [],
    get_pil_mask: bool = False,
    b64_encode_mask: bool = False,
):
    """returns a list of np masks, each with the shape (h,w) and dtype uint8"""
    assert (
        points or bboxes
    ), f"SAM2 Image Inference must have either bounding boxes or points. Neither were provided."
    if points:
        assert len(points) == len(
            point_labels
        ), f"{len(points)} points provided but {len(point_labels)} labels given."

    # determine multimask_output
    has_multi = False
    if points and bboxes:
        has_multi = True
    elif points and len(list(set(point_labels))) > 1:
        has_multi = True
    elif bboxes and len(bboxes) > 1:
        has_multi = True

    # parse provided bboxes
    bboxes = (
        [bbox_xyxy(**bbox) if isinstance(bbox, dict) else bbox for bbox in bboxes]
        if bboxes
        else []
    )
    points = (
        [point_xy(**p) if isinstance(p, dict) else p for p in points] if points else []
    )

    # setup inference
    image = np.array(image.convert("RGB"))
    model.set_image(image)

    box_coords = (
        np.array([[b.x0, b.y0, b.x1, b.y1] for b in bboxes]) if bboxes else None
    )
    point_coords = np.array([[p.x, p.y] for p in points]) if points else None
    point_labels = np.array(point_labels) if point_labels else None

    masks, scores, _ = model.predict(
        box=box_coords,
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=has_multi,
    )
    # mask here is of shape (X, h, w) of np array, X = number of masks

    if get_pil_mask:
        return show_masks(image, masks, scores=None, display_image=False)
    else:
        output_masks = []
        for i, mask in enumerate(masks):
            if mask.ndim > 2:  # shape (3, h, w)
                mask = np.transpose(mask, (1, 2, 0))  # shape (h,w,3)
                mask = Image.fromarray((mask * 255).astype(np.uint8)).convert("L")
                output_masks.append(np.array(mask))
            else:
                output_masks.append(mask.squeeze().astype(np.uint8))
        return (
            [b64_mask_encode(m) for m in output_masks]
            if b64_encode_mask
            else output_masks
        )


def run_sam_video_inference(
    model: Any,
    video_path: str,
    masks: np.ndarray,
    device: str = "cpu",
    sample_fps: int = None,
    every_x: int = None,
    do_tidy_up: bool = False,
    drop_mask: bool = True,
    async_frame_load: bool = False,
):
    # put video frames into directory
    # TODO:
    # change frame size
    # async frame load
    l_frames_fp = extract_frames(
        video_path,
        fps=sample_fps,
        every_x=every_x,
        overwrite=True,
        im_name_pattern="%05d.jpg",
    )
    vframes_dir = os.path.dirname(l_frames_fp[0])
    vinfo = VidInfo(video_path)
    w = vinfo["frame_width"]
    h = vinfo["frame_height"]

    inference_state = model.init_state(
        video_path=vframes_dir, device=device, async_loading_frames=async_frame_load
    )
    for i, mask in enumerate(masks):
        model.add_new_mask(
            inference_state=inference_state, frame_idx=0, obj_id=i, mask=mask
        )
    masks_generator = model.propagate_in_video(inference_state)

    detections = []
    for i, tracker_ids, mask_logits in masks_generator:
        masks = (mask_logits > 0.0).cpu().numpy().astype(np.uint8)
        for id, mask in zip(tracker_ids, masks):
            mask = mask.squeeze().astype(np.uint8)
            xyxy = mask_to_xyxy(mask)
            if not xyxy:  # mask is empty
                logger.debug(f"track_id {id} is missing mask at frame {i}")
                continue
            x0, y0, x1, y1 = xyxy
            det = {  # miro's detections format for videos
                "frame": i,
                "track_id": id,
                "x": x0 / w,
                "y": y0 / h,
                "w": (x1 - x0) / w,
                "h": (y1 - y0) / h,
                "conf": 1,
            }
            if not drop_mask:
                det["mask_b64"] = b64_mask_encode(mask)
            detections.append(det)

    if do_tidy_up:
        # remove vframes_dir
        shutil.rmtree(vframes_dir)

    return detections
