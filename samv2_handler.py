import os, shutil
import glob
import numpy as np
import torch
from PIL import Image
from typing import Literal, Any, Union, Generic, List
from pydantic import BaseModel
from transformers import (
    Sam2Model,
    Sam2Processor,
    Sam2VideoModel,
    Sam2VideoProcessor,
)
from ffmpeg_extractor import extract_frames, logger
from visualizer import mask_to_xyxy
from toolbox.vid_utils import VidInfo, VidReader
from toolbox.mask_encoding import b64_mask_encode

# from toolbox.img_utils import get_pil_im

# SAM2.1 model variants on the HuggingFace Hub; weights are auto-downloaded and cached.
variant_hf_mapping = {
    "tiny": "facebook/sam2.1-hiera-tiny",
    "small": "facebook/sam2.1-hiera-small",
    "base_plus": "facebook/sam2.1-hiera-base-plus",
    "large": "facebook/sam2.1-hiera-large",
}


class bbox_xyxy(BaseModel):
    x0: Union[int, float]
    y0: Union[int, float]
    x1: Union[int, float]
    y1: Union[int, float]


class point_xy(BaseModel):
    x: Union[int, float]
    y: Union[int, float]


def load_sam_image_model(
    # variant: Literal[*variant_hf_mapping.keys()],
    variant: Literal["tiny", "small", "base_plus", "large"],
    device: str = "cpu",
):
    """returns a (Sam2Model, Sam2Processor) tuple for image inference"""
    repo_id = variant_hf_mapping[variant]
    model = Sam2Model.from_pretrained(repo_id).to(device)
    processor = Sam2Processor.from_pretrained(repo_id)
    return model, processor


def load_sam_video_model(
    variant: Literal["tiny", "small", "base_plus", "large"] = "small",
    device: str = "cpu",
):
    """returns a (Sam2VideoModel, Sam2VideoProcessor) tuple for video inference"""
    repo_id = variant_hf_mapping[variant]
    model = Sam2VideoModel.from_pretrained(repo_id).to(device)
    processor = Sam2VideoProcessor.from_pretrained(repo_id)
    return model, processor


def run_sam_im_inference(
    model: Any,
    image: Image.Image,
    points: Union[List[point_xy], List[dict]] = [],
    point_labels: List[int] = [],
    bboxes: Union[List[bbox_xyxy], List[dict]] = [],
    b64_encode_mask: bool = False,
):
    """returns a list of np masks, each with the shape (h,w) and dtype uint8

    ``model`` is the (Sam2Model, Sam2Processor) tuple from ``load_sam_image_model``.
    Boxes produce one mask per box (in input order); a points prompt produces a single
    mask for the one object the points describe -- matching the previous behavior.
    """
    sam_model, processor = model

    assert (
        points or bboxes
    ), f"SAM2 Image Inference must have either bounding boxes or points. Neither were provided."
    if points:
        assert len(points) == len(
            point_labels
        ), f"{len(points)} points provided but {len(point_labels)} labels given."

    # parse provided bboxes / points into pydantic models, accepting either the
    # dict form ({"x0":..,...} / {"x":..,..}) or the bare-list form ([x0,y0,x1,y1] / [x,y])
    def _to_bbox(b):
        if isinstance(b, bbox_xyxy):
            return b
        if isinstance(b, dict):
            return bbox_xyxy(**b)
        return bbox_xyxy(x0=b[0], y0=b[1], x1=b[2], y1=b[3])

    def _to_point(p):
        if isinstance(p, point_xy):
            return p
        if isinstance(p, dict):
            return point_xy(**p)
        return point_xy(x=p[0], y=p[1])

    bboxes = [_to_bbox(b) for b in bboxes] if bboxes else []
    points = [_to_point(p) for p in points] if points else []

    image = image.convert("RGB")
    device = sam_model.device

    # Build transformers prompt inputs (see the SAM2 docs for the nesting depth):
    #   input_boxes:  (image, num_boxes, 4)            -> one object per box
    #   input_points: (image, num_objects, num_points, 2)
    #   input_labels: (image, num_objects, num_points)
    proc_kwargs = {}
    if bboxes:
        proc_kwargs["input_boxes"] = [[[b.x0, b.y0, b.x1, b.y1] for b in bboxes]]
    if points:
        proc_kwargs["input_points"] = [[[[p.x, p.y] for p in points]]]
        proc_kwargs["input_labels"] = [[list(point_labels)]]

    inputs = processor(images=image, return_tensors="pt", **proc_kwargs).to(device)
    outputs = sam_model(**inputs, multimask_output=False)

    # post_process_masks upscales to the original image size and binarizes.
    # Returns one tensor per image; [0] -> (num_objects, num_masks=1, h, w)
    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"]
    )[0]

    output_masks = [np.asarray(mask).squeeze().astype(np.uint8) for mask in masks]
    return (
        [b64_mask_encode(m).decode("ascii") for m in output_masks]
        if b64_encode_mask
        else output_masks
    )


def unpack_masks(
    masks_generator,
    frame_wh: tuple,
    drop_mask: bool = False,
):
    """return a list of detections in Miro's format given a SAM2 mask generator"""
    w, h = frame_wh
    detections = []
    for frame_idx, tracker_ids, mask_logits in masks_generator:
        masks = (mask_logits > 0.0).cpu().numpy().astype(np.uint8)

        # draw a couple frames for debug purpose
        # if frame_idx % 15 == 0:
        #     ann_masks = [m.squeeze() for m in masks if mask_to_xyxy(m.squeeze())]
        #     if len(ann_masks) > 0:
        #         annotate_masks(
        #             get_pil_im(np.array(vr.get_data(frame_idx))),
        #             masks=ann_masks,
        #         ).save(os.path.join(vframes_dir, f"{frame_idx}.png"))

        for id, mask in zip(tracker_ids, masks):
            mask = mask.squeeze().astype(np.uint8)
            xyxy = mask_to_xyxy(mask)
            if not xyxy:  # mask is empty
                # logger.debug(f"track_id {id} is missing mask at frame {frame_idx}")
                continue
            x0, y0, x1, y1 = xyxy
            det = {  # miro's detections format for videos
                "frame": frame_idx,
                "track_id": id,
                "x": x0 / w,
                "y": y0 / h,
                "w": (x1 - x0) / w,
                "h": (y1 - y0) / h,
                "conf": 1,
            }
            if not drop_mask:
                det["mask_b64"] = b64_mask_encode(mask).decode("ascii")
            detections.append(det)
    return detections


def _propagate_as_tuples(model, processor, session, w, h, **kwargs):
    """Adapt ``propagate_in_video_iterator`` to the (frame_idx, object_ids, mask_logits)
    tuple stream that ``unpack_masks`` expects. Masks are post-processed back to the
    original frame size with ``binarize=False`` so the downstream ``(mask_logits > 0.0)``
    thresholding behaves exactly as before."""
    for out in model.propagate_in_video_iterator(session, **kwargs):
        mask_logits = processor.post_process_masks(
            [out.pred_masks], original_sizes=[[h, w]], binarize=False
        )[0]
        yield out.frame_idx, out.object_ids, mask_logits


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
    ref_frame_idx: int = 0,
    video_load_device: str = "cpu",
):
    sam_model, processor = model

    # put video frames into directory
    # TODO:
    # change frame size
    l_frames_fp = extract_frames(
        video_path,
        fps=sample_fps,
        every_x=every_x,
        overwrite=True,
        im_name_pattern="%05d.jpg",
    )
    vframes_dir = os.path.dirname(l_frames_fp[0])
    vinfo = VidInfo(video_path)
    # cv2 reports dimensions as floats; post_process_masks needs integer sizes
    w = int(vinfo["frame_width"])
    h = int(vinfo["frame_height"])

    # Load the extracted frames for the video session. Filenames carry the frame PTS
    # (frame_pts=1), which exceeds 5 digits on long videos — sort numerically, not
    # lexicographically, or frame 100352 would order before 99999.
    frame_paths = sorted(
        glob.glob(os.path.join(vframes_dir, "*.jpg")),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0]),
    )
    frames = [Image.open(fp).convert("RGB") for fp in frame_paths]

    # bf16 halves frame-storage RAM and matches the app-wide cuda autocast; on CPU there
    # is no autocast, so fp32 is required to match the model weights' dtype.
    session_dtype = torch.bfloat16 if "cuda" in str(device) else torch.float32
    session = processor.init_video_session(
        video=frames,
        inference_device=device,
        # Preprocessing/storing the full video on cuda (the transformers default when only
        # inference_device is set) OOMs on long videos — normalizing 900+ frames needs
        # tens of GB transiently. Keep frames on video_load_device ("cpu" by default);
        # get_frame streams each one to inference_device on access.
        processing_device=video_load_device,
        video_storage_device=video_load_device,
        # Per-frame outputs (incl. per-object 1024^2 high_res_masks) accumulate over the
        # whole video — park them in RAM as well.
        inference_state_device="cpu",
        dtype=session_dtype,
    )
    # Seed all objects in a single call. add_inputs_to_inference_session overwrites
    # session.obj_with_new_inputs on every call, so adding masks one-per-call would leave
    # only the last object registered as "new" -- the earlier objects would then be treated
    # as non-initial-conditioning frames and crash gathering memory that doesn't exist yet.
    processor.add_inputs_to_inference_session(
        inference_session=session,
        frame_idx=ref_frame_idx,
        obj_ids=list(range(len(masks))),
        input_masks=[m for m in masks],
    )
    for mask_idx, mask in enumerate(masks):
        logger.debug(
            f"added mask {mask_idx} of shape {mask.shape} for frame {ref_frame_idx}, xyxy: {mask_to_xyxy(mask)}"
        )

    # seed the reference frame
    sam_model(inference_session=session, frame_idx=ref_frame_idx)

    detections = unpack_masks(
        _propagate_as_tuples(
            sam_model, processor, session, w, h, start_frame_idx=ref_frame_idx
        ),
        drop_mask=drop_mask,
        frame_wh=(w, h),
    )

    if ref_frame_idx != 0:
        logger.debug(f"propagating in reverse now from {ref_frame_idx}")
        # there's no need to reset state
        detections += unpack_masks(
            _propagate_as_tuples(
                sam_model,
                processor,
                session,
                w,
                h,
                start_frame_idx=ref_frame_idx,
                reverse=True,
            ),
            drop_mask=drop_mask,
            frame_wh=(w, h),
        )

    if do_tidy_up:
        # remove vframes_dir
        shutil.rmtree(vframes_dir)

    return detections
