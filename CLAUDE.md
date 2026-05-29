# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Hugging Face Space that serves SAM2 (Segment Anything 2) image and video segmentation. It runs on **ZeroGPU** via the Gradio SDK, and also exposes its functions as an **MCP server** and a **FastAPI/Swagger API** (`/docs`). It originated as a fork of [SkalskiP/florence-sam](https://huggingface.co/spaces/SkalskiP/florence-sam).

SAM2 is provided by HuggingFace `transformers` (`Sam2Model`/`Sam2Processor` for images, `Sam2VideoModel`/`Sam2VideoProcessor` for video), not the standalone `sam2`/`samv2` package.

The Space config lives in the YAML frontmatter of `README.md` (`sdk: gradio`, `app_file: app.py`).

## Running

```bash
pip install -r requirements.txt
python app.py            # launches Gradio UI + MCP server + FastAPI docs at /docs
```

Model weights are downloaded from the HuggingFace Hub on first use and cached by `transformers` — there is no manual checkpoint download step. The four variants (`tiny`, `small`, `base_plus`, `large`) map to SAM2.1 Hub repo IDs (`facebook/sam2.1-hiera-*`) in `samv2_handler.variant_hf_mapping`.

`ffmpeg_extractor.py` is also a standalone Typer CLI:

```bash
python ffmpeg_extractor.py extract-frames <video_path> --fps 5
python ffmpeg_extractor.py extract-keyframes <video_path> --threshold 0.3
python ffmpeg_extractor.py extract-keyframes-greedy <video_path>
```

There are currently **no tests** in the repo despite `pytest` being in `requirements.txt`.

## Architecture

The flow is layered: `app.py` (Gradio/MCP/API surface) → `samv2_handler.py` (SAM2 orchestration) → `ffmpeg_extractor.py` / `toolbox/` (media + encoding utilities).

- **`app.py`** — defines the two public endpoints `process_image` and `process_video` as Gradio Interfaces (`api_name=`). It handles input coercion (JSON strings → lists, base64 string cleanup) and validation, then delegates to the handler. Models are loaded lazily per-request via `load_im_model` / `load_vid_model`.

- **`samv2_handler.py`** — the SAM2 wrapper. The `load_sam_*_model` functions each return a `(model, processor)` tuple. `run_sam_im_inference` runs prompted segmentation (boxes and/or points) via the `Sam2Processor` + `Sam2Model` and `post_process_masks`. `run_sam_video_inference` extracts frames to a temp dir, loads them into a `Sam2VideoProcessor.init_video_session`, seeds the reference frame with input masks via `add_inputs_to_inference_session(input_masks=...)`, then iterates `propagate_in_video_iterator` to track objects. `unpack_masks` converts per-frame mask logits into the output detection format; a small `_propagate_as_tuples` adapter feeds it `(frame_idx, object_ids, mask_logits)` tuples (post-processed to original frame size, un-binarized).

- **`toolbox/mask_encoding.py`** — masks are exchanged as **base64-encoded 1-bit PNG strings**. `b64_mask_encode` writes a temp PNG and returns base64 bytes; `b64_mask_decode` reverses it. This is the canonical mask interchange format across image/video endpoints.

- **`toolbox/vid_utils.py`** — `VidInfo` (cv2 probe, used by video inference for frame width/height) and `VidReader` (imageio reader); plus general-purpose video read/write/slice/resize helpers.

- **`visualizer.py`** — `mask_to_xyxy` (mask → bounding box) is the only actively-used function. The `annotate_*` functions reference modules (`mcolors`, `im_draw_bbox`, `im_color_mask`) that are not imported and are effectively dead/debug-only code.

### ZeroGPU + SAM2 critical conventions

These are load-bearing and easy to break:

- Every GPU function is decorated with `@spaces.GPU`. `process_video` uses `@spaces.GPU(duration=120)` — the caller must have ≥2 min of ZeroGPU time available.
- `process_image` / `process_video` are wrapped with `@torch.inference_mode()` and `@torch.autocast(device_type="cuda", dtype=torch.bfloat16)`. The autocast context is also entered at module top in `app.py`. **This is required** to avoid the SAM2 `RuntimeError: No available kernel. Aborting execution.` error (see README changelog).
- `multimask_output=False` in `run_sam_im_inference` — SAM2 returns 3 masks per object when `True`; the code intentionally returns one mask per prompt (one per box, or a single mask for a points-only prompt).

### Output detection format ("Miro's format")

Video inference returns a list of per-frame detections with **normalized** coordinates:

```python
{"frame": int, "track_id": int, "x": x0/w, "y": y0/h, "w": (x1-x0)/w, "h": (y1-y0)/h, "conf": 1, "mask_b64": "..."}
```

`mask_b64` is omitted when `drop_masks=True`. Empty masks (no object in that frame) are skipped.

### Reference frame & bidirectional tracking

`ref_frame_idx` (default 0) is the frame the input masks correspond to. When it's nonzero, video inference propagates forward *and* then again with `reverse=True` from that frame, so objects are tracked in both directions without resetting state.
