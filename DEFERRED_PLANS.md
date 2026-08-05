# Deferred plans

Designs that were worked out and verified but deliberately not implemented yet.

## Implement `async_frame_load` for real

`async_frame_load` on the `process_video` endpoint is currently a **documented no-op**: the
parameter is accepted by `app.py` and `samv2_handler.run_sam_video_inference` but never used.
It is a vestige of the old `samv2` package's `async_loading_frames`, lost in the transformers
rewrite (commit `0f11cd9`). The design below implements it as a producer/consumer overlap of
CPU frame preprocessing with GPU propagation; it was verified against the installed
transformers 5.9.0 source (`transformers/models/sam2_video/`) in Aug 2026 — re-verify the
cited internals if transformers has been upgraded since.

**Value:** overlaps ~10–20 s of CPU preprocessing (for a ~900-frame video) with GPU
propagation inside the billed ZeroGPU window, and avoids the eager path's transient ~11 GB
fp32 CPU stack (frames get preprocessed per-chunk instead of all at once).

### Design

- **Producer thread** preprocesses JPEG chunks (`CHUNK = 8`) on CPU via
  `processor.video_processor(videos=pils, device="cpu", return_tensors="pt").pixel_values_videos[0]`
  and appends each frame via `session.add_new_frame(frame, idx)` — the session class is
  designed for streaming append (casts to session dtype, stores on `video_storage_device`,
  GIL-atomic dict insert). A `threading.Condition` guards a `_loaded` counter and an
  `_error` slot; the consumer calls `wait_for(frame_idx, timeout_s=90)`, which blocks until
  the frame is available, raises `TimeoutError` on a stalled producer, and re-raises
  producer exceptions as `RuntimeError from original`.
- **Consumer**: `propagate_in_video_iterator` is unusable — it reads
  `inference_session.num_frames` once at iteration start to build a fixed
  `processing_order`. Instead run a manual loop calling
  `model(inference_session=session, frame_idx=i, reverse=...)` per frame (exactly what the
  iterator does internally). Forward order `range(start_frame_idx, num_total)`; reverse
  `range(start_frame_idx, -1, -1)` if `start_frame_idx > 0` else empty. Total frame count is
  known upfront from the extracted JPEG file list.
- **Numeric-parity gate**: the only `num_frames`-sensitive numeric in `forward` (with
  `streaming=False`) is `min(num_total_frames, config.max_object_pointers_in_encoder=16)`
  in `_get_object_pointers` (it also normalizes the object pointers' temporal sine PE).
  Gate the consumer on `len(processed_frames) >= min(N, 16)` before the first forward call
  (`wait_for` uses `target = max(frame_idx + 1, min_ready)`) → bit-identical results to the
  eager path.
- **Streaming session setup**: `init_video_session()` with no `video` arg leaves
  `session.video_height/video_width = None`; set them manually after init (plain
  attributes). Mask seeding (`add_inputs_to_inference_session(input_masks=...)` + the
  `_use_mask_as_output` forward) needs only frame `ref_frame_idx` to be present — call
  `loader.wait_for(ref_frame_idx)` before the seeding forward.
- **Bidirectional tracking**: the forward pass ends at `num_total - 1`, which forces the
  producer to have finished, so the reverse pass (nonzero `ref_frame_idx`) always runs
  against a complete session.
- **Thread-locality**: `torch.inference_mode()` / `torch.autocast` decorators are
  thread-local. The producer does plain CPU preprocessing (needs neither); ALL model forward
  calls must stay on the main thread.
- **Cleanup**: wrap propagation in `try/finally`: `loader.join()` + move
  `shutil.rmtree(vframes_dir)` into the `finally` (today an exception leaks the frames dir —
  worth fixing even without async).
- Keep the eager path as the `async_frame_load=False` branch. When implemented, update the
  `process_video` docstring (currently says "reserved; currently has no effect") — it is the
  MCP tool schema, so load the `gradio-mcp-tool-docstrings` skill first.
