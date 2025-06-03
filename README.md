---
title: Sam2
emoji: 📚
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 5.32.0
app_file: app.py
pinned: false
short_description: sam2 images and video inference on ZeroGPU
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Changelog

this huggingspace demo borrows extensively from [SkalskiP/florence-sam](https://huggingface.co/spaces/SkalskiP/florence-sam/tree/main).
Here are some changes that's noteworthy:

* the `RuntimeError: No available kernel. Aborting execution.` error documented [here](https://github.com/facebookresearch/sam2/issues/48), [here](https://huggingface.co/spaces/SkalskiP/florence-sam/discussions/5), and [here](https://github.com/facebookresearch/sam2/issues/100) is fixed with the following lines in `app.py` (see [this commit](https://huggingface.co/spaces/SkalskiP/florence-sam/commit/488d99ebc8b60d6a87b84a545dc0561cfadefc65)) with the `@torch.inference_mode()` and
`@torch.autocast(device_type="cuda", dtype=torch.bfloat16)` before
the inference function:
```python
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
```
