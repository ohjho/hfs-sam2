import gradio as gr
import spaces, torch, os, requests, json
from pathlib import Path
from tqdm import tqdm
from samv2_handler import load_sam_image_model, run_sam_im_inference
from PIL import Image
from typing import Union

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
@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
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
                gr.Textbox(
                    label='Bounding Boxes (JSON list of dicts: [{"x0":..., "y0":..., "x1":..., "y1":...}, ...])',
                ),
                gr.Textbox(
                    label='Points (JSON list of dicts: [{"x":..., "y":...}, ...])',
                ),
                gr.Textbox(
                    label="Points Label (JSON list of integar)",
                ),
            ],
            outputs=gr.JSON(label="Output JSON"),
            title="SAM2 for Images",
        )
# Download checkpoints before launching the app
download_checkpoints()
demo.launch(
    mcp_server=True, app_kwargs={"docs_url": "/docs"}  # add FastAPI Swagger API Docs
)
