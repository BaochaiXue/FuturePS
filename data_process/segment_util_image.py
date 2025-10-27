import cv2
import torch
import numpy as np
import time
import logging
import traceback
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from groundingdino.util.inference import load_model, load_image, predict
from argparse import ArgumentParser

"""
Hyper parameters
"""

parser = ArgumentParser()
parser.add_argument(
    "--img_path",
    type=str,
)
parser.add_argument("--output_path", type=str)
parser.add_argument("--TEXT_PROMPT", type=str)
args = parser.parse_args()

img_path = args.img_path
output_path = args.output_path
TEXT_PROMPT = args.TEXT_PROMPT

MAX_RETRIES = 5
RETRY_DELAY_SECONDS = 3.0
logging.basicConfig(
    level=logging.INFO,
    format="[segment_util_image] %(levelname)s: %(message)s",
)


def run_segmentation() -> None:
    SAM2_CHECKPOINT = "./data_process/groundedSAM_checkpoints/sam2.1_hiera_large.pt"
    SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
    GROUNDING_DINO_CONFIG = (
        "./data_process/groundedSAM_checkpoints/GroundingDINO_SwinT_OGC.py"
    )
    GROUNDING_DINO_CHECKPOINT = (
        "./data_process/groundedSAM_checkpoints/groundingdino_swint_ogc.pth"
    )
    BOX_THRESHOLD = 0.35
    TEXT_THRESHOLD = 0.25
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # build SAM2 image predictor
    sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    # build grounding dino model
    grounding_model = load_model(
        model_config_path=GROUNDING_DINO_CONFIG,
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
        device=DEVICE,
    )

    # setup the input image and text prompt for SAM 2 and Grounding DINO
    text = TEXT_PROMPT

    image_source, image = load_image(img_path)

    sam2_predictor.set_image(image_source)

    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=text,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    )

    # process the box prompt for SAM 2
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    conf_values = (
        confidences.detach().cpu().numpy().tolist()
        if hasattr(confidences, "detach")
        else confidences
    )
    print(
        f"[GroundingDINO Debug] boxes shape={input_boxes.shape}, confidences={conf_values}"
    )

    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    with torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()
    ):
        masks, scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

    if masks.ndim == 4:
        masks = masks.squeeze(1)

    confidences = confidences.numpy().tolist()
    class_names = labels

    OBJECTS = class_names

    ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS)}

    print(f"Detected {len(masks)} objects")

    raw_img = cv2.imread(img_path)
    mask_img = (masks[0] * 255).astype(np.uint8)

    ref_img = np.zeros((h, w, 4), dtype=np.uint8)
    mask_bool = mask_img > 0
    ref_img[mask_bool, :3] = raw_img[mask_bool]
    ref_img[:, :, 3] = mask_bool.astype(np.uint8) * 255
    cv2.imwrite(output_path, ref_img)
    logging.info("Saved mask to %s", output_path)


def main() -> None:
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        if torch.cuda.is_available():
            try:
                stats = torch.cuda.memory_stats()
                allocated_mb = stats["allocated_bytes.all.current"] / (1024**2)
                reserved_mb = stats["reserved_bytes.all.current"] / (1024**2)
                logging.info(
                    "Attempt %d/%d: GPU memory allocated %.2f MB, reserved %.2f MB",
                    attempt,
                    MAX_RETRIES,
                    allocated_mb,
                    reserved_mb,
                )
            except Exception:  # pragma: no cover - stats may fail on some devices
                logging.debug("Unable to query torch.cuda.memory_stats()", exc_info=True)
        else:
            logging.info("Attempt %d/%d: CUDA unavailable, running on CPU", attempt, MAX_RETRIES)
        try:
            run_segmentation()
            return
        except Exception as exc:  # pylint: disable=broad-except
            last_exc = exc
            logging.error(
                "Segmentation attempt %d/%d failed for %s: %s",
                attempt,
                MAX_RETRIES,
                img_path,
                exc,
            )
            logging.debug("Stack trace:\n%s", traceback.format_exc())
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)
    raise SystemExit(
        f"Segmentation failed after {MAX_RETRIES} attempts; last error: {last_exc}"
    ) from last_exc


if __name__ == "__main__":
    main()
