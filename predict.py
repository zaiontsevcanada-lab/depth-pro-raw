# predict.py
# DepthPro wrapper for Replicate
# Returns: raw 16-bit depth PNG + JSON metadata (focal length, depth range, etc.)

import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from cog import BasePredictor, Input, Path as CogPath
from PIL import Image


class Predictor(BasePredictor):

    def setup(self):
        """
        Load DepthPro model into GPU memory.
        This runs ONCE when the container starts.
        Model weights (~1.5 GB) are downloaded from HuggingFace automatically.
        """
        from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation

        print("Loading DepthPro from HuggingFace...")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = DepthProImageProcessorFast.from_pretrained(
            "apple/DepthPro-hf"
        )
        self.model = DepthProForDepthEstimation.from_pretrained(
            "apple/DepthPro-hf"
        ).to(self.device)
        self.model.eval()

        print(f"DepthPro loaded on {self.device}")

    def predict(
        self,
        image: CogPath = Input(
            description="Input photo (JPG, PNG, WebP)"
        ),
        output_format: str = Input(
            description="raw16 = 16-bit PNG with depth in millimeters (for code). colorized = colored visualization (for humans). both = return all.",
            default="both",
            choices=["raw16", "colorized", "both"],
        ),
        max_depth_meters: float = Input(
            description="Maximum depth range for 16-bit encoding. 65535 pixel value = this many meters. Default 100m covers most outdoor scenes.",
            default=100.0,
            ge=1.0,
            le=1000.0,
        ),
    ) -> list[CogPath]:
        """
        Run DepthPro depth estimation.

        Returns list of files:
        - metadata.json (ALWAYS) — focal_length, fov, depth range, encoding info
        - depth_raw16.png (if raw16 or both) — 16-bit grayscale, each pixel = depth
        - depth_colorized.png (if colorized or both) — turbo colormap visualization
        """

        # ── Load image ──
        img = Image.open(str(image)).convert("RGB")
        img_width, img_height = img.size

        # ── Run DepthPro ──
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # ── Post-process: get depth in meters ──
        post = self.processor.post_process_depth_estimation(
            outputs, target_sizes=[(img_height, img_width)]
        )

        depth_tensor = post[0]["predicted_depth"]       # [H, W] tensor, meters
        depth_np = depth_tensor.cpu().numpy().astype(np.float32)

        # ── Extract focal length & FOV ──
        focal_length = None
        field_of_view = None

        if "field_of_view" in post[0] and post[0]["field_of_view"] is not None:
            fov_val = post[0]["field_of_view"]
            if hasattr(fov_val, 'item'):
                field_of_view = float(fov_val.item())
            else:
                field_of_view = float(fov_val)

        if "focal_length" in post[0] and post[0]["focal_length"] is not None:
            fl_val = post[0]["focal_length"]
            if hasattr(fl_val, 'item'):
                focal_length = float(fl_val.item())
            else:
                focal_length = float(fl_val)

        depth_min = float(depth_np.min())
        depth_max = float(depth_np.max())

        # ── Output directory ──
        out_dir = Path(tempfile.mkdtemp())
        result_files = []

        # ── 1. Metadata JSON (always returned first) ──
        metadata = {
            "image_width": img_width,
            "image_height": img_height,
            "depth_min_meters": round(depth_min, 4),
            "depth_max_meters": round(depth_max, 4),
            "focal_length_pixels": round(focal_length, 2) if focal_length else None,
            "field_of_view_degrees": round(field_of_view, 2) if field_of_view else None,
            "output_format": output_format,
            "max_depth_meters": max_depth_meters,
            "raw16_encoding": {
                "description": "16-bit grayscale PNG. To convert pixel to meters: depth_m = pixel_value / 65535 * max_depth_meters",
                "max_depth_meters": max_depth_meters,
                "formula": "depth_meters = pixel_value / 65535 * {}".format(max_depth_meters),
                "pixel_0": "0 meters (camera position)",
                "pixel_65535": "{} meters (max depth)".format(max_depth_meters),
            },
        }

        meta_path = out_dir / "metadata.json"
        with open(str(meta_path), "w") as f:
            json.dump(metadata, f, indent=2)
        result_files.append(CogPath(meta_path))

        # ── 2. Raw 16-bit depth PNG ──
        if output_format in ("raw16", "both"):
            depth_clipped = np.clip(depth_np, 0, max_depth_meters)
            depth_normalized = depth_clipped / max_depth_meters  # 0..1
            depth_16bit = (depth_normalized * 65535).astype(np.uint16)

            raw_path = out_dir / "depth_raw16.png"
            Image.fromarray(depth_16bit, mode="I;16").save(str(raw_path))
            result_files.append(CogPath(raw_path))

        # ── 3. Colorized visualization ──
        if output_format in ("colorized", "both"):
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.cm as cm

            depth_norm = (depth_np - depth_min) / (depth_max - depth_min + 1e-8)
            # Invert so close = warm, far = cool (more intuitive)
            colored = (cm.turbo(1.0 - depth_norm)[:, :, :3] * 255).astype(np.uint8)

            color_path = out_dir / "depth_colorized.png"
            Image.fromarray(colored).save(str(color_path))
            result_files.append(CogPath(color_path))

        return result_files
