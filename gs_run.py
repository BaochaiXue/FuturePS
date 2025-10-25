#!/usr/bin/env python3
"""
Python equivalent of the original `gs_run.sh`.

It prepares output directories, generates interpolated poses, and
launches training / rendering / video export for every scene found
under `data/gaussian_data/`.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def run_command(cmd: list[str]) -> None:
    """Execute a subprocess command, raising on failure."""
    subprocess.run(cmd, check=True)


def main() -> None:
    root = Path(__file__).resolve().parent
    data_dir = root / "data" / "gaussian_data"
    output_dir = root / "gaussian_output"
    video_dir = root / "gaussian_output_video"
    output_dir.mkdir(exist_ok=True, parents=True)
    video_dir.mkdir(exist_ok=True, parents=True)

    exp_name = "init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"

    # Prepare interpolated camera poses for every scene.
    run_command(["python", str(root / "gaussian_splatting" / "generate_interp_poses.py")])

    # Iterate over every scene directory under data/gaussian_data.
    for scene_path in sorted(data_dir.glob("*/")):
        if not scene_path.is_dir():
            continue

        scene_name = scene_path.name
        print(f"Processing: {scene_name}")

        model_dir = output_dir / scene_name / exp_name
        train_args = [
            "python",
            str(root / "gs_train.py"),
            "-s",
            str(scene_path),
            "-m",
            str(model_dir),
            "--iterations",
            "10000",
            "--lambda_depth",
            "0.001",
            "--lambda_normal",
            "0.0",
            "--lambda_anisotropic",
            "0.0",
            "--lambda_seg",
            "1.0",
            "--use_masks",
            "--isotropic",
            "--gs_init_opt",
            "hybrid",
        ]
        run_command(train_args)

        render_args = [
            "python",
            str(root / "gs_render.py"),
            "-s",
            str(scene_path),
            "-m",
            str(model_dir),
        ]
        run_command(render_args)

        video_args = [
            "python",
            str(root / "gaussian_splatting" / "img2video.py"),
            "--image_folder",
            str(model_dir / "test" / "ours_10000" / "renders"),
            "--video_path",
            str(video_dir / scene_name / f"{exp_name}.mp4"),
        ]
        # Ensure the per-scene video directory exists before exporting.
        (video_dir / scene_name).mkdir(exist_ok=True, parents=True)
        run_command(video_args)


if __name__ == "__main__":
    main()
