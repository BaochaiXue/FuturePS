output_dir="./gaussian_output"
output_video_dir="./gaussian_output_video"
mkdir -p "${output_dir}" "${output_video_dir}"

exp_name="init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"

python ./gaussian_splatting/generate_interp_poses.py

# Iterate over every scene folder under data/gaussian_data
for scene_path in ./data/gaussian_data/*/; do
    if [[ ! -d "${scene_path}" ]]; then
        continue
    fi
    scene_name="$(basename "${scene_path%/}")"
    echo "Processing: $scene_name"

    # Training
    python gs_train.py \
        -s "${scene_path%/}" \
        -m ${output_dir}/${scene_name}/${exp_name} \
        --iterations 10000 \
        --lambda_depth 0.001 \
        --lambda_normal 0.0 \
        --lambda_anisotropic 0.0 \
        --lambda_seg 1.0 \
        --use_masks \
        --isotropic \
        --gs_init_opt 'hybrid'

    # Rendering
    python gs_render.py \
        -s "${scene_path%/}" \
        -m ${output_dir}/${scene_name}/${exp_name} \

    # Convert images to video
    python gaussian_splatting/img2video.py \
        --image_folder ${output_dir}/${scene_name}/${exp_name}/test/ours_10000/renders \
        --video_path ${output_video_dir}/${scene_name}/${exp_name}.mp4
done
