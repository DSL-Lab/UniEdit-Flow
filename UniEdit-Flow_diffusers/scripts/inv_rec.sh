# examples using Stable Diffusion 3

# cherry
# unconditional inversion and reconstruction
python demo_inv_rec.py \
    --image_path ./examples/cherry.jpg \
    --save_folder ./outputs/sd3_uncond_cherry \
    --prompt ""

# conditional inversion and reconstruction
python demo_inv_rec.py \
    --image_path ./examples/cherry.jpg \
    --save_folder ./outputs/sd3_cond_cherry \
    --prompt "party in the park under cherry blossoms"

# smoky
# unconditional inversion and reconstruction
python demo_inv_rec.py \
    --image_path ./examples/smoky.jpg \
    --save_folder ./outputs/sd3_uncond_smoky \
    --prompt ""

# conditional inversion and reconstruction
python demo_inv_rec.py \
    --image_path ./examples/smoky.jpg \
    --save_folder ./outputs/sd3_cond_smoky \
    --prompt "for that perfect smoky eye"


# examples using Stable Diffusion XL

# cherry
# unconditional inversion and reconstruction
python demo_inv_rec.py \
    --model_path SG161222/RealVisXL_V4.0 \
    --image_path ./examples/cherry.jpg \
    --save_folder ./outputs/sdxl_uncond_cherry \
    --prompt ""

# conditional inversion and reconstruction
python demo_inv_rec.py \
    --model_path SG161222/RealVisXL_V4.0 \
    --image_path ./examples/cherry.jpg \
    --save_folder ./outputs/sdxl_cond_cherry \
    --prompt "party in the park under cherry blossoms"

# smoky
# unconditional inversion and reconstruction
python demo_inv_rec.py \
    --model_path SG161222/RealVisXL_V4.0 \
    --image_path ./examples/smoky.jpg \
    --save_folder ./outputs/sdxl_uncon_smoky \
    --prompt ""

# conditional inversion and reconstruction
python demo_inv_rec.py \
    --model_path SG161222/RealVisXL_V4.0 \
    --image_path ./examples/smoky.jpg \
    --save_folder ./outputs/sdxl_cond_smoky \
    --prompt "for that perfect smoky eye"

