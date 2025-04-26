# sd3
python demo_edit.py \
    --alpha 0.6 \
    --omega 5.0 \
    --step 15 \
    --image_path ./examples/rabbit.jpg \
    --save_folder ./outputs/sd3_edit_rabbit \
    --src_prompt "a rabbit sitting in front of colorful eggs" \
    --trg_prompt "a dog with a suit sitting in front of colorful eggs"

python demo_edit.py \
    --alpha 0.6 \
    --omega 5.0 \
    --step 15 \
    --image_path ./examples/koala.jpg \
    --save_folder ./outputs/sd3_edit_koala \
    --src_prompt "a koala is sitting on a tree" \
    --trg_prompt "a koala wearing a hat is sitting on a tree"

python demo_edit.py \
    --alpha 0.6 \
    --omega 5.0 \
    --step 15 \
    --image_path ./examples/cat.jpg \
    --save_folder ./outputs/sd3_edit_cat \
    --src_prompt "a long haired cat looking up at something" \
    --trg_prompt "a short haired cat with blue eyes looking up at something"

python demo_edit.py \
    --alpha 0.6 \
    --omega 5.0 \
    --step 15 \
    --image_path ./examples/flower.jpg \
    --save_folder ./outputs/sd3_edit_flower \
    --src_prompt "a woman with flowers in her hair" \
    --trg_prompt "watercolor of a woman with flowers in her hair"

python demo_edit.py \
    --alpha 0.6 \
    --omega 5.0 \
    --step 15 \
    --image_path ./examples/woman.jpg \
    --save_folder ./outputs/sd3_edit_woman \
    --src_prompt "a woman in sunglasses and leather pants sitting on a bench" \
    --trg_prompt "pixel art of a woman in sunglasses and leather pants sitting on a bench"



# sdxl
python demo_edit.py \
    --alpha 0.6 \
    --omega 5.0 \
    --step 15 \
    --model_path SG161222/RealVisXL_V4.0 \
    --image_path ./examples/rabbit.jpg \
    --save_folder ./outputs/sdxl_edit_rabbit \
    --src_prompt "a rabbit sitting in front of colorful eggs" \
    --trg_prompt "a dog with a suit sitting in front of colorful eggs"

python demo_edit.py \
    --alpha 0.6 \
    --omega 5.0 \
    --step 15 \
    --model_path SG161222/RealVisXL_V4.0 \
    --image_path ./examples/koala.jpg \
    --save_folder ./outputs/sdxl_edit_koala \
    --src_prompt "a koala is sitting on a tree" \
    --trg_prompt "a koala wearing a hat is sitting on a tree"

python demo_edit.py \
    --alpha 0.6 \
    --omega 5.0 \
    --step 15 \
    --model_path SG161222/RealVisXL_V4.0 \
    --image_path ./examples/cat.jpg \
    --save_folder ./outputs/sdxl_edit_cat \
    --src_prompt "a long haired cat looking up at something" \
    --trg_prompt "a short haired cat with blue eyes looking up at something"

python demo_edit.py \
    --alpha 0.6 \
    --omega 5.0 \
    --step 15 \
    --model_path SG161222/RealVisXL_V4.0 \
    --image_path ./examples/flower.jpg \
    --save_folder ./outputs/sdxl_edit_flower \
    --src_prompt "a woman with flowers in her hair" \
    --trg_prompt "watercolor of a woman with flowers in her hair"

python demo_edit.py \
    --alpha 0.6 \
    --omega 5.0 \
    --step 15 \
    --model_path SG161222/RealVisXL_V4.0 \
    --image_path ./examples/woman.jpg \
    --save_folder ./outputs/sdxl_edit_woman \
    --src_prompt "a woman in sunglasses and leather pants sitting on a bench" \
    --trg_prompt "pixel art of a woman in sunglasses and leather pants sitting on a bench"

