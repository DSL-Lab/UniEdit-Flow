# dog + sunglasses
python demo_wan_video_edit.py \
    --alpha 0.8 \
    --omega 5.0 \
    --step 25 \
    --video_path ./examples/video/dog.mp4 \
    --save_folder ./outputs/wan_video_edit_dog \
    --src_prompt "A golden retriever wearing a red harness is walking slowly with its nose close to the dry, leaf-covered ground in a fenced yard next to a bush and a road in the background." \
    --trg_prompt "A golden retriever wearing a red harness and sunglasses is walking slowly with its nose close to the dry, leaf-covered ground in a fenced yard next to a bush and a road in the background."

# hockey to sweeping
python demo_wan_video_edit.py \
    --alpha 0.8 \
    --omega 5.0 \
    --step 25 \
    --video_path ./examples/video/hockey.mp4 \
    --save_folder ./outputs/wan_video_edit_hockey \
    --src_prompt "A man wearing protective gear and rollerblades is playing street hockey on a concrete court, skillfully maneuvering a hockey stick toward a small yellow ball near a portable goal net, with trees and a playground visible in the background." \
    --trg_prompt "A man wearing casual clothes and rollerblades is sweeping the concrete court with a broom, focusing on clearing debris near a small portable net, while trees and a playground are visible in the background."

# surrounding + snow
python demo_wan_video_edit.py \
    --alpha 0.8 \
    --omega 5.0 \
    --step 25 \
    --video_path ./examples/video/car-turn.mp4 \
    --save_folder ./outputs/wan_video_edit_car-turn \
    --src_prompt "A silver SUV is navigating a sharp bend on a scenic mountain road surrounded by lush green meadows, dense pine forests, and towering rocky peaks in the background under a clear sky." \
    --trg_prompt "A silver SUV is navigating a sharp bend on a scenic mountain road blanketed with snow, surrounded by snow-covered meadows, frosted pine forests, and towering, snow-capped rocky peaks under a heavily snowing winter sky."

# koala to cat
python demo_wan_video_edit.py \
    --alpha 0.8 \
    --omega 5.0 \
    --step 25 \
    --video_path ./examples/video/koala.mp4 \
    --save_folder ./outputs/wan_video_edit_koala \
    --src_prompt "A koala with thick gray fur is captured mid-motion as it reaches out with its front paws to climb or move between tree branches, surrounded by lush green leaves and dappled sunlight in a forested area." \
    --trg_prompt "A cat with thick gray fur is captured mid-motion as it stretches out its front paws to climb or leap between tree branches, surrounded by lush green leaves and dappled sunlight in a forested area."

# lucia + hat
python demo_wan_video_edit.py \
    --alpha 0.6 \
    --omega 5.0 \
    --step 30 \
    --video_path ./examples/video/lucia.mp4 \
    --save_folder ./outputs/wan_video_edit_lucia \
    --src_prompt "A woman wearing a sleeveless black dress and white sneakers, with a yellow backpack slung over one shoulder, is walking along a sunlit brick path through a grassy park filled with wildflowers and shaded by leafy green trees." \
    --trg_prompt "A woman wearing a sleeveless black dress, a hat and white sneakers, with a yellow backpack slung over one shoulder, is walking along a sunlit brick path through a grassy park filled with wildflowers and shaded by leafy green trees."

# on a scooter to running
python demo_wan_video_edit.py \
    --alpha 0.8 \
    --omega 5.0 \
    --step 25 \
    --video_path ./examples/video/stunt.mp4 \
    --save_folder ./outputs/wan_video_edit_stunt \
    --src_prompt "A young man wearing a blue shirt, dark pants, and a cap is performing a stunt on a scooter at an urban skate park, balancing on the edge of a graffiti-covered ramp with modern buildings and trees lining the background." \
    --trg_prompt "A young man wearing a blue shirt, dark pants, and a cap is running at an urban skate park, balancing on the edge of a graffiti-covered ramp with modern buildings and trees lining the background."

# bmx bike to motorcycle
python demo_wan_video_edit.py \
    --alpha 0.8 \
    --omega 5.0 \
    --step 25 \
    --video_path ./examples/video/bmx-bumps.mp4 \
    --save_folder ./outputs/wan_video_edit_bmx-bumps \
    --src_prompt "A young rider wearing full protective gear, including a black helmet and motocross-style outfit, is navigating a BMX bike over a series of sandy dirt bumps on a track enclosed by a fence, with a red banner and green field visible in the background." \
    --trg_prompt "A young rider wearing full protective gear, including a black helmet and motocross-style outfit, is navigating a motorcycle over a series of sandy dirt bumps on a track enclosed by a fence, with a red banner and green field visible in the background."

# hike + snow
python demo_wan_video_edit.py \
    --alpha 0.8 \
    --omega 5.0 \
    --step 25 \
    --video_path ./examples/video/hike.mp4 \
    --save_folder ./outputs/wan_video_edit_hike \
    --src_prompt "A man wearing hiking gear and carrying a large backpack is walking along a rocky mountain trail under clear blue skies, surrounded by towering cliffs and patches of green vegetation scattered across the rugged landscape." \
    --trg_prompt "A man wearing hiking gear and carrying a large backpack is walking along a snowy trail under clear blue skies, surrounded by towering icebergs and patches of snow-covered terrain scattered across the frozen landscape."

# soccerball to crystal ball
python demo_wan_video_edit.py \
    --alpha 0.8 \
    --omega 5.0 \
    --step 25 \
    --video_path ./examples/video/soccerball.mp4 \
    --save_folder ./outputs/wan_video_edit_soccerball \
    --src_prompt "A black and white soccer ball is rolling quickly through a grassy garden area, captured in motion blur as it passes between two trees near a wire fence with lush green plants in the background." \
    --trg_prompt "A crystal ball is rolling quickly through a grassy garden area, captured in motion blur as it passes between two trees near a wire fence with lush green plants in the background."

# jump + large rock
python demo_wan_video_edit.py \
    --alpha 0.8 \
    --omega 5.0 \
    --step 25 \
    --video_path ./examples/video/rollerblade.mp4 \
    --save_folder ./outputs/wan_video_edit_rollerblade \
    --src_prompt "A young woman wearing rollerblades, a white crop top, and black shorts is captured mid-air in front of a vibrant pink and blue graffiti wall, with her hair flying upward as she performs a jump on a paved surface bordered by a strip of green grass." \
    --trg_prompt "A young woman wearing rollerblades, a white crop top, and black shorts is captured mid-air in front of a vibrant pink and blue graffiti wall, with her hair flying upward as she jumps over a large rock on a paved surface bordered by a strip of green grass."
