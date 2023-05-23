python custom_inpainting/generate.py \
--input_images_dir /media/data/vv/tasks/2023_05_11_inpainting/images \
--result_images_dir /media/data/vv/tasks/2023_04_08_css_heads_inpainting/result/2023_05_02/inpainted7/ \
--config_path configs/stable-diffusion/v2-inpainting-inference.yaml \
--weights_path /media/data/vv/tasks/2023_02_14_stable_diffusion_weights/weights/512-inpainting-ema.ckpt \
--generation_limit 100000 \
--base_prompt "small head asian man refugee in the underground looks back, amateur photo, assembly line, Stock Footage" \
--logs_file_path /media/data/vv/tasks/2022_12_05_debug_stable_diffusion2/head.log \
--context_bbox_size 160 \
--coco_ann_path /media/data/vv/tasks/2023_05_11_inpainting/heads_coco.json \
--coco_bbox_padding 20 

# --base_prompt "small head asian man at the garbage dump at night in fog, amateur photo, looks back, BAD, POOR quality" \f
# --base_prompt "small head asian man criminal theft is caught surfeillance camera footage, captive refugee in the underground, breaking news, looks back, amateur photo" \