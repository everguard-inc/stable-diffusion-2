python generate.py \
--input_images_dir /media/data/vv/tasks/2023_07_30_new_seah_for_inpainting/img_with_bbox \
--result_images_dir /media/data/vv/tasks/2023_07_30_heads_inpainting/result/2023_08_09/inpainted_sdxl_for_filtering/ \
--generation_limit 100000 \
--base_prompt "korean man at night in the underground looks back" \
--logs_file_path /media/data/vv/tasks/2022_12_05_debug_stable_diffusion2/head.log \
--context_bbox_size 192 \
--coco_ann_path /media/data/vv/inference_results/Prediction_yolov5m_2023-07-31-09-48-11_dataset_heads-gunsan_coco_3x736x736_5045e499_2023-08-01-01-03-25/detections_coco_with_bbox.json \
--coco_bbox_padding 20  \
--num_inference_steps 30