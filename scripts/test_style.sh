expname=flower
CUDA_VISIBLE_DEVICES=$1 python train_style.py \
--config configs/llff_style.txt \
--datadir ./data/nerf_llff_data/flower \
--expname $expname \
--ckpt log_style/$expname/$expname.th \
--style_img ./data/WikiArt/images/landscape/000d655562800587aceb35c35ed4c47cc.jpg \
--render_only 1 \
--render_train 0 \
--render_test 0 \
--render_path 1 \
--chunk_size 1024 \
--rm_weight_mask_thre 0.0001