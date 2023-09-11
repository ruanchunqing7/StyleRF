expname=room
CUDA_VISIBLE_DEVICES=$1 python train_style.py \
--config configs/llff_style.txt \
--datadir ./data/nerf_llff_data/room \
--expname $expname \
--ckpt log_style/$expname/$expname.th \
--style_img ./data/WikiArt/images/landscape/0d4bd90794ba2eceb6cbf08570bc6481c.jpg \
--render_only 1 \
--render_train 0 \
--render_test 0 \
--render_path 1 \
--chunk_size 1024 \
--rm_weight_mask_thre 0.0001