# # python -m torch.distributed.launch --nproc_per_node=4 --use_env RLIP_caption_o365.py
# pip install -I transformers==4.5.1 --no-cache-dir --force-reinstall
# pip install -r requirements_ParSeDETRHOI.txt;
# pip install submitit==1.3.0;
# pip install timm;
python RLIP_caption_o365.py \
    --segment_idx 1 \
    --total_segment 4 \