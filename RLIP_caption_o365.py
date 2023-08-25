from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from models.blip import blip_decoder
import json
import numpy as np
from pathlib import Path
import time
import argparse
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_demo_image(image_size, device):
    img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')   

    w,h = raw_image.size
    display(raw_image.resize((w//5,h//5)))
    
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    image = transform(raw_image).unsqueeze(0).to(device)   
    return image


def load_image(image_size, file_path_list, device):
    # img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
    raw_image_list = []
    for file_path in file_path_list:
        raw_image = Image.open(file_path).convert('RGB')
        raw_image_list.append(raw_image)
        
        # if os.path.exists(file_path):
        #     raw_image = Image.open(file_path).convert('RGB')
        #     raw_image_list.append(raw_image)
        # else:
        #     print(f'{file_path} does not exist.')

    w,h = raw_image.size
    # display(raw_image.resize((w//5,h//5)))
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    # image = transform(raw_image).unsqueeze(0).to(device)
    image = [transform(raw_image).unsqueeze(0).to(device) for raw_image in raw_image_list]
    image = torch.cat(image, dim = 0)
    return image


def generate_captions_for_images(
    dataset,
    BLIP_model,
    args,
    distributed,
    total_segment = 1,
    segment_idx = 1,
    sampling = False,
    sampling_times = 5,
    batch_size = 32,
    save_path = None,
):
    '''
    This function generate captions using pre-trained BLIP models.

    Args:
        dataset (string): the name for the dataset used.
                          Choice: ['coco2017']
        BLIP_model (string): the BLIP model to be used. 
                        Choice: ['model_large', 'model_large_caption']
        args: an argument container
        total_segment (int): for large dataset like Objects365, we could divide the datsets into several segments.
        segment_idx (int): this indicates the index of segment in all segments, starting from 1.
                    For example, if total_segment = 4 and segment_idx = 2,
                    it means that this funtion is generating captions for the 2nd segment of the datasets.
        sampling (bool): whether we do sampling during the caption generation.
                    If True, we are using beam search (Deterministic).
                    If False, we are using nucleus sampling (Stochastic).
        sampling_times (int): the number of times we perform sampling.
                              This is only valid when sampling is True.  
        batch_size (int): the batch size when we generation captions.
        save_path (string): the path to save the generated captions

    Returns:
        None
    '''
    if distributed:
        init_distributed_mode(args)

    start_time = time.time()
    image_size = 384
    if dataset == 'o365':
        anno_path = ['/mnt/data-nas/peizhi/data/Objects365/train/zhiyuan_objv2_train.json',
                     '/mnt/data-nas/peizhi/data/Objects365/val/zhiyuan_objv2_val.json']

        with open(anno_path[0], 'r') as f:
            train_annos = json.load(f)
            print('Finish reading training annotations.')
        with open(anno_path[1], 'r') as f:
            val_annos = json.load(f)
            print('Finish reading validation annotations.')
        print(f"Reading annotations costs {int(time.time() - start_time)} seconds.")

        # dict_keys(['images', 'annotations', 'categories', 'licenses'])
        # example of 'annotations': {'id': 51, 'iscrowd': 0, 'isfake': 0, 'area': 1584.6324365356109, 'isreflected': 0, 'bbox': [491.3955078011, 88.1856689664, 35.65588379410002, 44.442382796800004], 'image_id': 420917, 'category_id': 84}
        # example of 'images': {'height': 512, 'id': 420917, 'license': 5, 'width': 769, 'file_name': 'images/v1/patch8/objects365_v1_00420917.jpg', 'url': ''}
        bboxes = train_annos["annotations"] + val_annos["annotations"]
        # bboxes = bboxes[:10000] # for testing

        # Convert bboxes to a dict with image_id as keys
        start_time = time.time()
        bboxes_dict = {}
        for bbox in bboxes:
            image_id = str(bbox['image_id'])
            if image_id not in bboxes_dict.keys():
                bboxes_dict[image_id] = [bbox,]
            else:
                bboxes_dict[image_id].append(bbox)
        print(f"Converting bbox annotations costs {int(time.time() - start_time)} seconds.")
        print(len(bboxes_dict))

        # Obtain paths for all images
        # Note that we need to exclude invalid images (very few images are not contained in the dataset.).
        start_time = time.time()
        o365_folder = Path('/mnt/data-nas/peizhi/data/Objects365')
        val_img_ids = np.unique([str(img['id']) for img in val_annos['images']])
        id_to_filename = {str(img['id']):img['file_name'].split('/')[-2:] for img in (train_annos["images"] + val_annos["images"])}
        valid_img_path_list = []
        valid_image_id_list = []
        for image_id in bboxes_dict.keys():
            img_file_name = id_to_filename[image_id]
            patch_name, img_name = img_file_name[0], img_file_name[1] 
            if image_id in val_img_ids:
                img_path = o365_folder / 'val' / patch_name / img_name
            else:
                img_path = o365_folder / 'train' / patch_name / img_name

            if os.path.exists(img_path):
                valid_img_path_list.append(img_path)
                valid_image_id_list.append(image_id)
            else:
                print(f'{img_path} does not exist.')
        assert len(valid_img_path_list) == len(valid_image_id_list)
        print(f"Obtaining paths costs {int(time.time() - start_time)} seconds.")
        
        # Clip a segment of images
        start_time = time.time()
        num_images = len(valid_img_path_list) # len(val_annos['images']) + len(train_annos['images'])
        if total_segment > 1:
            assert segment_idx >= 1 and segment_idx <= total_segment
            segment_start = int(num_images*(segment_idx-1)/total_segment)
            segment_end = int(num_images*(segment_idx)/total_segment)
            valid_img_path_list = valid_img_path_list[segment_start:segment_end]
            valid_image_id_list = valid_image_id_list[segment_start:segment_end]
            # bboxes_dict = {j:k for i,(j,k) in enumerate(bboxes_dict.items()) if (i>=segment_start and i<segment_end)}
            print(f'Generate captions for Objects365\'s segment {segment_idx}/{total_segment}, starting from {segment_start} to {segment_end}.')
        print(f"Cliping a segment costs {int(time.time() - start_time)} seconds.")


    # Intiailize the model
    # Path /mnt/data-nas/peizhi/params/BLIP
    # File: model_large_caption.pth  model_large.pth
    model = blip_decoder(pretrained= f'/mnt/data-nas/peizhi/params/BLIP/{BLIP_model}.pth', image_size=image_size, vit='large')
    model = model.to(device)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, 
                                                          device_ids=[args.gpu], 
                                                          find_unused_parameters=True)
                                                          # find_unused_parameters=True) # Setting it True will causing problems in GLIP_attn.

    model.eval()
    captions_dict = {}
    # Split the list
    batch_img_path = []
    batch_image_id = []
    temp_img_path = []
    temp_image_id = []
    for img_path, image_id in zip(valid_img_path_list, valid_image_id_list):
        temp_img_path.append(img_path)
        temp_image_id.append(image_id)
        if len(temp_img_path) == batch_size:
            batch_img_path.append(temp_img_path)
            batch_image_id.append(temp_image_id)
            temp_img_path = []
            temp_image_id = []
    if len(temp_img_path) > 0:
        batch_img_path.append(temp_img_path)
        batch_image_id.append(temp_image_id)  

    # batch_img_path = [batch_img_path[0]] # for testing
    # batch_image_id = [batch_image_id[0]] # for testing
    for idx, (img_path_list, image_id_list) in enumerate(zip(batch_img_path, batch_image_id)):
        image = load_image(image_size=image_size, file_path_list=img_path_list, device=device)
        with torch.no_grad():
            if not sampling:
                # beam search (Deterministic)
                caption = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5) 
                for cap_idx, image_id in enumerate(image_id_list):
                    captions_dict[image_id] = [caption[cap_idx], ]
            else:
                # nucleus sampling (Stochastic)
                caption = []
                for _ in range(sampling_times):
                    caption += model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5) 
                for cap_idx, image_id in enumerate(image_id_list):
                    # captions_dict[image_id] = caption[cap_idx::batch_size]
                    captions_dict[image_id] = caption[cap_idx::len(batch_img_path)]
            # print('caption: ', caption)
        
        if (idx+1)%100 == 0:
            print(f"Finishing processing {idx+1}/{len(batch_img_path)} batch of images.")
        # print(captions_dict)

    if save_path:
        with open(save_path, "w") as f:
            json.dump(captions_dict, f)
        print(f"Saving to {save_path}.")


# def merge_captions_from_segments():
#     1. 
#     # 或许我们可以也分开运行生成scene_graph,分开匹配，分别运行verb tagger，最后merge得到伪标签。
#     2. 
#     # 实现total_segment = 1这一部分的实现

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--total_segment', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--segment_idx', default=1, type=int,
                        help='number of distributed processes')

    return parser

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        # args.gpu = torch.cuda.device_count() - 1
        # print('args.gpu ' + str(args.gpu))
        # print('LOCAL_RANK in os.environ' + str('LOCAL_RANK' not in os.environ))
        # if 'LOCAL_RANK' not in os.environ:
        #     args.gpu = 0
        # else:
        #     args.gpu = int(os.environ['LOCAL_RANK'])

    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

if __name__=="__main__":
    parser = argparse.ArgumentParser('BLIP caption extraction', parents=[get_args_parser()])
    args = parser.parse_args()

    # Perform sampling, Segment 1/4
    generate_captions_for_images(
        dataset = "o365",
        BLIP_model = "model_large_caption",
        args = args,
        distributed = False,
        total_segment = args.total_segment,
        segment_idx = args.segment_idx,
        sampling = True,
        sampling_times = 10,
        batch_size = 32,
        save_path = f'/mnt/data-nas/peizhi/data/Objects365/BLIP_captions/model_large_caption_nucleus10_o365trainval_{args.segment_idx}_{args.total_segment}.json')