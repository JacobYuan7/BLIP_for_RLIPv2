from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from models.blip import blip_decoder
import json
import numpy as np
from pathlib import Path

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
    image_size = 384
    if dataset == 'coco2017':
        anno_path = ['/mnt/data-nas/peizhi/data/coco2017/annotations/instances_train2017.json',
                     '/mnt/data-nas/peizhi/data/coco2017/annotations/instances_val2017.json']

        with open(anno_path[0], 'r') as f:
            train_bboxes = json.load(f)["annotations"]
        with open(anno_path[1], 'r') as f:
            val_bboxes = json.load(f)["annotations"]
        # dict_keys(['info', 'licenses', 'images', 'annotations', 'categories'])
        bboxes = train_bboxes + val_bboxes

        # convert bboxes to a dict with image_id as keys
        bboxes_dict = {}
        for bbox in bboxes:
            image_id = str(bbox['image_id'])
            if image_id not in bboxes_dict.keys():
                bboxes_dict[image_id] = [bbox,]
            else:
                bboxes_dict[image_id].append(bbox)

        coco2017_folder = Path('/mnt/data-nas/peizhi/data/coco2017')
        val_img_ids = np.unique([str(box['image_id']) for box in val_bboxes])
        img_path_list = []
        for image_id in bboxes_dict.keys():
            img_file_name = str(image_id).zfill(12) + '.jpg'
            if image_id in val_img_ids:
                img_path = coco2017_folder / 'val2017' / img_file_name
            else:
                img_path = coco2017_folder / 'train2017' / img_file_name
            img_path_list.append(img_path)
        
        # print(img_path_list[0], '\n',  img_path_list[-1], '\n', len(img_path_list))

    # Intiailize the model
    # Path /mnt/data-nas/peizhi/params/BLIP
    # File: model_large_caption.pth  model_large.pth
    model = blip_decoder(pretrained= f'/mnt/data-nas/peizhi/params/BLIP/{BLIP_model}.pth', image_size=image_size, vit='large')
    model.eval()
    model = model.to(device)
    captions_dict = {}
    image_id_list = list(bboxes_dict.keys())
    # Split the list
    batch_img_path = []
    batch_image_id = []
    temp_img_path = []
    temp_image_id = []
    for img_path, image_id in zip(img_path_list, image_id_list):
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
                    captions_dict[image_id] = caption[cap_idx::batch_size]
                    # captions_dict[image_id] = caption[cap_idx::len(batch_img_path)]
            # print('caption: ', caption)
        
        if (idx+1)%100 == 0:
            print(f"Finishing processing {idx+1}/{len(batch_img_path)} batch of images.")
    
        # print(captions_dict)

    if save_path:
        with open(save_path, "w") as f:
            json.dump(captions_dict, f)
        print(f"Saving to {save_path}.")


if __name__=="__main__":
    # Do not perform sampling 
    # generate_captions_for_images(
    #     dataset = "coco2017",
    #     BLIP_model = "model_large",
    #     sampling = False,
    #     batch_size = 32,
    #     save_path = '/mnt/data-nas/peizhi/data/coco2017/annotations/BLIP_captions/model_large_beam.json')
    
    # Perform sampling
    # generate_captions_for_images(
    #     dataset = "coco2017",
    #     BLIP_model = "model_large_caption",
    #     sampling = True,
    #     sampling_times = 20,
    #     batch_size = 32,
    #     save_path = '/mnt/data-nas/peizhi/data/coco2017/annotations/BLIP_captions/model_large_caption_nucleus20.json')

