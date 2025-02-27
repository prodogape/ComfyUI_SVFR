# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import warnings
import os
import numpy as np
import safetensors.torch
import torch
import torch.utils.checkpoint
from PIL import Image
from diffusers import AutoencoderKLTemporalDecoder
from diffusers.schedulers import EulerDiscreteScheduler
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.utils import save_image
import random
import cv2
import safetensors 

import gc
# pipeline 
from .src.pipelines.pipeline import LQ2VideoLongSVDPipeline
from .src.utils.util import save_videos_grid, seed_everything
from .src.models.id_proj import IDProjConvModel
from .src.models import model_insightface_360k
from .src.dataset.face_align.align import AlignImage
from .src.models.svfr_adapter.unet_3d_svd_condition_ip import UNet3DConditionSVDModel
from .src.dataset.dataset import get_affine_transform, mean_face_lm5p_256,get_union_bbox, process_bbox, crop_resize_img
from .node_utils import tensor2pil
warnings.filterwarnings("ignore")


def main_loader(weight_dtype, repo,UNET,VAE, unet_path, det_path, id_path, face_path,device,dtype):

   
    val_noise_scheduler = EulerDiscreteScheduler.from_pretrained(repo, subfolder="scheduler")
   
    input_unet_dic=safetensors.torch.load_file(UNET)
    unet_config=UNet3DConditionSVDModel.load_config(os.path.join(repo, "unet"))
    unet=UNet3DConditionSVDModel.from_config(unet_config).to(weight_dtype)
    unet.load_state_dict(input_unet_dic, strict=False)


    align_instance = AlignImage(device, det_path=det_path)
    
    input_vae_dic=safetensors.torch.load_file(VAE)
    vae_config=AutoencoderKLTemporalDecoder.load_config(os.path.join(repo, "vae"))
    vae=AutoencoderKLTemporalDecoder.from_config(vae_config).to(weight_dtype)
    vae.load_state_dict(input_vae_dic, strict=False)

    import torch.nn as nn
    class InflatedConv3d(nn.Conv2d):
        def forward(self, x):
            x = super().forward(x)
            return x
    
    # Add ref channel
    old_weights = unet.conv_in.weight
    old_bias = unet.conv_in.bias
    new_conv1 = InflatedConv3d(
        12,
        old_weights.shape[0],
        kernel_size=unet.conv_in.kernel_size,
        stride=unet.conv_in.stride,
        padding=unet.conv_in.padding,
        bias=True if old_bias is not None else False,
    )
    param = torch.zeros((320, 4, 3, 3), requires_grad=True)
    new_conv1.weight = torch.nn.Parameter(torch.cat((old_weights, param), dim=1))
    if old_bias is not None:
        new_conv1.bias = old_bias
    unet.conv_in = new_conv1
    unet.config["in_channels"] = 12
    unet.config.in_channels = 12
    
    id_linear = IDProjConvModel(in_channels=512, out_channels=1024).to(device=device)
    
    pre_unet_dict=torch.load(unet_path, map_location="cpu")
    pre_linear_dict=torch.load(id_path, map_location="cpu")

    # load pretrained weights
    unet.load_state_dict(pre_unet_dict,strict=True,)
    
    id_linear.load_state_dict(pre_linear_dict,strict=True,)
    
    net_arcface = model_insightface_360k.getarcface(face_path).eval().to(device=device)
    
    #image_encoder.to(weight_dtype)
    vae.to(weight_dtype)
    unet.to(weight_dtype)
    id_linear.to(weight_dtype)
    net_arcface.requires_grad_(False).to(weight_dtype)
    del input_unet_dic, input_vae_dic, pre_unet_dict, pre_linear_dict
    gc.collect()
    torch.cuda.empty_cache()
    pipe = LQ2VideoLongSVDPipeline(
        unet=unet,
        #image_encoder=image_encoder,
        vae=vae,
        scheduler=val_noise_scheduler,
        feature_extractor=None,
        )
    pipe = pipe.to(device, dtype=unet.dtype)
    
    return pipe, id_linear, net_arcface, align_instance


def main_sampler(pipe,align_instance, net_arcface, id_linear, save_dir, weight_dtype, seed, input_frames_pil, task_ids,
                 mask_array,
                 save_video, decode_chunk_size, noise_aug_strength, min_appearance_guidance_scale,
                 max_appearance_guidance_scale,
                 overlap, i2i_noise_strength, steps, n_sample_frames,device,crop_face_region):
    

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    seed_everything(seed)
    
    if 2 in task_ids and isinstance(mask_array, np.ndarray):
        white_positions = mask_array == 255
    
    print('task_ids:', task_ids)
    task_prompt = [0, 0, 0]
    for i in range(3):
        if i in task_ids:
            task_prompt[i] = 1
    print("task_prompt:", task_prompt)
    
    files_prefix = ''.join(random.choice("0123456789") for _ in range(5))
    video_name = f"infer_{files_prefix}"
    # print(video_name)
    
    if os.path.exists(os.path.join(save_dir, "result_frames", video_name[:-4])):
        print(os.path.join(save_dir, "result_frames", video_name[:-4]))
        # continue
    
    #import decord
    #cap = decord.VideoReader(input_frames_pil, fault_tol=1)
    
    total_frames = len(input_frames_pil)
    T = total_frames  #
    print("total_frames:", total_frames)
    step = 1
    drive_idx_start = 0
    drive_idx_list = list(range(drive_idx_start, drive_idx_start + T * step, step))
    assert len(drive_idx_list) == T
    
    if crop_face_region:
        # Crop faces from the video for further processing
        bbox_list = []
        frame_interval = 5
        for frame_count, drive_idx in enumerate(drive_idx_list):
            if frame_count % frame_interval != 0:
                continue  
            frame = np.array(input_frames_pil[drive_idx])
            _, _, bboxes_list = align_instance(frame[:,:,[2,1,0]], maxface=True)
            if bboxes_list==[]:
                continue
            x1, y1, ww, hh = bboxes_list[0]
            x2, y2 = x1 + ww, y1 + hh
            bbox = [x1, y1, x2, y2]
            bbox_list.append(bbox)
        bbox = get_union_bbox(bbox_list)
        bbox_s = process_bbox(bbox, expand_radio=0.4, height=frame.shape[0], width=frame.shape[1])


    imSameIDs = []
    vid_gt = []
    width,height=input_frames_pil[0].size
    for i, drive_idx in enumerate(drive_idx_list):
        imSameID = input_frames_pil[drive_idx]
        #imSameID = Image.fromarray(frame)
        if crop_face_region:
            imSameID = crop_resize_img(imSameID, bbox_s)
            imSameID = imSameID.resize((width,height))
        if 1 in task_ids:
            imSameID = imSameID.convert("L")  # Convert to grayscale
            imSameID = imSameID.convert("RGB")
        image_array = np.array(imSameID)
        if 2 in task_ids and isinstance(mask_array, np.ndarray):
            image_array[white_positions] = [255, 255, 255]  # mask for inpainting task
        vid_gt.append(np.float32(image_array / 255.))
        imSameIDs.append(imSameID)
    
    vid_lq = [(torch.from_numpy(frame).permute(2, 0, 1) - 0.5) / 0.5 for frame in vid_gt]  # torch.Size([3, 512, 512])
    
    val_data = dict(
        pixel_values_vid_lq=torch.stack(vid_lq, dim=0),
        # pixel_values_ref_img=self.to_tensor(target_image),
        # pixel_values_ref_concat_img=self.to_tensor(imSrc2),
        task_ids=task_ids,
        task_id_input=torch.tensor(task_prompt),
        total_frames=total_frames,
    )
    
    window_overlap = 0
    inter_frame_list = get_overlap_slide_window_indices(val_data["total_frames"], n_sample_frames, window_overlap)
    
    lq_frames = val_data["pixel_values_vid_lq"]
    task_ids = val_data["task_ids"]
    task_id_input = val_data["task_id_input"]
    height, width = val_data["pixel_values_vid_lq"].shape[-2:]
  
    print("Generating the first clip...")
    output = pipe(
        lq_frames[inter_frame_list[0]].to(device).to(weight_dtype),  # lq
        None,  # ref concat
        torch.zeros((1, len(inter_frame_list[0]), 49, 1024)).to(device).to(weight_dtype),  # encoder_hidden_states
        task_id_input.to(device).to(weight_dtype),
        height=height,
        width=width,
        num_frames=len(inter_frame_list[0]),
        decode_chunk_size=decode_chunk_size,
        noise_aug_strength=noise_aug_strength,
        min_guidance_scale=min_appearance_guidance_scale,
        max_guidance_scale=max_appearance_guidance_scale,
        overlap=overlap,
        frames_per_batch=len(inter_frame_list[0]),
        num_inference_steps=steps,
        i2i_noise_strength=i2i_noise_strength,
    )
    video = output.frames
    ref_img_tensor = video[0][:, -1]

    ref_img = (video[0][:, -1] * 0.5 + 0.5).clamp(0, 1) * 255.
    ref_img = ref_img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    pts5 = align_instance(ref_img[:, :, [2, 1, 0]], maxface=True)[0][0]
    
    # 修改：根据 crop_face_region 的值调整变换矩阵
    if crop_face_region:
        warp_mat = get_affine_transform(pts5, mean_face_lm5p_256 * height / 256)
        ref_img = cv2.warpAffine(np.array(Image.fromarray(ref_img)), warp_mat, (height, width), flags=cv2.INTER_CUBIC)
    else:
        # 不裁剪时，使用更小的变换范围以保持更多原始细节
        target_pts = mean_face_lm5p_256 * height / 256
        # 计算中心点
        src_center = np.mean(pts5, axis=0)
        target_center = np.mean(target_pts, axis=0)
        # 仅进行轻微的对齐调整
        warp_mat = cv2.getAffineTransform(
            np.float32([pts5[0], pts5[1], pts5[2]]),
            np.float32([target_pts[0], target_pts[1], target_pts[2]])
        )
        # 调整变换矩阵以减少变形程度
        warp_mat = warp_mat * 0.3 + np.array([[1, 0, 0], [0, 1, 0]]) * 0.7
        ref_img = cv2.warpAffine(np.array(Image.fromarray(ref_img)), warp_mat, (height, width), flags=cv2.INTER_CUBIC)
    
    ref_img = to_tensor(ref_img).to(device).to(weight_dtype)
    
    save_image(ref_img * 0.5 + 0.5, f"{save_dir}/ref_img_align.png")
    
    ref_img = F.interpolate(ref_img.unsqueeze(0)[:, :, 0:224, 16:240], size=[112, 112], mode='bilinear')
    _, id_feature_conv = net_arcface(ref_img)
    id_embedding = id_linear(id_feature_conv)
    
    print('Generating all video clips...')
    video = pipe(
        lq_frames.to(device).to(weight_dtype),  # lq
        ref_img_tensor.to(device).to(weight_dtype),
        id_embedding.unsqueeze(1).repeat(1, len(lq_frames), 1, 1).to(device).to(weight_dtype),  # encoder_hidden_states
        task_id_input.to(device).to(weight_dtype),
        height=height,
        width=width,
        num_frames=val_data["total_frames"],  # frame_num,
        decode_chunk_size=decode_chunk_size,
        noise_aug_strength=noise_aug_strength,
        min_guidance_scale=min_appearance_guidance_scale,
        max_guidance_scale=max_appearance_guidance_scale,
        overlap=overlap,
        frames_per_batch=n_sample_frames,
        num_inference_steps=steps,
        i2i_noise_strength=i2i_noise_strength,
    ).frames


    video = (video * 0.5 + 0.5).clamp(0, 1)
    video = torch.cat([video.to(device=device)], dim=0).cpu()  # torch.Size([1, 3, 160, 512, 512])

   
    if save_video:
        save_videos_grid(video, f"{save_dir}/{video_name[:-4]}_{seed}.mp4", n_rows=1, fps=25)
    
    # if restore_frames:
    #     video = video.squeeze(0)
    #     os.makedirs(os.path.join(save_dir, "result_frames", f"{video_name[:-4]}_{seed}"),exist_ok=True)
    #     print(os.path.join(save_dir, "result_frames", video_name[:-4]))
    #     for i in range(video.shape[1]):
    #         save_frames_path = os.path.join(f"{save_dir}/result_frames", f"{video_name[:-4]}_{seed}", f'{i:08d}.png')
    #         save_image(video[:,i], save_frames_path)
   
    return video.squeeze(0).permute(1, 2, 3, 0)  # bcthw to B,H,W,C


def get_overlap_slide_window_indices(video_length, window_size, window_overlap):
    inter_frame_list = []
    for j in range(0, video_length, window_size - window_overlap):
        inter_frame_list.append([e % video_length for e in range(j, min(j + window_size, video_length))])
    
    return inter_frame_list
