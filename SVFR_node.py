# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import os
import torch
from .infer import main_loader,main_sampler
from .node_utils import nomarl_upscale,tensor_upscale
from .src.dataset.dataset import get_union_bbox, process_bbox, mean_face_lm5p_256
import folder_paths
import cv2

MAX_SEED = np.iinfo(np.int32).max
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
current_path = os.path.dirname(os.path.abspath(__file__))

weigths_SVFR_current_path = os.path.join(folder_paths.models_dir, "SVFR")
if not os.path.exists(weigths_SVFR_current_path):
    os.makedirs(weigths_SVFR_current_path)

folder_paths.add_model_folder_path("SVFR", weigths_SVFR_current_path)


class SVFR_LoadModel:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        yolo_ckpt_list = [i for i in folder_paths.get_filename_list("SVFR") if
                          "yolo" in i]
        insightface_ckpt_list = [i for i in folder_paths.get_filename_list("SVFR") if
                          "insightface" in i]
        unet_ckpt_list = [i for i in folder_paths.get_filename_list("SVFR") if
                                 "unet" in i]
        id_ckpt_list = [i for i in folder_paths.get_filename_list("SVFR") if
                                 "id" in i]
        return {
            "required": {
                "checkpoints": (folder_paths.get_filename_list("checkpoints"),),
                "vae": (folder_paths.get_filename_list("vae"),),
                "unet": (["none"] + unet_ckpt_list,),
                "yolo_ckpt": (["none"] + yolo_ckpt_list,),
                "id_ckpt": (["none"] + id_ckpt_list,),
                "insightface": (["none"] + insightface_ckpt_list,),
                "dtype": (["fp16","bf16","fp32"],),
            }
        }
    
    RETURN_TYPES = ("MODEL_SVFR",)
    RETURN_NAMES = ("model",)
    FUNCTION = "main_loader"
    CATEGORY = "SVFR"
    
    def main_loader(self,checkpoints,vae, unet, yolo_ckpt, id_ckpt, insightface,dtype):
        
        I2V_repo=os.path.join(current_path, "svd_repo")
        if dtype == "fp16":
            weight_dtype = torch.float16
        elif dtype == "fp32":
            weight_dtype = torch.float32
        else:
            weight_dtype = torch.bfloat16
            
        if unet == "none" or yolo_ckpt == "none" or id_ckpt == "none" or insightface == "none":
           raise "need choice  ckpt in menu"
        else:
            unet_path=folder_paths.get_full_path("SVFR",unet)
            det_path=folder_paths.get_full_path("SVFR",yolo_ckpt)
            id_path=folder_paths.get_full_path("SVFR",id_ckpt)
            face_path=folder_paths.get_full_path("SVFR",insightface)
            UNET= folder_paths.get_full_path("checkpoints",checkpoints)
            vae=folder_paths.get_full_path("vae",vae)
            pipe,id_linear,net_arcface,align_instance=main_loader(weight_dtype,I2V_repo,UNET,vae,unet_path,det_path,id_path,face_path,device,dtype)
        print("****** Load model is done.******")
        return (
        {"pipe": pipe, "id_linear": id_linear, "net_arcface": net_arcface, "align_instance": align_instance, "weight_dtype": weight_dtype},)


class SVFR_Sampler:
    @classmethod
    def INPUT_TYPES(s):
        
        return {
            "required": {
                "image": ("IMAGE",),  # [B,H,W,C], C=3,B>1
                "model": ("MODEL_SVFR",),
                "seed": ("INT", {"default": 77, "min": 0, "max": MAX_SEED}),
                "width": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 64, "display": "number"}),
                "height": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 64, "display": "number"}),
                "decode_chunk_size": ("INT", {"default":16, "min": 4, "max": 128, "step": 4,}),
                "n_sample_frames": ("INT", {"default": 16, "min": 8, "max": 100, "step": 1,}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 4096, "step": 1, "display": "number"}),
                "noise_aug_strength": ("FLOAT", {"default": 0.00, "min": 0.00, "max": 1.00, "step": 0.01, "round": 0.001}),
                "overlap": ("INT", {"default":3, "min": 1, "max": 64}),
                "min_appearance_guidance_scale": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1, "round": 0.01}),
                "max_appearance_guidance_scale": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1, "round": 0.01}),
                "i2i_noise_strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.1, "round": 0.01}),
                "infer_mode":(["bfr","colorization","inpainting","bfr_color","bfr_color_inpaint"],),
                "save_video": ("BOOLEAN", {"default": False},),
                "crop_face_region": ("BOOLEAN", {"default": True},),
            },
            "optional": {"mask": ("MASK",),
                         },
        }
    
    RETURN_TYPES = ("IMAGE", "BBOX_INFO")
    RETURN_NAMES = ("images", "bbox_info")
    FUNCTION = "sampler_main"
    CATEGORY = "SVFR"
    
    def sampler_main(self, image, model, seed, width, height,decode_chunk_size,n_sample_frames
                     , steps,noise_aug_strength, overlap,min_appearance_guidance_scale,max_appearance_guidance_scale, i2i_noise_strength,infer_mode,save_video,crop_face_region,**kwargs):
        
        pipe = model.get("pipe")
        id_linear = model.get("id_linear")
        net_arcface = model.get("net_arcface")
        align_instance = model.get("align_instance")
        weight_dtype = model.get("weight_dtype")
        mask=kwargs.get("mask")
        bbox_info = None
        
        if isinstance(mask,torch.Tensor): #缩放至图片尺寸
            if mask.shape[-1]==64 and mask.shape[-2]==64:
                raise "input mask is not a useful,looks like a default comfyUI mask"

            if len(mask.shape) == 3:  # 1,h,w
                mask_array = mask.squeeze().mul(255).clamp(0, 255).byte().numpy()
            elif len(mask.shape)==2 and mask.shape[0]!=1:  # h,w
                mask_array=mask.mul(255).clamp(0, 255).byte().numpy()
            else:
                raise "check input mask's shape"
            mask_array=np.where(mask_array > 0, 255, 0)
        else:
            mask_array=None
        video_len, _, _, _ = image.size()
        if video_len < 8:
            raise "input video has not much frames below 8 frame,change your input video!"
        else:
            tensor_list = list(torch.chunk(image, chunks=video_len))
            input_frames_pil = [nomarl_upscale(i,width, height) for i in tensor_list]  # tensor to pil
        
        if infer_mode=="bfr":
            task_ids=[0]
        elif infer_mode=="colorization":
            task_ids = [1]
        elif infer_mode=="inpainting":
            task_ids = [2]
        elif infer_mode=="bfr_color":
            task_ids = [0,1]
        else:
            task_ids = [0, 1,2 ]
        
        if not isinstance(mask_array,np.ndarray) and 2 in task_ids:
            raise "If use inpainting need link a mask or a batch mask in the front."
        
        print("******** start infer *********")
        
        if crop_face_region:
            # Store face detection results for later use
            bbox_list = []
            frame_interval = 5
            for frame_idx in range(0, video_len, frame_interval):
                frame = np.array(input_frames_pil[frame_idx])
                _, _, bboxes_list = align_instance(frame[:,:,[2,1,0]], maxface=True)
                if bboxes_list==[]:
                    continue
                x1, y1, ww, hh = bboxes_list[0]
                x2, y2 = x1 + ww, y1 + hh
                bbox = [x1, y1, x2, y2]
                bbox_list.append(bbox)
            if not bbox_list:  # 如果没有检测到任何人脸
                raise "No face detected in the video frames!"
            
            # 获取所有检测到的人脸框的并集
            bbox = get_union_bbox(bbox_list)
            
            # 扩展边界框
            bbox_h = bbox[3] - bbox[1]
            bbox_w = bbox[2] - bbox[0]
            expand_ratio = 0.4
            
            # 扩展时保持原始宽高比
            expand_x1 = max(bbox[0] - expand_ratio * bbox_w, 0)
            expand_y1 = max(bbox[1] - expand_ratio * bbox_h, 0)
            expand_x2 = min(bbox[2] + expand_ratio * bbox_w, frame.shape[1])
            expand_y2 = min(bbox[3] + expand_ratio * bbox_h, frame.shape[0])
            
            bbox_s = [expand_x1, expand_y1, expand_x2, expand_y2]
            bbox_info = (bbox_s, frame.shape[0], frame.shape[1])
        
        images=main_sampler(pipe, align_instance, net_arcface, id_linear, folder_paths.get_output_directory(), weight_dtype,
                          seed,input_frames_pil,task_ids,mask_array,save_video,decode_chunk_size,noise_aug_strength,
                          min_appearance_guidance_scale,max_appearance_guidance_scale,
                          overlap,i2i_noise_strength,steps,n_sample_frames,device,crop_face_region)
        
        #model.to("cpu")#显存不会自动释放，手动迁移，不然很容易OOM
        torch.cuda.empty_cache()
        return (images, bbox_info)


class SVFR_img2mask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),  # [B,H,W,C], C=3,B=1
                "threshold": ("INT", {"default": 0, "min": 0, "max": 254, "step": 0, "display": "number"}),
                "center_crop": ("BOOLEAN", {"default": False},),
                "width": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 64, "display": "number"}),
                "height": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 64, "display": "number"}),
        }
        }
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "main"
    CATEGORY = "SVFR"
    
    def main(self, image,threshold,center_crop,width,height):
        if center_crop:
            image=tensor_upscale(image, width, height)
        np_img=image.squeeze().mul(255).clamp(0, 255).byte().numpy()
        np_img=np.mean(np_img, axis=2).astype(np.uint8)
        
        black_threshold = 50  # 黑色阈值，小于这个值的像素被认为是黑色
        white_threshold = 200  # 白色阈值，大于这个值的像素被认为是白色
        black_pixels = np.sum(np_img < black_threshold)    # 计算黑色像素的数量
        white_pixels = np.sum(np_img > white_threshold) # 计算白色像素的数量
        if black_pixels>white_pixels: #黑多白少，按白色为遮罩
            out = np.where(np_img > threshold, 255, 0).astype(np.float32) / 255.0
        else:
            out = np.where(np_img > threshold, 0, 255).astype(np.float32) / 255.0
        return (torch.from_numpy(out).unsqueeze(0),)


class SVFR_Combine:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "original_frames": ("IMAGE",),  # Original video frames [B,H,W,C]
                "processed_faces": ("IMAGE",),  # Processed face regions from SVFR_Sampler
                "bbox_info": ("BBOX_INFO",),    # Bounding box information from SVFR_Sampler
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("combined_frames",)
    FUNCTION = "combine_faces"
    CATEGORY = "SVFR"

    def combine_faces(self, original_frames, processed_faces, bbox_info):
        if bbox_info is None:
            return (processed_faces,)
            
        bbox_s, orig_height, orig_width = bbox_info
        x1, y1, x2, y2 = [int(x) for x in bbox_s]
        
        # 确保边界框在有效范围内
        x1 = max(0, x1)
        x2 = min(x2, original_frames.shape[2])
        y1 = max(0, y1)
        y2 = min(y2, original_frames.shape[1])
        
        face_region_height = y2 - y1
        face_region_width = x2 - x1
        
        # 创建输出张量
        output_frames = original_frames.clone()
        
        # 创建平滑过渡遮罩
        feather = min(face_region_height, face_region_width) // 8
        feather = max(5, min(feather, 40))  # 限制羽化范围
        
        y, x = np.ogrid[:face_region_height, :face_region_width]
        edge_mask = np.ones((face_region_height, face_region_width))
        
        # 创建渐变边缘
        edge_mask = np.minimum(edge_mask, x / feather)  # 左边缘
        edge_mask = np.minimum(edge_mask, (face_region_width - 1 - x) / feather)  # 右边缘
        edge_mask = np.minimum(edge_mask, y / feather)  # 上边缘
        edge_mask = np.minimum(edge_mask, (face_region_height - 1 - y) / feather)  # 下边缘
        edge_mask = np.clip(edge_mask, 0, 1)
        
        # 应用高斯模糊
        edge_mask = cv2.GaussianBlur(edge_mask, (0, 0), feather/3)
        mask = torch.from_numpy(edge_mask).float().to(original_frames.device)
        mask = mask.unsqueeze(-1).expand(-1, -1, 3)  # 扩展到3通道
        
        # 处理每一帧
        for i in range(len(original_frames)):
            # 将处理后的人脸缩放到目标大小
            face = processed_faces[i]
            face_cropped = face[:, y1:y2, x1:x2] if len(face.shape) == 3 else face[y1:y2, x1:x2]
            
            # 应用遮罩并混合
            output_frames[i, y1:y2, x1:x2] = \
                output_frames[i, y1:y2, x1:x2] * (1 - mask) + \
                face_cropped * mask
        
        return (output_frames,)


NODE_CLASS_MAPPINGS = {
    "SVFR_LoadModel": SVFR_LoadModel,
    "SVFR_Sampler": SVFR_Sampler,
    "SVFR_img2mask":SVFR_img2mask,
    "SVFR_Combine": SVFR_Combine,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SVFR_LoadModel": "SVFR_LoadModel",
    "SVFR_Sampler": "SVFR_Sampler",
    "SVFR_img2mask":"SVFR_img2mask",
    "SVFR_Combine": "SVFR_Combine",
}
