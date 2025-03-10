# ComfyUI_SVFR
[SVFR](https://github.com/wangzhiyaoo/SVFR/tree/main) is a unified framework for face video restoration that supports tasks such as BFR, Colorization, Inpainting，you can use it in ComfyUI

# Update 
* 24/02/12 修改模型加载模式为单体模型，此模型调用cofmy的vaae会偏色严重，所以只能用diffuser方法了/Change the model loading mode to a monolithic model

# 1. Installation

In the ./ComfyUI /custom_node directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_SVFR.git
```
# 2. Requirements  
```
pip install -r requirements.txt
```
# 3. Models Required 
* 3.1 download SVFR checkpoints from [google drive](https://drive.google.com/drive/folders/1nzy9Vk-yA_DwXm1Pm4dyE2o0r7V6_5mn) After decompression, place the model in the following file format，从谷歌云盘下载模型，解压后按以下文件格式放置模型；
```
├── Comfyui/models/SVFR/
|   ├── id_linear.pth
|   ├── insightface_glint360k.pth
|   ├── unet.pth
|   ├── yoloface_v5m.pt
```
 * 3.2 [svd_xt.safetensors](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt)  or [svd_xt_1_1.safetensors](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1) 
```
├── Comfyui/models/checkpoints/
|   ├── svd_xt.safetensors or svd_xt_1_1.safetensors
├── Comfyui/models/vae/
|   ├──svd.ave.fp16.safetensors  #rename from stabilityai/stable-video-diffusion-img2vid-xt/vae 重命名repo下的vae模型既可以，不命名也行
```

# 4 Inference mode

* "bfr,colorization,inpainting,bfr_color,bfr_color_inpaint",inpainting and bfr_color_inpaint mode need a mask(use comfyUI mask or black/white jpg)
  
# 5 Example
![](https://github.com/smthemex/ComfyUI_SVFR/blob/main/exampleA.png)

# 6 Citation
```
@misc{wang2025svfrunifiedframeworkgeneralized,
      title={SVFR: A Unified Framework for Generalized Video Face Restoration}, 
      author={Zhiyao Wang and Xu Chen and Chengming Xu and Junwei Zhu and Xiaobin Hu and Jiangning Zhang and Chengjie Wang and Yuqi Liu and Yiyi Zhou and Rongrong Ji},
      year={2025},
      eprint={2501.01235},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.01235}, 
}
```
