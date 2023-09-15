English|[简体中文](README_ch.md)
[![colab](https://img.shields.io/static/v1?label=colab&message=colab&color=yellow)](https://colab.research.google.com/drive/1i1hO-_yS6kZdrLden8Mo_hfeHFD-cBt0?usp=sharing)

# DoFaker: A very simple face swapping tool
Insightface based face swapping tool to replace faces in videos or images. Windows and linux support CPU and GPU. Onnxruntime inference without pytorch.

<p align="center">
<img src="https://github.com/justld/dofaker/blob/main/docs/images/source.gif" width="300" height="150"><img src="https://github.com/justld/dofaker/blob/main/docs/images/trump.jpg" width="300" height="150">
</p>

<p align="center">
<img src="https://github.com/justld/dofaker/blob/main/docs/images/swapped.gif" width="600" height="300"/>
</p>

<p align="center">
    <img src="https://github.com/justld/dofaker/blob/main/docs/test/multi.png" width="600" height="300"/>
    <img src="https://github.com/justld/dofaker/blob/main/docs/images/multi.png" width="600" height="300"/>
</p>

# Update
- 2023/9/14 update face enhance(GFPGAN) and image super resolution(BSRGAN)

# Tutorial
[dofaker tutorial in youtube：face swap so easy](https://www.youtube.com/watch?v=qd1-JSpiZao)   
[colab online](https://colab.research.google.com/drive/1i1hO-_yS6kZdrLden8Mo_hfeHFD-cBt0?usp=sharing)


# Qiuck Start
install dofaker
```bash
git clone https://github.com/justld/dofaker.git
cd dofaker
conda create -n dofaker
conda activate dofaker
pip install onnxruntime # onnxruntime-gpu
pip install -e .
```

open web ui(The model weights will be downloaded automatically):
```bash
dofaker
```

command line(linux):
```
bash test.sh
```


# Install from source code
## 一、Installation
You should install onnxruntime or onnxruntime-gpu manually.

### conda install
create virtual environment:
```bash
git clone https://github.com/justld/dofaker.git
cd dofaker
conda create -n dofaker
conda activate dofaker
pip install -r requirements.txt
pip install onnxruntime # onnxruntime-gpu
pip install -e .
```

### pip install
```bash
git clone https://github.com/justld/dofaker.git
cd dofaker
pip install -r requirements.txt
pip install onnxruntime # onnxruntime-gpu
pip install -e .
```

## 二、Download Weight
All weights can be downloaded from [release](https://github.com/justld/dofaker/releases). These weight come from links refer to the botton links.

Unzip the zip file, the dir looks like follow:
```
|-dofaker
|-docs
|-weights
----|-models
--------|-buffalo_l
----------|-1k3d68.onnx
----------|-2d106det.onnx
----------|-...
--------|-buffalo_l.zip
--------|-inswapper_128.onnx
--------|-GFPGANv1.3.onnx
--------|-bsrgan_4.onnx
|-run.py
|-web_ui.py
```


## 三、Usage
You can use dofaker in web_ui or command line.
### web ui
web gui only support one face swap once, if you want to swap multiple faces, please refer to command usage.
```bash
python web_ui.py
```

### command
You can swap multiple faces in command.
```bash
python run.py --source "image or video path to be swapped" --dst_face_paths "dst_face1_path" "dst_face2_path" ... --src_face_paths "src_face1_path" "src_face2_path" ...
```

The command follow will replace dst_face1 and dst_face2 detected in input_video.mp4 with src_face1 and src_face2:
```bash
python run.py --source input_video.mp4 --dst_face_paths dst_face1.jpg dst_face2.jpg --src_face_paths src_face1.jpg src_face2.jpg
```

|args|description|
|:---:|:---:|
|source|The image or video to be swapped|
|dst_face_paths|The images includding faces in source to be swapped. If None, replace all faces in source media.|
|src_face_paths|The images includding faces in source to be swapped|


# Attention
Do not apply this software to scenarios that violate morality, law, or infringement. The consequences caused by using this software shall be borne by the user themselves.

# Sponsor
[Thank you for support](https://www.paypal.com/paypalme/justldu)

# Thanks
- [insightface](https://github.com/deepinsight/insightface)  
- [GFPGAN](https://github.com/TencentARC/GFPGAN)  
- [GFPGAN-onnxruntime-demo](https://github.com/xuanandsix/GFPGAN-onnxruntime-demo)  
- [BSRGAN](https://github.com/cszn/BSRGAN)  
