# DoFaker: A very simple face swapping tool
Insightface based face swapping tool to replace faces in videos or images.

<img src="https://github.com/justld/dofaker/blob/main/docs/images/source.gif" width="180" height="105"><img src="https://github.com/justld/dofaker/blob/main/docs/images/trump.jpg" width="180" height="105">

<img src="https://github.com/justld/dofaker/blob/main/docs/images/swapped.gif" width="180" height="105"/>



# 一、Installation
For GPU users, replace all 'requirements.txt' with 'requirements_gpu.txt' in the follow commands.

## conda install
create virtual environment:
```bash
git clone https://github.com/justld/dofaker.git
cd dofaker
conda create -n dofaker
conda activate dofaker
pip install -r requirements.txt
```

## pip install
```bash
git clone https://github.com/justld/dofaker.git
cd dofaker
pip install -r requirements.txt
```

# 二、Download Weight
All weights can be downloaded from [insightface](https://github.com/deepinsight/insightface), you can also
download face det model and face swap model in :[google drive](https://drive.google.com/drive/folders/1R6yMDQiHQg938M5GIz4_mOOhpF8ybrv9?usp=sharing) or [baidu drive(extract code:tkf3)](https://pan.baidu.com/s/1sF3QbwAK1sVqdie1KqgkkA)

The dir looks like follow:
```
|-dofaker
|-docs
|-weights
----|-models
--------|-buffalo_l.zip
--------|-inswapper_128.onnx
|-run.py
|-web_ui.py
```


# 三、Usage
You can use dofaker in web_ui or command line.
## web ui
web gui only support one face swap once, if you want to swap multiple faces, please refer to command usage.
```bash
python web_ui.py
```

## command
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

# Thanks
[insightface](https://github.com/deepinsight/insightface)
