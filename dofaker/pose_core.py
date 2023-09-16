import os
import cv2

import numpy as np
from moviepy.editor import VideoFileClip

from .pose import PoseEstimator, PoseTransfer
from .face_enhance import GFPGAN
from .super_resolution import BSRGAN
from .face_det import FaceAnalysis


class PoseSwapper:

    def __init__(self,
                 pose_estimator_name='openpose_body',
                 pose_estimator_model_dir='weights/models',
                 pose_transfer_name='pose_transfer',
                 pose_transfer_model_dir='weights/models',
                 image_sr_model='bsrgan',
                 image_sr_model_dir='weights/models',
                 face_enhance_name='gfpgan',
                 face_enhance_model_dir='weights/models/',
                 face_det_model='buffalo_l',
                 face_det_model_dir='weights/models',
                 log_iters=10,
                 use_enhancer=True,
                 use_sr=True,
                 scale=1):
        pose_estimator = PoseEstimator(name=pose_estimator_name,
                                       root=pose_estimator_model_dir)
        self.pose_transfer = PoseTransfer(name=pose_transfer_name,
                                          root=pose_transfer_model_dir,
                                          pose_estimator=pose_estimator)

        if use_enhancer:
            self.det_model = FaceAnalysis(name=face_det_model,
                                          root=face_det_model_dir)
            self.det_model.prepare(ctx_id=1, det_size=(640, 640))
            self.face_enhance = GFPGAN(name=face_enhance_name,
                                       root=face_enhance_model_dir)
        self.use_enhancer = use_enhancer

        if use_sr:
            self.sr_model = BSRGAN(name=image_sr_model,
                                   root=image_sr_model_dir,
                                   scale=scale)
            self.scale = scale
        else:
            self.scale = 1
        self.use_sr = use_sr
        self.log_iters = log_iters

    def run(self, input_path, target_path, output_dir='output'):
        assert os.path.exists(
            input_path), "The input path {} not exists.".format(input_path)
        assert os.path.exists(
            target_path), "The target path {} not exists.".format(target_path)
        os.makedirs(output_dir, exist_ok=True)
        assert input_path.lower().endswith(
            ('jpg', 'jpeg', 'webp', 'png', 'bmp')
        ), "pose swapper input must be image endswith ('jpg', 'jpeg', 'webp', 'png', 'bmp'), but got {}.".format(
            input_path)
        if target_path.lower().endswith(('jpg', 'jpeg', 'webp', 'png', 'bmp')):
            return self.transfer_image(input_path, target_path, output_dir)
        else:
            return self.transfer_video(input_path, target_path, output_dir)

    def transfer_image(self, input_path, target_path, output_dir):
        source = cv2.imread(input_path)
        target = cv2.imread(target_path)
        transferred_image = self.transfer_pose(source,
                                               target,
                                               image_format='bgr')
        base_name = os.path.basename(input_path)
        output_path = os.path.join(output_dir, base_name)
        cv2.imwrite(output_path, transferred_image)
        return output_path

    def transfer_video(self, input_path, target_path, output_dir):
        source = cv2.imread(input_path)
        video = cv2.VideoCapture(target_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        height, width, _ = source.shape
        frame_size = (width, height)
        print('video fps: {}, total_frames: {}, width: {}, height: {}'.format(
            fps, total_frames, width, height))

        video_name = os.path.basename(input_path).split('.')[0]
        four_cc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        temp_video_path = os.path.join(output_dir,
                                       'temp_{}.mp4'.format(video_name))
        save_video_path = os.path.join(output_dir, '{}.mp4'.format(video_name))
        output_video = cv2.VideoWriter(
            temp_video_path, four_cc, fps,
            (int(frame_size[0] * self.scale), int(frame_size[1] * self.scale)))
        i = 0
        while video.isOpened():
            ret, frame = video.read()
            if ret:
                transferred_image = self.transfer_pose(source,
                                                       frame,
                                                       image_format='bgr')
                i += 1
                if i % self.log_iters == 0:
                    print('processing {}/{}'.format(i, total_frames))
                output_video.write(transferred_image)
            else:
                break

        video.release()
        output_video.release()
        print(temp_video_path)
        self.add_audio_to_video(target_path, temp_video_path, save_video_path)
        os.remove(temp_video_path)
        return save_video_path

    def transfer_pose(self, source, target, image_format='bgr'):
        transferred_image = self.pose_transfer.get(source,
                                                   target,
                                                   image_format=image_format)
        if self.use_enhancer:
            faces = self.det_model.get(transferred_image, max_num=1)
            for face in faces:
                transferred_image = self.face_enhance.get(
                    transferred_image,
                    face,
                    paste_back=True,
                    image_format=image_format)

        if self.use_sr:
            transferred_image = self.sr_model.get(transferred_image,
                                                  image_format=image_format)
        return transferred_image

    def add_audio_to_video(self, src_video_path, target_video_path,
                           save_video_path):
        audio = VideoFileClip(src_video_path).audio
        target_video = VideoFileClip(target_video_path)
        target_video = target_video.set_audio(audio)
        target_video.write_videofile(save_video_path)
        return target_video_path
