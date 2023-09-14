import os

import numpy as np
import cv2
from moviepy.editor import VideoFileClip

from .face_det import FaceAnalysis
from dofaker.face_swap import get_swapper_model


class DoFaker:

    def __init__(self,
                 face_det_model='buffalo_l',
                 face_swap_model='inswapper',
                 face_det_model_dir='weights/models',
                 face_swap_model_dir='weights/models',
                 face_sim_thre=0.5,
                 log_iters=10):
        self.face_sim_thre = face_sim_thre
        self.log_iters = log_iters

        self.det_model = FaceAnalysis(name=face_det_model,
                                      root=face_det_model_dir)
        self.det_model.prepare(ctx_id=0, det_size=(640, 640))

        self.swapper_model = get_swapper_model(name=face_swap_model,
                                               root=face_swap_model_dir)

    def run(self,
            input_path: str,
            dst_face_paths,
            src_face_paths,
            output_dir='output'):
        if isinstance(dst_face_paths, str):
            dst_face_paths = [dst_face_paths]
        if isinstance(src_face_paths, str):
            src_face_paths = [src_face_paths]
        if input_path.lower().endswith(('jpg', 'jpeg', 'webp', 'png', 'bmp')):
            return self.swap_image(input_path, dst_face_paths, src_face_paths,
                                   output_dir)
        else:
            return self.swap_video(input_path, dst_face_paths, src_face_paths,
                                   output_dir)

    def swap_video(self,
                   input_video_path,
                   dst_face_paths,
                   src_face_paths,
                   output_dir='output'):
        assert os.path.exists(
            input_video_path), 'The input video path {} not exist.'
        os.makedirs(output_dir, exist_ok=True)
        src_faces = self.get_faces(src_face_paths)
        if dst_face_paths is not None:
            dst_faces = self.get_faces(dst_face_paths)
            dst_face_embeddings = self.get_faces_embeddings(dst_faces)
            assert len(dst_faces) == len(
                src_faces
            ), 'The detected faces in source images not equal target image faces.'

        video = cv2.VideoCapture(input_video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_size = (width, height)
        print('video fps: {}, total_frames: {}, width: {}, height: {}'.format(
            fps, total_frames, width, height))

        video_name = os.path.basename(input_video_path).split('.')[0]
        four_cc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        temp_video_path = os.path.join(output_dir,
                                       'temp_{}.mp4'.format(video_name))
        save_video_path = os.path.join(output_dir, '{}.mp4'.format(video_name))
        output_video = cv2.VideoWriter(temp_video_path, four_cc, fps,
                                       frame_size)

        i = 0
        while video.isOpened():
            ret, frame = video.read()
            if ret:
                if dst_face_paths is not None:
                    swapped_image = self.swap_faces(frame,
                                                    dst_face_embeddings,
                                                    src_faces=src_faces)
                else:
                    swapped_image = self.swap_all_faces(frame,
                                                        src_faces=src_faces)
                i += 1
                if i % self.log_iters == 0:
                    print('processing {}/{}'.format(i, total_frames))
                output_video.write(swapped_image)
            else:
                break

        video.release()
        output_video.release()
        self.add_audio_to_video(input_video_path, temp_video_path,
                                save_video_path)
        os.remove(temp_video_path)
        return save_video_path

    def swap_image(self,
                   image_path,
                   dst_face_paths,
                   src_face_paths,
                   output_dir='output'):
        os.makedirs(output_dir, exist_ok=True)
        src_faces = self.get_faces(src_face_paths)
        if dst_face_paths is not None:
            dst_faces = self.get_faces(dst_face_paths)
            dst_face_embeddings = self.get_faces_embeddings(dst_faces)
            assert len(dst_faces) == len(
                src_faces
            ), 'The detected faces in source images not equal target image faces.'

        image = cv2.imread(image_path)
        if dst_face_paths is not None:
            swapped_image = self.swap_faces(image,
                                            dst_face_embeddings,
                                            src_faces=src_faces)
        else:
            swapped_image = self.swap_all_faces(image, src_faces=src_faces)
        base_name = os.path.basename(image_path)
        save_path = os.path.join(output_dir, base_name)
        cv2.imwrite(save_path, swapped_image)
        return save_path

    def add_audio_to_video(self, src_video_path, target_video_path,
                           save_video_path):
        audio = VideoFileClip(src_video_path).audio
        target_video = VideoFileClip(target_video_path)
        target_video = target_video.set_audio(audio)
        target_video.write_videofile(save_video_path)
        return target_video_path

    def get_faces(self, image_paths):
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        faces = []
        for image_path in image_paths:
            image = cv2.imread(image_path)
            img_faces = self.det_model.get(image, max_num=1)
            assert len(
                img_faces
            ) == 1, 'The detected face in image {} must be 1, but got {}, please ensure your image including one face.'.format(
                image_path, len(img_faces))
            faces += img_faces
        return faces

    def swap_faces(self, image, dst_face_embeddings: np.ndarray,
                   src_faces: list) -> np.ndarray:
        res = image.copy()
        image_faces = self.det_model.get(image)
        image_face_embeddings = self.get_faces_embeddings(image_faces)
        sim = np.dot(dst_face_embeddings, image_face_embeddings.T)

        for i in range(dst_face_embeddings.shape[0]):
            index = np.where(sim[i] > self.face_sim_thre)[0].tolist()
            for idx in index:
                res = self.swapper_model.get(res,
                                             image_faces[idx],
                                             src_faces[i],
                                             paste_back=True)
        return res

    def swap_all_faces(self, image, src_faces: list) -> np.ndarray:
        assert len(
            src_faces
        ) == 1, 'If replace all faces in source, the number of src face should be 1, but got {}.'.format(
            len(src_faces))
        res = image.copy()
        image_faces = self.det_model.get(image)
        for image_face in image_faces:
            res = self.swapper_model.get(res,
                                         image_face,
                                         src_faces[0],
                                         paste_back=True)
        return res

    def get_faces_embeddings(self, faces):
        feats = []
        for face in faces:
            feats.append(face.normed_embedding)
        if len(feats) == 1:
            feats = np.array(feats, dtype=np.float32).reshape(1, -1)
        else:
            feats = np.array(feats, dtype=np.float32)
        return feats
