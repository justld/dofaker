import argparse
from dofaker import FaceSwapper


def parse_args():
    parser = argparse.ArgumentParser(description='running face swap')
    parser.add_argument('--source',
                        help='select an image or video to be swapped',
                        dest='source',
                        required=True)
    parser.add_argument('--dst_face_paths',
                        help='select images in source to be swapped',
                        dest='dst_face_paths',
                        nargs='+',
                        default=None)
    parser.add_argument(
        '--src_face_paths',
        help='select images to replace dst_faces in source image or video.',
        dest='src_face_paths',
        nargs='+',
        required=True)
    parser.add_argument('--output_dir',
                        help='output directory',
                        dest='output_dir',
                        default='output')
    parser.add_argument('--det_model_name',
                        help='detection model name for insightface',
                        dest='det_model_name',
                        default='buffalo_l')
    parser.add_argument('--det_model_dir',
                        help='detection model dir for insightface',
                        dest='det_model_dir',
                        default='weights/models')
    parser.add_argument('--swap_model_name',
                        help='swap model name',
                        dest='swap_model_name',
                        default='inswapper')
    parser.add_argument('--image_sr_model',
                        help='image super resolution model',
                        dest='image_sr_model',
                        default='bsrgan')
    parser.add_argument('--face_swap_model_dir',
                        help='swap model path',
                        dest='face_swap_model_dir',
                        default='weights/models')
    parser.add_argument('--image_sr_model_dir',
                        help='image super resolution model dir',
                        dest='image_sr_model_dir',
                        default='weights/models')
    parser.add_argument('--face_enhance_name',
                        help='face enhance model',
                        dest='face_enhance_name',
                        default='gfpgan')
    parser.add_argument('--face_enhance_model_dir',
                        help='face enhance model dir',
                        dest='face_enhance_model_dir',
                        default='weights/models')
    parser.add_argument('--face_sim_thre',
                        help='similarity of face embedding threshold',
                        dest='face_sim_thre',
                        default=0.5)
    parser.add_argument('--log_iters',
                        help='print log intervals',
                        dest='log_iters',
                        default=10,
                        type=int)
    parser.add_argument('--use_enhancer',
                        help='whether use face enhance model',
                        dest='use_enhancer',
                        action='store_true')
    parser.add_argument('--use_sr',
                        help='whether use image super resolution model',
                        dest='use_sr',
                        action='store_true')
    parser.add_argument('--sr_scale',
                        help='image super resolution scale',
                        dest='sr_scale',
                        default=1,
                        type=float)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    faker = FaceSwapper(
        face_det_model=args.det_model_name,
        face_det_model_dir=args.det_model_dir,
        face_swap_model=args.swap_model_name,
        face_swap_model_dir=args.face_swap_model_dir,
        image_sr_model=args.image_sr_model,
        image_sr_model_dir=args.image_sr_model_dir,
        face_enhance_model=args.face_enhance_name,
        face_enhance_model_dir=args.face_enhance_model_dir,
        face_sim_thre=args.face_sim_thre,
        log_iters=args.log_iters,
        use_enhancer=args.use_enhancer,
        use_sr=args.use_sr,
        scale=args.sr_scale,
    )

    faker.run(
        input_path=args.source,
        dst_face_paths=args.dst_face_paths,
        src_face_paths=args.src_face_paths,
        output_dir=args.output_dir,
    )
