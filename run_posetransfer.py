import argparse
from dofaker import PoseSwapper


def parse_args():
    parser = argparse.ArgumentParser(description='running face swap')
    parser.add_argument('--source',
                        help='select an image or video to be swapped',
                        dest='source',
                        required=True)
    parser.add_argument('--target',
                        help='the target pose image',
                        dest='target',
                        required=True)
    parser.add_argument('--output_dir',
                        help='output directory',
                        dest='output_dir',
                        default='output')
    parser.add_argument('--pose_estimator_name',
                        help='pose estimator name',
                        dest='pose_estimator_name',
                        default='openpose_body')
    parser.add_argument('--pose_estimator_model_dir',
                        help='pose estimator model dir',
                        dest='pose_estimator_model_dir',
                        default='weights/models')
    parser.add_argument('--pose_transfer_name',
                        help='pose transfer name',
                        dest='pose_transfer_name',
                        default='pose_transfer')
    parser.add_argument('--pose_transfer_model_dir',
                        help='pose transfer model dir',
                        dest='pose_transfer_model_dir',
                        default='weights/models')
    parser.add_argument('--det_model_name',
                        help='detection model name for insightface',
                        dest='det_model_name',
                        default='buffalo_l')
    parser.add_argument('--det_model_dir',
                        help='detection model dir for insightface',
                        dest='det_model_dir',
                        default='weights/models')
    parser.add_argument('--image_sr_model',
                        help='image super resolution model',
                        dest='image_sr_model',
                        default='bsrgan')
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
    faker = PoseSwapper(
        pose_estimator_name=args.pose_estimator_name,
        pose_estimator_model_dir=args.pose_estimator_model_dir,
        pose_transfer_name=args.pose_transfer_name,
        pose_transfer_model_dir=args.pose_transfer_model_dir,
        face_det_model=args.det_model_name,
        face_det_model_dir=args.det_model_dir,
        image_sr_model=args.image_sr_model,
        image_sr_model_dir=args.image_sr_model_dir,
        face_enhance_name=args.face_enhance_name,
        face_enhance_model_dir=args.face_enhance_model_dir,
        log_iters=args.log_iters,
        use_enhancer=args.use_enhancer,
        use_sr=args.use_sr,
        scale=args.sr_scale,
    )

    faker.run(
        input_path=args.source,
        target_path=args.target,
        output_dir=args.output_dir,
    )
