import argparse

import gradio as gr
from dofaker import DoFaker

import argparse
from dofaker import DoFaker

faker = None


def parse_args():
    parser = argparse.ArgumentParser(description='running face swap')
    parser.add_argument(
        '--inbrowser',
        help=
        'whether to automatically launch the interface in a new tab on the default browser.',
        dest='inbrowser',
        default=True)
    parser.add_argument(
        '--server_port',
        help=
        'will start gradio app on this port (if available). Can be set by environment variable GRADIO_SERVER_PORT. If None, will search for an available port starting at 7860.',
        dest='server_port',
        type=int,
        default=None)
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
    parser.add_argument('--face_swap_model_dir',
                        help='swap model path',
                        dest='face_swap_model_dir',
                        default='weights/models')
    parser.add_argument('--face_sim_thre',
                        help='similarity of face embedding threshold',
                        dest='face_sim_thre',
                        default=0.5)
    parser.add_argument('--log_iters',
                        help='print log intervals',
                        dest='log_iters',
                        default=10)
    return parser.parse_args()


def swap(input_path, dst_path, src_path):
    global faker
    output_path = faker.run(input_path, dst_path, src_path)
    return output_path


def main():
    args = parse_args()

    global faker
    faker = DoFaker(
        face_det_model=args.det_model_name,
        face_det_model_dir=args.det_model_dir,
        face_swap_model=args.swap_model_name,
        face_swap_model_dir=args.face_swap_model_dir,
        face_sim_thre=args.face_sim_thre,
        log_iters=args.log_iters,
    )

    with gr.Blocks(title='DoFaker') as web_ui:
        gr.Markdown('DoFaker: Face Swap Web UI')
        with gr.Tab('Image'):
            with gr.Row():
                with gr.Column():
                    gr.Markdown('The source image to be swapped')
                    image_input = gr.Image(type='filepath')
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown('target face included in source image')
                            dst_face_image = gr.Image(type='filepath')
                        with gr.Column():
                            gr.Markdown('source face to replace target face')
                            src_face_image = gr.Image(type='filepath')

                with gr.Column():
                    output_image = gr.Image(type='filepath')
                    convert_button = gr.Button('Swap')
                    convert_button.click(
                        fn=swap,
                        inputs=[image_input, dst_face_image, src_face_image],
                        outputs=[output_image],
                        api_name='image swap')

        with gr.Tab('Video'):
            with gr.Row():
                with gr.Column():
                    gr.Markdown('The source video to be swapped')
                    video_input = gr.Video(type='filepath')
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown('target face included in source image')
                            dst_face_image = gr.Image(type='filepath')
                        with gr.Column():
                            gr.Markdown('source face to replace target face')
                            src_face_image = gr.Image(type='filepath')

                with gr.Column():
                    output_video = gr.Video(type='filepath')
                    convert_button = gr.Button('Swap')
                    convert_button.click(
                        fn=swap,
                        inputs=[video_input, dst_face_image, src_face_image],
                        outputs=[output_video],
                        api_name='video swap')

    web_ui.launch(inbrowser=args.inbrowser, server_port=args.server_port)


if __name__ == '__main__':
    main()
