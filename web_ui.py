import argparse

import gradio as gr
from dofaker import FaceSwapper, PoseSwapper


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
    return parser.parse_args()


def swap_face(input_path, dst_path, src_path, use_enhancer, use_sr, scale,
              face_sim_thre):
    faker = FaceSwapper(use_enhancer=use_enhancer,
                        use_sr=use_sr,
                        scale=scale,
                        face_sim_thre=face_sim_thre)
    output_path = faker.run(input_path, dst_path, src_path)
    return output_path


def swap_pose(input_path, target_path, use_enhancer, use_sr, scale):
    faker = PoseSwapper(use_enhancer=use_enhancer, use_sr=use_sr, scale=scale)
    output_path = faker.run(input_path, target_path)
    return output_path


def main():
    args = parse_args()

    with gr.Blocks(title='DoFaker') as web_ui:
        gr.Markdown('DoFaker: Face Swap and pose swap web ui')
        with gr.Tab('FaceSwapper'):
            gr.Markdown('DoFaker: Face Swap Web UI')
            with gr.Tab('Image'):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown('The source image to be swapped')
                        image_input = gr.Image(type='filepath')
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown(
                                    'target face included in source image')
                                dst_face_image = gr.Image(type='filepath')
                            with gr.Column():
                                gr.Markdown(
                                    'source face to replace target face')
                                src_face_image = gr.Image(type='filepath')

                    with gr.Column():
                        output_image = gr.Image(type='filepath')
                        use_enhancer = gr.Checkbox(
                            label="face enhance",
                            info="Whether use face enhance model.")
                        with gr.Row():
                            use_sr = gr.Checkbox(
                                label="super resolution",
                                info="Whether use image resolution model.")
                            scale = gr.Number(
                                value=1, label='image super resolution scale')
                        with gr.Row():
                            face_sim_thre = gr.Number(
                                value=0.6,
                                label='face similarity threshold',
                                minimum=0.0,
                                maximum=1.0)
                        convert_button = gr.Button('Swap')
                        convert_button.click(fn=swap_face,
                                             inputs=[
                                                 image_input, dst_face_image,
                                                 src_face_image, use_enhancer,
                                                 use_sr, scale, face_sim_thre
                                             ],
                                             outputs=[output_image],
                                             api_name='image swap')

            with gr.Tab('Video'):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown('The source video to be swapped')
                        video_input = gr.Video(type='filepath')
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown(
                                    'target face included in source image')
                                dst_face_image = gr.Image(type='filepath')
                            with gr.Column():
                                gr.Markdown(
                                    'source face to replace target face')
                                src_face_image = gr.Image(type='filepath')

                    with gr.Column():
                        output_video = gr.Video(type='filepath')
                        use_enhancer = gr.Checkbox(
                            label="face enhance",
                            info="Whether use face enhance model.")
                        with gr.Row():
                            use_sr = gr.Checkbox(
                                label="super resolution",
                                info="Whether use image resolution model.")
                            scale = gr.Number(
                                value=1, label='image super resolution scale')
                        with gr.Row():
                            face_sim_thre = gr.Number(
                                value=0.6,
                                label='face similarity threshold',
                                minimum=0.0,
                                maximum=1.0)
                        convert_button = gr.Button('Swap')
                        convert_button.click(fn=swap_face,
                                             inputs=[
                                                 video_input, dst_face_image,
                                                 src_face_image, use_enhancer,
                                                 use_sr, scale, face_sim_thre
                                             ],
                                             outputs=[output_video],
                                             api_name='video swap')

        with gr.Tab('PoseSwapper'):
            gr.Markdown('DoFaker: Pose Swap Web UI')
            with gr.Tab('Image'):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown('The source image to be swapped')
                        image_input = gr.Image(type='filepath')
                        gr.Markdown('The target image with pose')
                        target = gr.Image(type='filepath')

                    with gr.Column():
                        output_image = gr.Image(type='filepath')
                        use_enhancer = gr.Checkbox(
                            label="face enhance",
                            info="Whether use face enhance model.")
                        with gr.Row():
                            use_sr = gr.Checkbox(
                                label="super resolution",
                                info="Whether use image resolution model.")
                            scale = gr.Number(
                                value=1, label='image super resolution scale')
                        convert_button = gr.Button('Swap')
                        convert_button.click(fn=swap_pose,
                                             inputs=[
                                                 image_input, target,
                                                 use_enhancer, use_sr, scale
                                             ],
                                             outputs=[output_image],
                                             api_name='image swap')

            # with gr.Tab('Video'):
            #     with gr.Row():
            #         with gr.Column():
            #             gr.Markdown('The source video to be swapped')
            #             video_input = gr.Image(type='filepath')
            #             gr.Markdown('The target image with pose')
            #             target = gr.Video(type='filepath')

            #         with gr.Column():
            #             output_video = gr.Video(type='filepath')
            #             use_enhancer = gr.Checkbox(
            #                 label="face enhance",
            #                 info="Whether use face enhance model.")
            #             with gr.Row():
            #                 use_sr = gr.Checkbox(
            #                     label="super resolution",
            #                     info="Whether use image resolution model.")
            #                 scale = gr.Number(value=1,
            #                                 label='image super resolution scale')
            #             convert_button = gr.Button('Swap')
            #             convert_button.click(fn=swap_pose,
            #                                 inputs=[
            #                                     video_input, target, use_enhancer,
            #                                     use_sr, scale
            #                                 ],
            #                                 outputs=[output_video],
            #                                 api_name='video swap')

    web_ui.launch(inbrowser=args.inbrowser, server_port=args.server_port)


if __name__ == '__main__':
    main()
