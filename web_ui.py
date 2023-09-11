import gradio as gr
from dofaker import DoFaker


def swap(input_path, dst_path, src_path):
    faker = DoFaker()
    output_path = faker.run(input_path, dst_path, src_path)
    return output_path


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
                convert_button.click(fn=swap, inputs=[image_input, dst_face_image, src_face_image], outputs=[output_image], api_name='image swap')

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
                convert_button.click(fn=swap, inputs=[video_input, dst_face_image, src_face_image], outputs=[output_video], api_name='video swap')


web_ui.launch()
