import os, sys
import pathlib
import logging
import argparse
from pathlib import Path
import gradio as gr
import transformers

project_path = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(project_path))

from mllm.utils import ImageBoxState, bbox_draw, parse_boxes
from mllm.demo.demo_util import NextChatInference
import time

log_level = logging.ERROR
transformers.logging.set_verbosity(log_level)
transformers.logging.enable_default_handler()
transformers.logging.enable_explicit_format()

TEMP_FILE_DIR = Path(__file__).parent / 'temp'
TEMP_FILE_DIR.mkdir(parents=True, exist_ok=True)

#########################################
# mllm model init
#########################################
parser = argparse.ArgumentParser("NExT-Chat Web Demo")
parser.add_argument('--load_in_8bit', action='store_true')
parser.add_argument('--server_name', default=None)
parser.add_argument('--server_port', type=int, default=None)
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--vit_path', type=str, required=True)
parser.add_argument('--image_token_len', type=int, default=576)


args = parser.parse_args()
print(args)

pipe = NextChatInference(args.model_path, args.vit_path, args.image_token_len)


def post_process_response(response):
    if "<at> <boxes>" not in response:
        return response.replace("<", "&lt;").replace(">", "&gt;")
    splits = response.split("<at> <boxes>")
    to_concat = [f"[{i}]" for i in range(len(splits) - 1)]
    rst = [splits[i // 2] if i % 2 == 0 else to_concat[i // 2]
           for i in range(len(splits) + len(to_concat))]
    rst = "".join(rst)
    rst = rst.replace("<", "&lt;").replace(">", "&gt;")
    return rst


def chat_one_turn(
        input_text,
        temperature,
        top_p,
        top_k,
        input_image,
        history,
        hidden_image,
        state,
):
    boxes = state["ibs"].boxes
    gpt_input_text, boxes_seq = parse_boxes(input_text)
    inputs = {"image":input_image['image'], "text": gpt_input_text}
    response, _, _, img = pipe(inputs, temperature=temperature, top_p=top_p, top_k=top_k,
                            boxes=boxes, boxes_seq=boxes_seq)

    ret_text = [(post_process_response(input_text), post_process_response(response))]

    filename_grounding = None
    if img is not None:
        print("detection")
        timestamp = int(time.time())
        filename_grounding = f"tmp/{timestamp}.jpg"
        if not os.path.exists("tmp/"):
            os.makedirs("tmp/")
        img.save(filename_grounding)

    if img is not None:
        ret_text.append((None, (filename_grounding,)))
    return "", ret_text, hidden_image


default_chatbox = [("", "Please begin the chat.")]


def shortcut_func(task_name, text):
    task_name = task_name[0]
    if task_name == "Grounding":
        return "Where is XXX in the image?"
    elif task_name == "Caption":
        return "Can you provide a description of the image and include the locations for each mentioned object?"
    elif task_name == "Explain":
        return text.strip()+" Please include object locations and explain."
    elif task_name == "Region Cap":
        return "What is region [0]?"
    return ""

def new_state():
    return {"ibs": ImageBoxState()}


def clear_fn(value):
    return "", default_chatbox, None, None, new_state()


def clear_fn2(value):
    return default_chatbox, None, new_state()


if __name__ == '__main__':
    # conversation = prepare_interactive(model_args, preprocessor)
    # predict(model, "tmp.jpg", "Find person bottom left in <image>.", boxes=None, boxes_seq=None)
    # import IPython
    # IPython.embed()

    with gr.Blocks() as demo:
        gr.HTML(
            f"""
            <h1 align="center"><font color="#966661">NExT-Chat</font></h1>
            <p align="center">
                <a href='' target='_blank'>[Project]</a>
                <a href='' target='_blank'>[Paper]</a>
            </p>
            <h2>User Manual</h2>
            <ul>
            <li><p><strong>Grounding:</strong> Where is XXX in the &lt;image&gt;? </p></li>
            <li><p><strong>Caption with objects: </strong>Can you provide a description of the image &lt;image&gt; and include the locations for each mentioned object? </p></li>
            <li><p><strong>The model is default not to include obj locations at most time.</strong> </p></li>
            <li><p><strong>To let the model include object locations. You can add prompts like:</strong> </p></li>
                <ul>
                <li><p>Please include object locations and explain. </p></li>
                <li><p>Make sure to include object locations and explain. </p></li>
                <li><p>Please include object locations as much as possible. </p></li>
                </ul>
            <li><p><strong>Region Understanding:</strong> draw boxes and ask like "what is region [0]?" </p></li>

            <ul>
            """
        )

        with gr.Row():
            with gr.Column(scale=6):
                with gr.Group():
                    input_shortcuts = gr.Dataset(components=[gr.Textbox(visible=False)], samples=[
                        ["Grounding"],
                        ["Caption"], ["Explain"], ["Region Cap"]], label="Shortcut Dataset")

                    input_text = gr.Textbox(label='Input Text',
                                            placeholder='Please enter text prompt below and press ENTER.')

                    with gr.Row():
                        input_image = gr.ImageMask()
                        out_imagebox = gr.Image(label="Parsed Sketch Pad")
                    input_image_state = gr.State(new_state())

                with gr.Row():
                    temperature = gr.Slider(maximum=1, value=0.8, minimum=0, label='Temperature')
                    top_p = gr.Slider(maximum=1, value=0.7, minimum=0, label='Top P')
                    top_k = gr.Slider(maximum=100, value=5, minimum=1, step=1, label='Top K')

                with gr.Row():
                    run_button = gr.Button('Generate')
                    clear_button = gr.Button('Clear')

            with gr.Column(scale=4):
                output_text = gr.components.Chatbot(label='Multi-round conversation History',
                                                    value=default_chatbox).style(height=550)
                output_image = gr.Textbox(visible=False)

        input_shortcuts.click(fn=shortcut_func, inputs=[input_shortcuts, input_text], outputs=[input_text])

        run_button.click(fn=chat_one_turn, inputs=[input_text, temperature, top_p, top_k,
                                                   input_image, output_text, output_image,
                                                   input_image_state],
                         outputs=[input_text, output_text, output_image])
        input_text.submit(fn=chat_one_turn, inputs=[input_text, temperature, top_p, top_k,
                                                    input_image, output_text, output_image,
                                                    input_image_state],
                          outputs=[input_text, output_text, output_image])
        clear_button.click(fn=clear_fn, inputs=clear_button,
                           outputs=[input_text, output_text, input_image, out_imagebox, input_image_state])
        input_image.upload(fn=clear_fn2, inputs=clear_button, outputs=[output_text, out_imagebox, input_image_state])
        input_image.clear(fn=clear_fn2, inputs=clear_button, outputs=[output_text, out_imagebox, input_image_state])
        input_image.edit(
            fn=bbox_draw,
            inputs=[input_image, input_image_state],
            outputs=[out_imagebox, input_image_state],
            queue=False,
        )

        with gr.Row():
            gr.Examples(
                examples=[
                    [
                        os.path.join(os.path.dirname(__file__), "assets/dog.jpg"),
                        "Can you describe the image and include object locations?",
                        new_state(),
                    ],
                    [
                        os.path.join(os.path.dirname(__file__), "assets/fishing.jpg"),
                        "A boy is sleeping on bed, is this correct? Please include object locations.",
                        new_state(),
                    ],
                    [
                        os.path.join(os.path.dirname(__file__), "assets/rec_bear.png"),
                        "Where is the bear wearing the red decoration in the image?",
                        new_state(),
                    ],
                    [
                        os.path.join(os.path.dirname(__file__), "assets/woman.jpeg"),
                        "What is the woman doing? Please include object locations.",
                        new_state(),
                    ],
                ],
                inputs=[input_image, input_text, input_image_state],
            )

    print("launching...")
    demo.queue().launch(server_name=args.server_name, server_port=args.server_port, share=True)