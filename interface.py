import gradio as gr
from sd_image_mosaic import generate_mosaique

demo = gr.Interface(fn=generate_mosaique, inputs=[
    gr.Image(type="filepath"),
    "textbox", 
    gr.Slider(1, 30, value=15, label="rows", info="Choose between 1 and 30"),
    gr.Slider(1, 30, value=15, label="cols", info="Choose between 1 and 30")
    ], outputs="image")
demo.launch()