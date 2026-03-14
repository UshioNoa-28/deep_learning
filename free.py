import gradio as gr

def hello(x):
    return x

demo = gr.Interface(fn=hello, inputs="text", outputs="text")

demo.launch()