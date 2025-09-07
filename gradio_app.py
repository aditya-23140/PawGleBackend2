import gradio as gr
from models.predict import predict_animal

iface = gr.Interface(
    fn=predict_animal,
    inputs=gr.Image(type="pil"),
    outputs="label",
    title="Cat & Dog Classifier"
)

if __name__ == "__main__":
    iface.launch(share=True)
