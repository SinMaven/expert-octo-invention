import gradio as gr
import tempfile
import soundfile as sf
import numpy as np
from model import process_audio

def handle_audio(audio):
    if audio is None:
        return "No audio recorded", "No audio recorded"
    
    sample_rate, data = audio
    
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    
    data = data.astype(np.float32)
    if data.max() > 1.0:
        data = data / 32768.0
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, data, sample_rate)
        raw_text, redacted_text = process_audio(f.name)
    
    return raw_text, redacted_text

app = gr.Interface(
    fn=handle_audio,
    inputs=gr.Audio(
        sources=["microphone", "upload"],
        label="Speak or Upload Audio"
    ),
    outputs=[
        gr.Textbox(label="Original Text"),
        gr.Textbox(label="Redacted Text")
    ],
    title="Zero-Knowledge Voice",
    description="Speech to Text with PII Redaction - 100% Offline"
)

app.launch()