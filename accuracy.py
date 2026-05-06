from jiwer import wer
from data_loader import load_dataset
from model import transcribe

def calculate_wer():
    files = load_dataset()
    
    references = []
    hypotheses = []

    for i, file in enumerate(files[:10]):
        print(f"Processing {i+1}/10: {file['id']}")
        
        ref = file['transcript'].lower()
        hyp = transcribe(file['path']).lower()
        
        references.append(ref)
        hypotheses.append(hyp)
        
        print(f"Expected : {ref}")
        print(f"Got      : {hyp}\n")

    error_rate = wer(references, hypotheses)
    
    print("=" * 50)
    print(f"Your WER     : {error_rate * 100:.2f}%")
    print(f"Whisper Paper: 8.80%")
    print(f"Difference   : {(error_rate - 0.088) * 100:.2f}%")
    print("=" * 50)

if __name__ == "__main__":
    calculate_wer()