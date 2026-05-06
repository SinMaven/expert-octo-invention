from faster_whisper import WhisperModel
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

print("Loading models...")
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()
print("Models loaded!")


def transcribe(audio_path):
    segments, info = whisper_model.transcribe(audio_path)
    text = " ".join([seg.text for seg in segments])
    return text.strip()


def redact_pii(text):
    results = analyzer.analyze(
    text=text,
    entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "CREDIT_CARD", "US_SSN", "LOCATION"],
    language="en"
        )
    anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
    return anonymized.text


def process_audio(audio_path):
    raw_text = transcribe(audio_path)
    print(f"Original : {raw_text}")

    redacted_text = redact_pii(raw_text)
    print(f"Redacted : {redacted_text}")

    return raw_text, redacted_text


if __name__ == "__main__":
    from data_loader import load_dataset

    files = load_dataset()
    first_file = files[0]
    print(f"\nProcessing: {first_file['id']}")
    print(f"Expected : {first_file['transcript']}\n")
    process_audio(first_file['path'])