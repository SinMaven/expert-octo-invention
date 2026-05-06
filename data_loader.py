import os
import soundfile as sf

def load_dataset(data_path="data/1089"):
   
    audio_files = []
    transcripts = {}

    for root, dirs, files in os.walk(data_path):
        # تحميل الـ transcripts
        for file in files:
            if file.endswith(".trans.txt"):
                with open(os.path.join(root, file), "r") as f:
                    for line in f:
                        parts = line.strip().split(" ", 1)
                        if len(parts) == 2:
                            transcripts[parts[0]] = parts[1]

        # تحميل ملفات الصوت
        for file in files:
            if file.endswith(".flac"):
                file_id = file.replace(".flac", "")
                file_path = os.path.join(root, file)
                if file_id in transcripts:
                    audio_files.append({
                        "path": file_path,
                        "id": file_id,
                        "transcript": transcripts[file_id]
                    })

    print(f" sound file {len(audio_files)} loaded successfully")
    return audio_files


# تجربة
if __name__ == "__main__":
    files = load_dataset()
    print("\n--- First 3 files---")
    for f in files[:3]:
        print(f"📁 {f['id']}")
        print(f"📝 {f['transcript']}\n")