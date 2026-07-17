import os

os.environ["PATH"] += r";C:\ffmpeg\bin"
import json
import whisper

model = whisper.load_model("large-v2")

audios = os.listdir("audios")

for audio in audios:
    # print(audio)
    if "_" in audio:
        number = audio.split("_")[0]
        title = audio.split("_")[1][:-4]

        print(number, title)

        result = model.transcribe(
            audio=f"audios/{audio}",
            # result = model.transcribe(audio = f"audios/sample.ogg",
            language="hi",
            task="translate",
            word_timestamps=False,
        )

        chunks = []
        for segment in result["segments"]:
            chunks.append(
                {
                    "number": number,
                    "title": title,
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"],
                }
            )

        chunks_with_metadeta = {"chunks": chunks, "text": result["text"]}

        with open(f"jsons/{audio[:-4]}.json", "w") as f:
            json.dump(chunks_with_metadeta, f)
