# RAG Based AI

A local, fully offline retrieval-augmented question answering (RAG) pipeline built over the Sigma Web Development course videos. It converts course videos to audio, transcribes them into timestamped chunks, generates embeddings for those chunks, and answers user questions by retrieving the most relevant transcript segments and passing them to a local LLM — no external APIs, no cloud calls, everything runs on your own machine via [Ollama](https://ollama.com/).

## What the project does

1. Extracts MP3 audio from source course videos.
2. Transcribes each audio file into chunked JSON with timestamps (using Whisper).
3. Generates embeddings for every transcript chunk (using a local Ollama model).
4. Given a user question, finds the most relevant chunks via cosine similarity.
5. Sends only those retrieved chunks to a local generation model, which answers strictly from that context.

## Project layout

| Path                  | Purpose                                                                                              |
| --------------------- | ---------------------------------------------------------------------------------------------------- |
| `videos/`             | Input MP4 course videos (you provide these — see [Adding your own videos](#adding-your-own-videos)). |
| `audios/`             | MP3 files extracted from `videos/`. Generated, not committed.                                        |
| `jsons/`              | Transcript chunks (text + timestamps) for each video. Generated, not committed.                      |
| `embeddings.joblib`   | Cached dataframe of transcript chunks + their embeddings. Generated, not committed.                  |
| `mp4_to_mp3.py`       | Converts videos in `videos/` to MP3 files in `audios/`.                                              |
| `mp3_to_json.py`      | Transcribes each MP3 into chunked JSON using Whisper.                                                |
| `preprocess_json.py`  | Reads every JSON in `jsons/`, builds embeddings via Ollama, saves `embeddings.joblib`.               |
| `process_incoming.py` | Takes a user question, retrieves relevant chunks, and generates an answer.                           |
| `unused/stt.py`       | Older, standalone transcription experiment — not part of the active pipeline.                        |
| `prompt.txt`          | The exact prompt sent to the LLM on the last run. Overwritten each run.                              |
| `response.txt`        | The model's answer from the last run. Overwritten each run.                                          |
| `output.json`         | Intermediate/debug output from the pipeline. Overwritten each run.                                   |
| `requirements.txt`    | Python dependencies.                                                                                 |
| `.venv start.txt`     | Notes/commands for activating the virtual environment.                                               |

## Requirements

- Python 3.10+
- `ffmpeg` installed and available on your system `PATH`
- [Ollama](https://ollama.com/) installed and running locally (default: `http://localhost:11434`)
- A local embedding model pulled in Ollama: `bge-m3`
- A local generation model pulled in Ollama: `deepseek-r1:latest`
- Enough disk space for your source videos + extracted audio + transcripts (course videos can be large; budget several GB depending on course length)

## Install

Clone the repo, then set up a virtual environment and install dependencies:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Pull the required Ollama models:

```bash
ollama pull bge-m3
ollama pull deepseek-r1:latest
```

Confirm Ollama is running:

```bash
ollama serve
```

(If you see a "port already in use" error, Ollama is already running in the background — that's fine, you can skip this step.)

## Adding your own videos

This repo does **not** ship with the course videos themselves (see [Notes on content](#notes-on-content--licensing) below) — you provide your own.

1. Create a `videos/` folder in the project root if it doesn't already exist.
2. Add your MP4 files, named with a **leading tutorial number**, for example:
   ```
   videos/
     1_Introduction_to_Web_Development.mp4
     2_How_Websites_Work.mp4
     3_Installing_VS_Code.mp4
   ```
   The leading number is important — `mp3_to_json.py` and the retrieval step use it to label and cite the correct video/tutorial in answers.
3. Keep filenames free of special characters beyond underscores to avoid path issues on Windows.
4. There's no strict limit on number of videos, but more videos means more transcript chunks, which means larger embedding batches (see [Troubleshooting](#troubleshooting) if you hit request-size issues during embedding).

Once your videos are in place, run the pipeline in order (below).

## End-to-end workflow

Run the scripts in this order from the project root, with your virtual environment activated.

### 1. Convert videos to audio

```bash
python mp4_to_mp3.py
```

Reads every file in `videos/` and writes a matching MP3 to `audios/`.

### 2. Transcribe audio into JSON

```bash
python mp3_to_json.py
```

Uses Whisper to transcribe each MP3 in `audios/` into timestamped chunks, saved as JSON files in `jsons/` (one JSON file per video). Note: this currently translates Hindi audio to English — if your source videos are in a different language, you'll want to adjust the Whisper task/language setting in this script.

This step can take a while depending on video length and whether you're running Whisper on CPU or GPU.

### 3. Build embeddings

```bash
python preprocess_json.py
```

Reads every JSON file in `jsons/`, sends the chunk text to the local Ollama embedding model (`bge-m3`), and saves everything (text, timestamps, embeddings) into a single `embeddings.joblib` dataframe.

> **Note on large courses:** if you have a video with a very large number of chunks (roughly 500+), sending them all to Ollama in a single request can cause the embedding request to fail or the Ollama runner to crash partway through. See [Troubleshooting](#troubleshooting) below.

### 4. Ask questions

```bash
python process_incoming.py
```

You'll be prompted to enter a question. The script will:

1. Embed your question using `bge-m3`.
2. Compare it against all stored chunk embeddings via cosine similarity.
3. Select the top 5 most relevant chunks.
4. Build a prompt (saved to `prompt.txt`) instructing the model to answer only from those chunks, citing video number, title, and timestamp where possible.
5. Send the prompt to `deepseek-r1:latest` and save the answer to `response.txt`.

## Configuration

Model names and the Ollama endpoint are currently hardcoded in the scripts (`bge-m3` for embeddings, `deepseek-r1:latest` for generation, `http://localhost:11434`). If you want to use different models or point to a remote Ollama instance, update these values directly in `preprocess_json.py` and `process_incoming.py`. (If this project grows, moving these into a `.env` file or a `config.py` is a natural next step — the `.gitignore` already excludes `.env` in anticipation of this.)

## Troubleshooting

**Embedding requests fail with a `KeyError: 'embeddings'` or an Ollama connection-refused error mid-request.**
This usually means too many chunks are being sent to Ollama's `/api/embed` endpoint in a single request. Large batches (hundreds of texts at once) can overwhelm the model runner and cause it to crash mid-response, even on capable hardware. Fix: send the chunk texts to Ollama in smaller batches (e.g. 32 at a time) inside `create_embedding`, rather than one giant batch — the function's inputs/outputs stay the same, only the internal request pattern changes.

**Ollama seems unresponsive or `ollama serve` won't start.**
Check whether Ollama is already running in the background (system tray icon on Windows) — you'll get a "port already in use" error if so, which is expected. To see live logs, quit the background instance first, then run `ollama serve` in its own terminal.

**`ffmpeg` not found.**
Make sure `ffmpeg` is installed and its `bin` folder is on your system `PATH`. On Windows, `mp3_to_json.py` appends `C:\ffmpeg\bin` to `PATH` automatically, but you still need `ffmpeg` actually installed at that location (or adjust the path in the script to match your install location).

**`process_incoming.py` errors saying `embeddings.joblib` not found.**
Run `preprocess_json.py` first — `process_incoming.py` assumes the embeddings file already exists.

## Notes on content & licensing

- This project is built around the _Sigma Web Development_ course as an example use case, but no course video, audio, or transcript content is included in this repository — only the pipeline code.
- If you plan to share this repo publicly, do not commit `videos/`, `audios/`, `jsons/`, or `embeddings.joblib`, since these would contain (or be derived from) copyrighted course material. The included `.gitignore` already excludes these.
- Anyone using this project should supply their own video content, and should ensure they have the right to transcribe and process that content locally.

## Example output

After a question is processed, you'll have:

- `prompt.txt` — the exact prompt sent to the model, including the retrieved transcript chunks.
- `response.txt` — the model's final answer, including citations back to video number/title/timestamp where applicable.

## License

No license file is currently included in this repository. If you intend for others to use, modify, or redistribute this code, consider adding an open-source license (e.g. MIT) to the repo root.
