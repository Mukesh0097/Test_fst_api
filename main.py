from fastapi import FastAPI, HTTPException, Request, UploadFile, File,Form
from fastapi.responses import StreamingResponse, JSONResponse,FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from docx import Document
from pathlib import Path
import os
import tempfile
import yt_dlp
import logging
import re
import asyncio
import aiohttp
import os
import tempfile
import yt_dlp
import shutil
import uuid
import subprocess
from openai import OpenAI


load_dotenv()


logger = logging.getLogger(__name__)
app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_URL = "https://api.openai.com/v1/audio/transcriptions"

class DownloadRequest(BaseModel):
    url: str

client = OpenAI(api_key=OPENAI_API_KEY)

def sanitize_filename(name: str) -> str:
    return re.sub(r'[^\x00-\x7f]', '_', name)

def merge_results(results):
    merged_segments = []
    offset = 0.0

    for r in results:
        # r["segments"] is a list of segments
        for seg in r["segments"]:
            aligned_seg = dict(seg)  # shallow copy
            aligned_seg["start"] = seg["start"] + offset
            aligned_seg["end"]   = seg["end"]   + offset
            merged_segments.append(aligned_seg)

        # increase offset by the duration of this chunk
        offset += r.get("duration", 0)

    return {"segments": merged_segments}


async def transcribe_chunk(session, chunk_path: Path):
    try:
        with open(chunk_path, "rb") as f:
            form = aiohttp.FormData()
            form.add_field("file", f, filename=chunk_path.name, content_type="audio/wav")
            form.add_field("model", "whisper-1")  
            form.add_field("response_format", "verbose_json")  
            form.add_field("prompt", """You are a professional transcription agent. Your job is to transcribe an audio file into a complete, accurate, and timestamped dialogue script.

        Follow these STRICT RULES:

        1. Transcribe Every Word (No skipping)
        - Do not summarize or paraphrase.
        - Transcribe the full content, word-for-word.

        2. Timestamps Format: [HH:MM:SS]
        - Begin each new speaker segment with a timestamp in the format [HH:MM:SS].
        - Do NOT include milliseconds or invalid values (e.g. [00:03:94] is invalid).
        - Example: [00:01:05] Speaker 1: Welcome to the podcast.

        3. Speaker Labeling:
        - Use consistent speaker labels like "Speaker 1:", "Speaker 2:", etc., unless names are clearly mentioned in the audio.
        - Every speaker turn must have a new line and label.

        4. Google Docs Formatting:
        - Use hard line breaks (no `\n` or Markdown).
        - Do not use code blocks, tables, or special formatting.
        - Example format:
            [00:02:15] Speaker 1: This is a correct line.
            [00:02:22] Speaker 2: And this is the response.

        5. Accuracy Checks:
        After generating the transcript, ensure:
        - The transcript covers the entire audio duration.
        - Timestamps are in correct format and match audio flow.
        - No hallucinated content is added.
        - No formatting issues like `\n`, `[00:12:900]`, or missing speaker tags.

        6. If audio is unavailable:
        Respond only with:  
        **"Unable to access the audio. Please provide a valid input."**

        Repeat: DO NOT create placeholder or assumed transcripts.

        Failure to follow formatting will lead to rejection of your response.
        """)
     
            async with session.post(
                OPENAI_URL,
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                data=form
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise HTTPException(status_code=resp.status, detail=text)
                result = await resp.json()
                return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/api/download")
# async def download_audio(body: DownloadRequest, request: Request):
#     logger.info("Extracting audio info...")
#     try:
#         with tempfile.TemporaryDirectory() as temp_dir:
#             ydl_opts = {
#                 'format': 'bestaudio/best',
#                 'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),
#                 'noplaylist': True,
#                 'quiet': True,
#                 'cookiefile': '/etc/secrets/www.youtube.com_cookies.txt',
#                 'postprocessors': [{
#                     'key': 'FFmpegExtractAudio',
#                     'preferredcodec': 'mp3',
#                     'preferredquality': '192',
#                 }],
#             }

#             with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#                 info = ydl.extract_info(body.url, download=True)
#                 filename = ydl.prepare_filename(info)
#                 audio_file = os.path.splitext(filename)[0] + '.mp3'
#                 logger.info(f"Audio file path: {audio_file}")

#                 if not os.path.exists(audio_file):
#                     raise HTTPException(status_code=500, detail="Audio file not found")

#                 sanitized_title = sanitize_filename(info.get('title', 'audio'))

#                 return StreamingResponse(
#                     open(audio_file, "rb"),
#                     media_type="audio/mpeg",
#                     headers={
#                         "Content-Disposition": f'attachment; filename="{sanitized_title}.mp3"'
#                     }
#                 )
#     except Exception as e:
#         logger.error(f"Audio download failed: {e}")
#         raise HTTPException(status_code=500, detail="Audio download failed")


@app.post("/api/download")
async def download_audio(body: DownloadRequest, request: Request):
    logger.info("Extracting audio info...")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),
                'noplaylist': True,
                'quiet': True,
                'cookiefile': '/etc/secrets/www.youtube.com_cookies.txt',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(body.url, download=True)
                filename = ydl.prepare_filename(info)
                audio_file = os.path.splitext(filename)[0] + '.mp3'
                logger.info(f"Audio file path: {audio_file}")

                if not os.path.exists(audio_file):
                    raise HTTPException(status_code=500, detail="Audio file not found")

                sanitized_title = sanitize_filename(info.get('title', 'audio'))

                return FileResponse(
                    path=audio_file,
                    media_type="audio/mpeg",
                    filename=f"{sanitized_title}.mp3"
                )

    except Exception as e:
        logger.error(f"Audio download failed: {e}")
        raise HTTPException(status_code=500, detail="Audio download failed")


@app.post("/api/extract-docx")
async def extract_docx(file: UploadFile = File(...)):
    if not file.filename.endswith(".docx"):
        raise HTTPException(status_code=400, detail="Only .docx files are supported")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        doc = Document(tmp_path)
        os.remove(tmp_path)  # Clean up

        text = "\n".join([para.text for para in doc.paragraphs])
        return JSONResponse(content={"content": text})

    except Exception as e:
        logger.error(f"DOCX extraction failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to extract DOCX content")


@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")

    # Create temp working directory
    session_id = uuid.uuid4().hex
    work_dir = Path(f"tmp_{session_id}")
    chunks_dir = work_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    temp_file_path = work_dir / file.filename
    try:
        # Save uploaded file
        with open(temp_file_path, "wb") as f:
            f.write(await file.read())

        # Split into ~10 sec chunks using ffmpeg
        subprocess.run(
            [
                "ffmpeg", "-i", str(temp_file_path),
                "-f", "segment", "-segment_time", "600",
                "-c", "copy", f"{chunks_dir}/chunk_%03d.wav", "-y"
            ],
            check=True
        )

        # Get sorted list of chunk files
        chunk_files = sorted(chunks_dir.glob("chunk_*.wav"))
        if not chunk_files:
            raise HTTPException(status_code=400, detail="No audio chunks created")

        # Transcribe chunks in parallel
        async with aiohttp.ClientSession() as session:
            tasks = [transcribe_chunk(session, chunk) for chunk in chunk_files]
            results = await asyncio.gather(*tasks)
        
         # merge all segments into one single object
        merged = merge_results(results)

        return merged
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"FFmpeg error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup temp files
        shutil.rmtree(work_dir, ignore_errors=True)

    


