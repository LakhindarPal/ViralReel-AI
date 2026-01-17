# --- IMPORTS ---
import torch
import numpy as np
import os, cv2, json, subprocess, re
from concurrent.futures import ThreadPoolExecutor

import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import whisperx
from google import genai
from google.genai import types
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
MAX_DURATION = 60

# --- AUTH ---
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)
except Exception as e:
    print(f"❌ Secret Error: {e}")


# --- CORE ENGINE ---


class ContentBrain:
    def __init__(self):
        print(f"🚀 Loading WhisperX on {DEVICE}...")
        self.model = whisperx.load_model(
            "large-v3-turbo", DEVICE, compute_type="float16", vad_method="silero"
        )
        self.align_model, self.metadata = whisperx.load_align_model(
            language_code="en", device=DEVICE
        )

    def transcribe(self, audio_path):
        result = self.model.transcribe(audio_path, batch_size=BATCH_SIZE)
        aligned = whisperx.align(
            result["segments"],
            self.align_model,
            self.metadata,
            audio_path,
            DEVICE,
            return_char_alignments=False,
        )
        return aligned

    def analyze(self, text):
        print("🧠 Thinking (Gemini 2.5 Flash)...")
        prompt = f"""
        Act as a viral content strategist. Analyze this transcript.
        Identify exactly 3 segments (30-50s duration) that work as standalone viral shorts.
        Return JSON ONLY: [{{"start_text": "unique start phrase", "end_text": "unique end phrase", "title": "Engaging Headline"}}]
        Transcript: {text[:150000]}...
        """
        try:
            res = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7, response_mime_type="application/json"
                ),
            )
            return json.loads(res.text)
        except:
            return []


class SmartCam:
    def __init__(self):
        base_opts = python.BaseOptions(model_asset_path="detector.tflite")
        opts = vision.FaceDetectorOptions(
            base_options=base_opts, min_detection_confidence=0.5
        )
        self.detector = vision.FaceDetector.create_from_options(opts)

    def get_face_center(self, frame):
        mp_img = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
        res = self.detector.detect(mp_img)
        if res.detections:
            largest = max(res.detections, key=lambda d: d.bounding_box.width)
            return (
                largest.bounding_box.origin_x + (largest.bounding_box.width / 2)
            ) / frame.shape[1]
        return 0.5


class Renderer:
    def __init__(self):
        self.font_path = "/usr/share/fonts/truetype/roboto/Roboto-Black.ttf"
        try:
            self.font_size = 75
            self.font = ImageFont.truetype(self.font_path, self.font_size)
            self.small_font = ImageFont.truetype(self.font_path, 40)
        except:
            self.font = ImageFont.load_default()
            self.small_font = ImageFont.load_default()

    def get_text_width(self, words, draw):
        total = 0
        for wd in words:
            bbox = draw.textbbox((0, 0), wd["word"], font=self.font)
            total += (bbox[2] - bbox[0]) + 20
        return total - 20

    def draw_wrapped_title(self, draw, title, w):
        words = title.upper().split()
        lines = []
        curr = []
        for word in words:
            test = curr + [word]
            bbox = draw.textbbox((0, 0), " ".join(test), font=self.small_font)
            if (bbox[2] - bbox[0]) < (w - 100):
                curr = test
            else:
                lines.append(" ".join(curr))
                curr = [word]
        lines.append(" ".join(curr))

        y = 80
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=self.small_font)
            x = (w - (bbox[2] - bbox[0])) // 2
            draw.text(
                (x, y),
                line,
                font=self.small_font,
                fill="#00ffff",
                stroke_width=3,
                stroke_fill="black",
            )
            y += 50

    def draw_karaoke(self, frame, words, time, title):
        draw = ImageDraw.Draw(frame)
        w, h = frame.size
        self.draw_wrapped_title(draw, title, w)

        active_idx = -1
        for i, word in enumerate(words):
            if word["start"] <= time <= word["end"] + 0.2:
                active_idx = i
                break

        if active_idx == -1:
            return frame

        chunk_size = 3
        start = (active_idx // chunk_size) * chunk_size
        end = min(len(words), start + chunk_size)
        visible = words[start:end]

        if self.get_text_width(visible, draw) > (w - 80):
            visible = [words[active_idx]]

        y = h - 450
        x = (w - self.get_text_width(visible, draw)) // 2

        for wd in visible:
            color = "#FFE135" if wd == words[active_idx] else "white"
            draw.text(
                (x, y),
                wd["word"],
                font=self.font,
                fill=color,
                stroke_width=5,
                stroke_fill="black",
            )
            bbox = draw.textbbox((0, 0), wd["word"], font=self.font)
            x += (bbox[2] - bbox[0]) + 20

        return frame


# --- WORKER FUNCTIONS ---
brain = ContentBrain()
cam = SmartCam()
renderer = Renderer()


def clean_filename(title):
    clean = re.sub(r"[^\w\s-]", "", title).strip().lower()
    return re.sub(r"[-\s]+", "_", clean)


def render_worker(args):
    i, hook, vid_path, all_words = args
    print(f"   ▶️ Processing: {hook['title']}")

    start_t, end_t = 0, 0
    s_txt, e_txt = hook["start_text"].strip(), hook["end_text"].strip()

    for w in all_words:
        if s_txt.startswith(w["word"].strip()):
            start_t = w["start"]
            break

    if start_t > 0:
        for w in all_words:
            if (
                w["start"] > start_t
                and w["end"] < (start_t + MAX_DURATION)
                and e_txt.endswith(w["word"].strip())
            ):
                end_t = w["end"]

    if end_t <= start_t:
        start_t = i * 60 + 60
        end_t = start_t + 50
    if (end_t - start_t) > MAX_DURATION:
        end_t = start_t + MAX_DURATION

    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_f, end_f = int(start_t * fps), int(end_t * fps)

    safe_title = clean_filename(hook["title"])
    out = f"{safe_title}.mp4"
    tmp = f"temp_{i}.mp4"

    writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (720, 1280))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    curr_f = start_f
    curr_center = 0.5

    while curr_f < end_f:
        ret, frame = cap.read()
        if not ret:
            break

        tgt_center = cam.get_face_center(frame)
        curr_center = curr_center * 0.9 + tgt_center * 0.1

        h, w, _ = frame.shape
        tgt_w = int(h * (9 / 16))
        if tgt_w % 2 != 0:
            tgt_w -= 1

        cx = int(curr_center * w)
        x1 = max(0, min(cx - (tgt_w // 2), w - tgt_w))

        final = cv2.resize(frame[0:h, x1 : x1 + tgt_w], (720, 1280))
        img = Image.fromarray(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
        img = renderer.draw_karaoke(img, all_words, curr_f / fps, hook["title"])
        writer.write(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
        curr_f += 1

    cap.release()
    writer.release()

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            tmp,
            "-ss",
            str(start_t),
            "-to",
            str(end_t),
            "-i",
            vid_path,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "ultrafast",
            "-c:a",
            "aac",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-shortest",
            out,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    return out


def run_pipeline(vid_file, url, progress=gr.Progress()):
    try:
        # 1. Input
        path = "input.mp4"
        if vid_file:
            path = vid_file
        elif url:
            progress(0.1, desc="Loading Video...")
            cmd = [
                "yt-dlp",
                "--add-header",
                "User-Agent:Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36",
                "-f",
                "bestvideo[ext=mp4][vcodec^=avc]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                "--force-overwrites",
                "-o",
                path,
                url,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                return [
                    gr.update(value=None),
                    gr.update(value=None),
                    gr.update(value=None),
                    f"❌ Download Failed:\n{result.stderr[-200:]}",
                ]
        else:
            return [
                gr.update(value=None),
                gr.update(value=None),
                gr.update(value=None),
                "❌ No input provided",
            ]

        if not os.path.exists(path):
            return [
                gr.update(value=None),
                gr.update(value=None),
                gr.update(value=None),
                "❌ Download failed (File not found)",
            ]

        # 2. Audio
        progress(0.2, desc="Analysing Audio...")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                path,
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                "temp.wav",
            ],
            stdout=subprocess.DEVNULL,
        )

        try:
            aligned = brain.transcribe("temp.wav")
        except Exception as e:
            return [
                gr.update(value=None),
                gr.update(value=None),
                gr.update(value=None),
                f"❌ Whisper Error: {e}",
            ]

        text = " ".join([s["text"] for s in aligned["segments"]])
        words = []
        for s in aligned["segments"]:
            words.extend(s["words"])

        # 3. AI
        progress(0.4, desc="Selecting Hooks...")
        hooks = brain.analyze(text)
        if not hooks:
            return [
                gr.update(value=None),
                gr.update(value=None),
                gr.update(value=None),
                "❌ No viral hooks found by AI",
            ]

        # 4. Render
        progress(0.6, desc="Rendering...")
        with ThreadPoolExecutor(max_workers=3) as exe:
            files = list(
                exe.map(
                    render_worker, [(i, h, path, words) for i, h in enumerate(hooks)]
                )
            )

        # 5. Dynamic Return (Update Labels)
        outputs = []
        for i, f in enumerate(files):
            # Dynamic Label Update: "Reel 1" -> "Actual Title"
            outputs.append(gr.update(value=f, label=hooks[i]["title"]))

        while len(outputs) < 3:
            outputs.append(gr.update(value=None, label="No Reel Generated"))

        outputs.append("✅ Processing Complete!")
        return outputs[0], outputs[1], outputs[2], outputs[3]

    except Exception as e:
        return [
            gr.update(value=None),
            gr.update(value=None),
            gr.update(value=None),
            f"❌ Critical Error:\n{str(e)}",
        ]


# --- GRADIO UI ---

with gr.Blocks(title="ViralReel AI") as app:
    gr.Markdown("# 🚀 ViralReel AI")
    gr.Markdown("Automated Short-Form Content Generator · Powered by Gemini & WhisperX")

    with gr.Column():
        with gr.Tabs():
            with gr.TabItem("Upload Video"):
                v_in = gr.Video(label="Source File")
            with gr.TabItem("Paste URL"):
                l_in = gr.Textbox(
                    label="YouTube / Drive Link", placeholder="https://..."
                )

        btn = gr.Button("Generate Reels", variant="primary")
        status = gr.Textbox(label="System Logs", interactive=False)

    with gr.Row():
        o1 = gr.Video(label="Pending Reel 1...")
        o2 = gr.Video(label="Pending Reel 2...")
        o3 = gr.Video(label="Pending Reel 3...")

    btn.click(run_pipeline, inputs=[v_in, l_in], outputs=[o1, o2, o3, status])

app.queue().launch(share=True, debug=True)
