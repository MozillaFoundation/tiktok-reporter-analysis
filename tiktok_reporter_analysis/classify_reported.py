import argparse
from tempfile import NamedTemporaryFile

import numpy as np
import whisper
from moviepy.editor import VideoFileClip
from PIL import Image

from .common import extract_frames, multi_modal_analysis


def extract_frames_n_audio(video_path, audio_path, num_frames=2):
    clip = VideoFileClip(video_path)
    audio = clip.audio
    audio.write_audiofile(audio_path)

    n_frames_in_video = int(clip.fps * clip.duration)
    frame_timestamps = np.linspace(0, clip.duration, n_frames_in_video)
    selected_frames = extract_frames(frame_timestamps)

    frames = {
        np.where(frame_timestamps == time)[0][0]: Image.fromarray(clip.get_frame(time)) for time in selected_frames
    }

    return frames


def extract_transcript(audio_path):
    whisper_model = whisper.load_model("base")
    print(whisper_model.device)
    transcript = whisper_model.transcribe(audio_path)
    return transcript


def classify_reported(video_path, results_path, testing=False):
    with NamedTemporaryFile(suffix=".wav") as tmpfile:
        frames = extract_frames_n_audio(video_path, tmpfile.name)
        transcript = extract_transcript(tmpfile.name)

    print(transcript["text"])

    multi_modal_analysis({0: frames}, results_path, transcript=transcript, testing=testing)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract transcript and frames from a video.")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument("audio_path", type=str, help="Path to the audio track")
    parser.add_argument("results_path", type=str, help="Path to results directory")

    args = parser.parse_args()

    classify_reported(args.video_path, args.audio_path)