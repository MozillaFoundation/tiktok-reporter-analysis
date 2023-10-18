import argparse
import random
from moviepy.editor import VideoFileClip
import whisper
import numpy as np
import torch

def extract_frames(video_path, num_frames=2):
    clip = VideoFileClip(video_path)
    duration = clip.duration
    frames = []

    for _ in range(num_frames):
        time = random.uniform(0, duration)
        frames.append(clip.get_frame(time))

    return frames


def pydub_to_np(audio):
    """
    Converts pydub audio segment into np.float32 of shape [duration_in_seconds*sample_rate, channels],
    where each value is in range [-1.0, 1.0]. 
    Returns tuple (audio_np_array, sample_rate).
    """
    return np.array(audio.get_array_of_samples(), dtype=np.float32).reshape((-1, audio.channels)) / (
            1 << (8 * audio.sample_width - 1)), audio.frame_rate

def extract_transcript(audio_path):
    whisper_model = whisper.load_model("base")
    print(whisper_model.device)
    transcript = whisper_model.transcribe(audio_path)
    return transcript

def main():
    parser = argparse.ArgumentParser(description='Extract transcript and frames from a video.')
    parser.add_argument('video_path', type=str, help='Path to the video file')
    parser.add_argument('audio_path', type=str, help='Path to the audio track')

    args = parser.parse_args()

    transcript = extract_transcript(args.video_path)
    frames = extract_frames(args.video_path)

    print(transcript['text'])

if __name__ == "__main__":
    main()
