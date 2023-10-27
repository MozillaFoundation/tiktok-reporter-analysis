import argparse

from .analyze_screen_recording import analyze_screen_recording
from .classify_reported import classify_reported
from .classify_videos import classify_videos
from .extract_frames import extract_frames_from_video
from .train import train


def analyze(video_path, frames_folder, checkpoint_path, results_path, testing, audio_path=None):
    if audio_path:
        classify_reported(video_path, audio_path, results_path, testing)
    else:
        # extract frames from videos
        extract_frames_from_video(video_path, frames_folder)

        # analyze screen recordings
        analyze_screen_recording(frames_folder, checkpoint_path, results_path)

        # classify videos
        classify_videos(frames_folder, results_path, testing)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="command")

    # create the parser for the "train" command
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--train_dir", help="path to the training data directory", default="./data/frames")
    train_parser.add_argument(
        "--labels_file", help="path to the labels file", default="./data/training_data/labels.txt"
    )
    train_parser.add_argument("--checkpoint_dir", help="path to the checkpoint directory", default="./data/checkpoints")

    # create the parser for the "analyze" command
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("video_path", help="path to the video file")
    analyze_parser.add_argument("--frames_folder", help="path to the extracted frames folder", default="./data/frames")
    analyze_parser.add_argument(
        "--checkpoint_path", help="path to the checkpoint file", default="./data/checkpoints/best_model.pth"
    )
    analyze_parser.add_argument("--results_path", help="path to the results folder", default="./data/results")
    analyze_parser.add_argument("--testing", help="test with smaller random model", action="store_true")
    analyze_parser.add_argument("--audio_path", help="path to the audio file")

    args = parser.parse_args()

    if args.command == "train":
        train(args.train_dir, args.labels_file, args.checkpoint_dir)
    elif args.command == "analyze":
        analyze(
            args.video_path, args.frames_folder, args.checkpoint_path, args.results_path, args.testing, args.audio_path
        )
