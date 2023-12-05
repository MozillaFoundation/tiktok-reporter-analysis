import argparse
import logging
import sys

from .classify_reported import classify_reported
from .classify_videos import classify_videos
from .extract_frames import extract_frames_from_video
from .render_output import generate_html_report
from .train import train

if __name__ == "__main__":
    logger = logging.getLogger("tiktok_reporter_analysis")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s:%(name)s:%(lineno)s - %(message)s")
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="command")

    # create the parser for the "train" command
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--frames_dir", help="path for the training data frames", default="./data/frames")
    train_parser.add_argument(
        "--recordings_dir",
        help="path for the training data screen recordings",
        default="./data/training_data/screen_recordings",
    )
    train_parser.add_argument(
        "--labels_file", help="path to the labels file", default="./data/training_data/labels.json"
    )
    train_parser.add_argument("--checkpoint_dir", help="path to the checkpoint directory", default="./data/checkpoints")

    # create the parser for the "analyze" command
    analyze_parser = subparsers.add_parser("analyze_screen_recording")
    analyze_parser.add_argument("video_path", help="path to the video file or folder containing multiple videos")
    analyze_parser.add_argument(
        "--checkpoint_path", help="path to the checkpoint file", default="./data/checkpoints/best_model.pth"
    )
    analyze_parser.add_argument("--results_path", help="path to the results folder", default="./data/results")
    analyze_parser.add_argument("--testing", help="test with smaller random model", action="store_true")

    # create the parser for the "reported" command
    reported_parser = subparsers.add_parser("analyze_reported")
    reported_parser.add_argument("video_path", help="path to the video file or folder containing multiple videos")
    reported_parser.add_argument("--results_path", help="path to the results folder", default="./data/results")
    reported_parser.add_argument("--testing", help="test with smaller random model", action="store_true")

    # create the parser for the "report" command
    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--results_path", help="path to the results folder", default="./data/results")

    # create the parser for the "extract" command
    report_parser = subparsers.add_parser("extract")
    report_parser.add_argument("--frames_path", help="path to the frames folder", default="./data/frames")
    report_parser.add_argument("--video_path", help="path to the video file")

    args = parser.parse_args()

    if args.command == "train":
        train(args.frames_dir, args.recordings_dir, args.labels_file, args.checkpoint_dir)
    elif args.command == "analyze_screen_recording":
        classify_videos(args.video_path, args.checkpoint_path, args.results_path, args.testing)
    elif args.command == "analyze_reported":
        classify_reported(args.video_path, args.results_path, args.testing)
    elif args.command == "report":
        generate_html_report(args.results_path)
    elif args.command == "extract":
        extract_frames_from_video(args.video_path, args.frames_path)
    else:
        logger.error("Invalid command")
