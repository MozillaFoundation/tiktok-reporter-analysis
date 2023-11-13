import argparse

from .classify_reported import classify_reported
from .classify_videos import classify_videos
from .render_output import generate_html_report
from .train import train

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

    args = parser.parse_args()

    if args.command == "train":
        train(args.train_dir, args.labels_file, args.checkpoint_dir)
    elif args.command == "analyze_screen_recording":
        classify_videos(args.video_path, args.checkpoint_path, args.results_path, args.testing)
    elif args.command == "analyze_reported":
        classify_reported(args.video_path, args.results_path, args.testing)
    elif args.command == "report":
        generate_html_report(args.results_path)
    else:
        print("Invalid command")
