import argparse
import logging
import sys

from tiktok_reporter_analysis.classify_reported import classify_reported
from tiktok_reporter_analysis.classify_videos import classify_videos
from tiktok_reporter_analysis.extract_frames import extract_frames_from_video
from tiktok_reporter_analysis.multimodal import multi_modal_from_saved
from tiktok_reporter_analysis.render_output import generate_html_report
from tiktok_reporter_analysis.train import train
from tiktok_reporter_analysis.util import load_descriptions

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
        default="./data/screen_recordings",
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
    analyze_parser.add_argument("--multimodal", help="run multimodal analysis", action="store_true")
    analyze_parser.add_argument(
        "--prompt_file", help="Prompt to use", default="tiktok_reporter_analysis/prompts/gpt_prompt.txt"
    )
    analyze_parser.add_argument("--backend", help="Backend to use (ollama, openai, or gemini)", default="ollama")
    analyze_parser.add_argument("--model", help="Model to use", default="")
    analyze_parser.add_argument("--context", help="Context length to use for ollama", default="")
    analyze_parser.add_argument("--twopass", help="Use two pass approach", action="store_true")
    analyze_parser.add_argument(
        "--modality_image", help="Specify the number of frames to use from each video", type=int, default=0
    )
    analyze_parser.add_argument(
        "--modality_text", help="Specify whether to include a transcript or not", action="store_true"
    )
    analyze_parser.add_argument(
        "--modality_video", help="Specify whether to include video file or not", action="store_true"
    )

    # create the parser for the "reported" command
    reported_parser = subparsers.add_parser("analyze_reported")
    reported_parser.add_argument("video_path", help="path to the video file or folder containing multiple videos")
    reported_parser.add_argument("--results_path", help="path to the results folder", default="./data/results")
    reported_parser.add_argument(
        "--nomultimodal", help="do not run multimodal analysis", action="store_false", dest="multimodal", default=True
    )
    reported_parser.add_argument("--prompt_file", help="Prompt to use", default="")
    reported_parser.add_argument("--fs_example_file", help="Few-shot examples to use", default="")
    reported_parser.add_argument("--backend", help="Backend to use (ollama, openai, or gemini)", default="ollama")
    reported_parser.add_argument("--model", help="Model to use", default="")
    reported_parser.add_argument("--context", help="Context length to use for ollama", default="")
    reported_parser.add_argument("--twopass", help="Use two pass approach", action="store_true")
    reported_parser.add_argument(
        "--modality_image", help="Specify the number of frames to use from each video", type=int, default=0
    )
    reported_parser.add_argument(
        "--modality_text", help="Specify whether to include a transcript or not", action="store_true"
    )
    reported_parser.add_argument(
        "--modality_video", help="Specify whether to include video file or not", action="store_true"
    )

    # create the parser for the "multimodal" command
    multimodal_parser = subparsers.add_parser("analyze_multimodal")
    multimodal_parser.add_argument("--results_path", help="path to the results folder", default="./data/results")
    multimodal_parser.add_argument("--prompt_file", help="Prompt to use")
    multimodal_parser.add_argument("--fs_example_file", help="Few-shot examples to use", default="")
    multimodal_parser.add_argument("--backend", help="Backend to use (ollama, lmstudio, openai, or gemini)", default="ollama")
    multimodal_parser.add_argument("--model", help="Model to use", default="")
    multimodal_parser.add_argument("--context", help="Context length to use for ollama", default="")
    multimodal_parser.add_argument("--twopass", help="Use two pass approach", action="store_true")
    multimodal_parser.add_argument(
        "--modality_image", help="Specify the number of frames to use from each video", type=int, default=0
    )
    multimodal_parser.add_argument(
        "--modality_text", help="Specify whether to include a transcript or not", action="store_true"
    )
    multimodal_parser.add_argument(
        "--modality_video", help="Specify whether to include video file or not", action="store_true"
    )

    # create the parser for the "report" command
    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--results_path", help="path to the results folder", default="./data/results")

    # create the parser for the "extract" command
    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--frames_path", help="path to the frames folder", default="./data/frames")
    extract_parser.add_argument("--video_path", help="path to the video file")

    load_parser = subparsers.add_parser("load_descriptions")
    load_parser.add_argument(
        "--descriptions_path",
        help="path to the descriptions parquet",
        default="data/results/video_descriptions.parquet",
    )
    load_parser.add_argument(
        "--sheet_id", help="ID of Google sheet to load to", default="1idnaMs-9k7adGO1kIOeu5wR8wwkkmclQ7LjF8y4NAZE"
    )
    load_parser.add_argument("--current_model", help="Model used to generate descriptions")

    args = parser.parse_args()

    if args.command == "train":
        train(args.frames_dir, args.recordings_dir, args.labels_file, args.checkpoint_dir)
    elif args.command == "analyze_screen_recording":
        classify_videos(
            args.video_path,
            args.checkpoint_path,
            args.prompt_file,
            args.backend,
            args.model,
            args.context,
            args.modality_image,
            args.modality_text,
            args.modality_video,
            args.results_path,
            args.multimodal,
        )
    elif args.command == "analyze_reported":
        classify_reported(
            args.video_path,
            args.results_path,
            args.prompt_file,
            args.fs_example_file,
            args.backend,
            args.model,
            args.context,
            args.modality_image,
            args.modality_text,
            args.modality_video,
            args.multimodal,
            args.twopass,
        )
    elif args.command == "analyze_multimodal":
        multi_modal_from_saved(
            args.results_path,
            args.prompt_file,
            args.fs_example_file,
            args.backend,
            args.model,
            args.context,
            args.modality_image,
            args.modality_text,
            args.modality_video,
            args.twopass,
        )
    elif args.command == "report":
        generate_html_report(args.results_path)
    elif args.command == "extract":
        extract_frames_from_video(args.video_path, args.frames_path)
    elif args.command == "load_descriptions":
        load_descriptions(args.descriptions_path, args.sheet_id, args.current_model)
    else:
        logger.error("Invalid command")
