import argparse
import logging
import os

import pandas as pd
import timm
import torch
import torchvision.transforms as transforms

from tiktok_reporter_analysis.common import format_ms_timestamp, set_backend

logger = logging.getLogger(__name__)


def load_checkpoint(checkpoint_path, device):
    # Load the model
    model = timm.create_model("vit_large_patch16_224", pretrained=False, num_classes=8)

    # Load the checkpoint
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    model = model.to(device)

    # Set the model to evaluation mode
    model.eval()

    return model


def predict(image, model, device):
    # Define the same transform as used in training
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    # Load the image
    image = image.convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    # Make a prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return predicted.item()


def analyze_screen_recording(frames_dataframe, results_path, checkpoint_path):
    logger.info("Analyzing screen recording frames")
    device = set_backend()
    model = load_checkpoint(checkpoint_path, device)
    frames_to_timestamps = frames_dataframe.set_index("frame")["timestamp"].to_dict()

    logger.info("Predicting frame classifications")
    predictions = []
    for _, row in frames_dataframe.iterrows():
        prediction = predict(row["image"], model, device)
        predictions.append((row["frame"], prediction))
    logger.info("Frame classifications predicted")

    logger.info("Post-processing frame classifications")
    event_names = {
        0: "Not TikTok",
        1: "TikTok video player",
        2: "Scrolling",
        3: "Liked video player",
        4: "Sharing",
        5: "Why recommended",
        6: "Ad Player",
        7: "Other"
    }

    # Convert the list to a pandas DataFrame
    raw_predictions = pd.DataFrame({"classification": [pred[1] for pred in predictions]})
    raw_predictions["timestamp"] = raw_predictions.index.map(frames_to_timestamps)
    raw_predictions["timestamp"] = format_ms_timestamp(raw_predictions["timestamp"])

    # Identify single frame events and replace their classification with the previous frame's classification
    # Only if the classification of the current frame is also different from the previous frame's classification
    # And the frame numbers are adjacent
    single_frame_events = raw_predictions[
        (raw_predictions["classification"].shift(-1) != raw_predictions["classification"])
        & (raw_predictions["classification"].shift(1) != raw_predictions["classification"])
    ]
    for index in single_frame_events.index:
        if index > 0:  # Skip the first frame
            raw_predictions.loc[index, "classification"] = raw_predictions.loc[index - 1, "classification"]

    df = raw_predictions.copy(deep=True)
    # Calculate the difference between consecutive rows
    df["change"] = df["classification"].diff()

    # Filter out rows where there's no change and reset the index
    change_df = df[df["change"].notna() & (df["change"] != 0)].reset_index()

    # Map classification to event name
    change_df["event_name"] = change_df["classification"].map(event_names)

    # The resulting DataFrame
    result_df = change_df[["index", "timestamp", "event_name"]].rename(columns={"index": "frame"})

    # Save the DataFrame to a CSV file
    os.makedirs(results_path, exist_ok=True)
    result_df.to_csv(results_path + "/frame_event_data.csv", index=False)
    raw_predictions["event_name"] = raw_predictions["classification"].map(event_names)
    raw_predictions.reset_index().rename(columns={"index": "frame"})[
        ["frame", "timestamp", "classification", "event_name"]
    ].to_csv(results_path + "/frame_classification_data.csv", index=False)
    logger.info("Frame classifications post-processed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze images in a directory")
    parser.add_argument("image_dir", type=str, help="Directory of images to analyze")
    parser.add_argument("checkpoint_path", type=str, help="Path to model checkpoint")
    parser.add_argument("results_path", type=str, help="Path to results directory")
    args = parser.parse_args()

    analyze_screen_recording(args.image_dir, args.checkpoint_path)
