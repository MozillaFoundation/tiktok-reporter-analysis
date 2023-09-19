import os
from PIL import Image
import torch
import torchvision.transforms as transforms
import timm
import argparse
import pandas as pd

def load_checkpoint_and_predict(image_path, checkpoint_path):
    # Define the same transform as used in training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load the image
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension

    # Load the model
    model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=7)

    # Load the checkpoint
    model.load_state_dict(torch.load(checkpoint_path))

    # Check if CUDA is available and if so, use it
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)
        image = image.to(device)
    else:
        device = torch.device("cpu")

    # Set the model to evaluation mode
    model.eval()

    # Make a prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return predicted.item()

def main():
    parser = argparse.ArgumentParser(description='Analyze images in a directory')
    parser.add_argument('image_dir', type=str, help='Directory of images to analyze')
    parser.add_argument('checkpoint_path', type=str, help='Path to model checkpoint')
    args = parser.parse_args()

    image_files = sorted([os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir) if os.path.isfile(os.path.join(args.image_dir, f))])

    predictions = []
    for image_path in image_files:
        prediction = load_checkpoint_and_predict(image_path, args.checkpoint_path)
        predictions.append((image_path, prediction))

    event_names = {
        0: 'Not TikTok',
        1: 'TikTok video player',
        2: 'Scrolling',
        3: 'Liked video player',
        4: 'Sharing',
        5: 'About this ad',
        6: 'Why recommended',
    }

    # Convert the list to a pandas DataFrame
    df = pd.DataFrame({'classification': [pred[1] for pred in predictions]})

    # Calculate the difference between consecutive rows
    df['change'] = df['classification'].diff()

    # Filter out rows where there's no change and reset the index
    change_df = df[df['change'].notna() & (df['change'] != 0)].reset_index()

    # Map classification to event name
    change_df['event_name'] = change_df['classification'].map(event_names)

    # The resulting DataFrame
    result_df = change_df[['index', 'event_name']].rename(columns={'index': 'frame'})

    print(result_df)

if __name__ == "__main__":
    main()
