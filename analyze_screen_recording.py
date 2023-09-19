import os
from PIL import Image
import torch
import torchvision.transforms as transforms
import timm
import argparse

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

    print(f"The predicted classes are: {predictions}")  

if __name__ == "__main__":
    main()
