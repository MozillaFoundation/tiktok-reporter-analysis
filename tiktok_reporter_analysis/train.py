import json
import logging
import os

import timm
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

from tiktok_reporter_analysis.common import set_backend
from tiktok_reporter_analysis.common import extract_frames

logger = logging.getLogger(__name__)


def generate_label_list(transitions):
    # Convert keys to integers and find the max key to determine the size of the list
    transitions = {int(k): v for k, v in transitions.items()}
    max_index = max(transitions.keys())
    label_list = [0] * (max_index + 1)  # Initialize with zeros, or any default value

    previous_index = 0
    for index in sorted(transitions.keys()):
        label_list[previous_index:index] = [transitions[previous_index]] * (index - previous_index)
        previous_index = index

    # Set the label for the final segment
    label_list[previous_index:] = [transitions[previous_index]] * (max_index + 1 - previous_index)

    return label_list


class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label, img_path


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _ in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(train_loader), 100.0 * correct / total


def evaluate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, _ in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(test_loader), 100.0 * correct / total


def train(frames_dir, recordings_dir, labels_file, checkpoint_dir):
    # Load paths and labels
    image_folder = frames_dir
    labels = {"test": [], "train": []}
    image_files = {"test": [], "train": []}

    with open(labels_file, "r") as f:
        data = json.load(f)

    for i in range(len(data)):
        split = data[i]["split"]
        current_labels = generate_label_list(data[i]["events"])

        video_path = os.path.join(recordings_dir, data[i]["filename"])
        logger.info(f"Extracting frames from video: {video_path}")
        frames_path = os.path.join(image_folder, os.path.basename(video_path).split(".")[0])
        os.makedirs(frames_path, exist_ok=True)
        extract_frames(video_path, frames_path, save_frames=True)

        current_image_files = [
            os.path.join(frames_path, f)
            for f in os.listdir(frames_path)
            if os.path.isfile(os.path.join(frames_path, f)) and not f.endswith(".pkl")
        ]
        logger.info(f"There are {len(current_image_files)} files and {len(current_labels)} labels.")
        current_image_files.sort(key=lambda x: int("".join(filter(str.isdigit, os.path.basename(x)))))
        labels[split] += current_labels
        image_files[split] += current_image_files

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    logger.info(f"There train dataset has {len(image_files['train'])} files and {len(labels['train'])} labels.")
    logger.info(f"There test dataset has {len(image_files['test'])} files and {len(labels['test'])} labels.")
    train_dataset = CustomDataset(image_files["train"], labels["train"], transform=transform)
    test_dataset = CustomDataset(image_files["test"], labels["test"], transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    # full_loader = DataLoader(full_dataset, batch_size=32, shuffle=False)

    model = timm.create_model("vit_large_patch16_224", pretrained=True, num_classes=8)
    device = set_backend()
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    n_epochs = 10
    best_test_loss = float("inf")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    for epoch in range(n_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        logger.info(
            f"Epoch: {epoch+1}/{n_epochs}.. "
            f"Train Loss: {train_loss:.4f}, "
            f"Train Acc: {train_acc:.2f}%, "
            f"Test Loss: {test_loss:.4f}, "
            f"Test Acc: {test_acc:.2f}%"
        )

        # Save the model checkpoint if it has the best test loss so far
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
            logger.info(f"New best model saved to {checkpoint_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train the model with specified directory of training data and labels text file."
    )
    parser.add_argument("train_dir", type=str, help="Directory of training data")
    parser.add_argument("labels_file", type=str, help="File of labels")
    parser.add_argument("checkpoint_dir", type=str, help="Directory to save the best model checkpoint")
    args = parser.parse_args()

    train(args.train_dir, args.labels_file, args.checkpoint_dir)
