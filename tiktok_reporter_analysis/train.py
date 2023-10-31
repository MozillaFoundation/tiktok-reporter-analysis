import json
import os

import timm
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

from .common import set_backend
from .extract_frames import extract_frames_from_video


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


def train(train_dir, labels_file, checkpoint_dir):
    # Load paths and labels
    image_folder = train_dir
    labels = []

    with open(labels_file, "r") as f:
        data = json.load(f)
        labels = generate_label_list(data[0]["events"])  # TEMP HACK

    screen_recordings_dir = os.path.join(os.path.dirname(labels_file), "screen_recordings")
    video_path = os.path.join(screen_recordings_dir, data[0]["filename"])
    print(video_path)
    extract_frames_from_video(video_path, image_folder)

    image_files = [
        os.path.join(image_folder, f) for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))
    ]
    print(f"There are {len(image_files)} files and {len(labels)} labels.")
    image_files.sort()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    full_dataset = CustomDataset(image_files, labels, transform=transform)

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    # full_loader = DataLoader(full_dataset, batch_size=32, shuffle=False)

    model = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=7)
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
        print(
            print(
                f"Epoch: {epoch+1}/{n_epochs}.. "
                f"Train Loss: {train_loss:.4f}, "
                f"Train Acc: {train_acc:.2f}%, "
                f"Test Loss: {test_loss:.4f}, "
                f"Test Acc: {test_acc:.2f}%"
            )
        )

        # Save the model checkpoint if it has the best test loss so far
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
            print(f"New best model saved to {checkpoint_dir}")


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
