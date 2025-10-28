import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from timm import create_model

import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.nn import functional as F


class CustomResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight_layer1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.weight_layer2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # F(x) path: x -> weight layer -> relu -> weight layer
        fx = self.weight_layer1(x)
        fx = self.relu(fx)
        fx = self.weight_layer2(fx)
        
        # F(x) + x (add identity/shortcut)
        out = fx + x
        
        # Final relu
        return self.relu(out)


BASE_DIR = "./cats-and-dogs-plus-plus/"
TRAIN_CSV_PATH = os.path.join(BASE_DIR, "train.csv")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train")
MODEL_SAVE_PATH = "resnet50_multilabel_best_2.pth"

NUM_TARGET_LABELS = 5
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = 'resnet50'

CLASS_NAMES = ['Biatrix', 'Cat', 'Dog', 'Moris', 'Motya', 'Other']
LABEL_COLUMNS = [c for c in CLASS_NAMES if c != 'Other']


def preprocess_dataframe(csv_path):
    df = pd.read_csv(csv_path)

    def get_label_combination(row):
        active_labels = sorted([col for col in LABEL_COLUMNS if row[col] == 1])
        if not active_labels:
            return 'None'
        return '+'.join(active_labels)

    df['label_combo'] = df.apply(get_label_combination, axis=1)
    return df


class CustomImageDataset(Dataset):
    def __init__(self, data_frame, img_dir, transform=None):
        self.df = data_frame.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

        self.label_tensors = {}
        for idx, row in self.df.iterrows():
            one_hot = torch.zeros(len(LABEL_COLUMNS), dtype=torch.float32)
            for i, label in enumerate(LABEL_COLUMNS):
                if row[label] == 1:
                    one_hot[i] = 1.0
            self.label_tensors[idx] = one_hot

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, 'image_id'] + '.jpg'
        img_path = os.path.join(self.img_dir, img_name)
        label = self.label_tensors[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Warning: Missing file {img_path}. Retrying...")
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            image = self.transform(image)

        return image, label

def log_params(model, epoch):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Epoch {epoch} - Trainable: {trainable:,} / Total: {total:,} ({trainable/total:.1%})")


def run_epoch(model, dataloader, criterion, optimizer=None, mode='train', epoch=0, log_gradcam=False):
    """Runs a single training or validation epoch for multi-label classification."""
    model.train() if mode == 'train' else model.eval()
    is_train = mode == 'train'

    running_loss, correct_preds_total = 0.0, 0
    total_samples = 0

    with torch.set_grad_enabled(is_train):
        for batch_idx, (inputs, labels) in enumerate(tqdm(dataloader, desc=mode.capitalize())):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            total_samples += inputs.size(0)

            if is_train:
                optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            # Multi-label prediction: Sigmoid + Threshold (0.5)
            probs = torch.sigmoid(outputs)
            predictions = (probs > 0.5).long()
            correct_predictions = (predictions == labels.long()).all(dim=1)
            correct_preds_total += correct_predictions.sum().item()

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_preds_total / total_samples
    return epoch_loss, epoch_acc


def main():
    full_df = preprocess_dataframe(TRAIN_CSV_PATH)

    train_df, val_df = train_test_split(
        full_df,
        test_size=0.2,
        stratify=full_df['label_combo'],
        random_state=42
    )

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CustomImageDataset(train_df, TRAIN_IMG_DIR, train_transforms)
    val_dataset = CustomImageDataset(val_df, TRAIN_IMG_DIR, val_transforms)

    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Training samples: {len(train_df)}. Validation samples: {len(val_df)}")
    print(f"Model: {MODEL_NAME}. Starting fine-tuning on {DEVICE}...")

    model = create_model(MODEL_NAME, pretrained=True).to(DEVICE)

    for param in model.parameters():
        param.requires_grad = False
    custom_block = CustomResidualBlock(2048)
    
    def new_forward(x):
        x = model.forward_features(x)
        x = custom_block(x)
        x = model.global_pool(x)
        x = model.fc(x.flatten(1))
        return x
    
    model.forward = new_forward
    model.custom_block = custom_block

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_TARGET_LABELS)

    for param in model.custom_block.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

    model = model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, 
                                        mode='train', epoch=epoch+1)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, mode='val', 
                                    epoch=epoch+1, log_gradcam=True)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}:")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        log_params(model, epoch+1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(">>> Model saved <<<")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', label='Train')
    plt.plot(val_losses, 'r-', label='Val')
    plt.title('Loss Curves')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, 'b-', label='Train')
    plt.plot(val_accs, 'r-', label='Val')
    plt.title('Accuracy Curves')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()

    print(f"\nTraining complete. Best val acc: {best_val_acc:.4f}")


if __name__ == '__main__':
    main()
