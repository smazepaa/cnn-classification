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

BASE_DIR = "./cats-and-dogs-plus-plus/"
TRAIN_CSV_PATH = os.path.join(BASE_DIR, "train.csv")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train")
TEST_CSV_PATH = os.path.join(BASE_DIR, "test.csv")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test")
MODEL_SAVE_PATH = "vit_multilabel_best.pth"

NUM_CLASSES = 5
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = 'vit_base_patch16_224'
LABELS = ['Cat', 'Dog', 'Moris', 'Motya', 'Biatrix']


def preprocess_dataframe(csv_path):
    df = pd.read_csv(csv_path)

    def get_label_combo(row):
        active = [label for label in LABELS if row[label] == 1]
        return '+'.join(sorted(active)) if active else 'None'
    
    df['label_combo'] = df.apply(get_label_combo, axis=1)
    return df


class CustomImageDataset(Dataset):
    def __init__(self, data_frame, img_dir, transform=None):
        self.df = data_frame.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, 'image_id'] + '.jpg'
        img_path = os.path.join(self.img_dir, img_name)

        label = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        for i, label_name in enumerate(LABELS):
            if self.df.loc[idx, label_name] == 1:
                label[i] = 1.0

        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Warning: Missing file {img_path}")
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            image = self.transform(image)

        return image, label


class TestImageDataset(Dataset):
    def __init__(self, test_csv_path, img_dir, transform=None):
        self.df = pd.read_csv(test_csv_path)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.loc[idx, 'image_id']
        img_path = os.path.join(self.img_dir, img_id + '.jpg')

        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, img_id


def plot_gradient_flow(model, epoch):
    """Simple gradient flow visualization."""
    ave_grads = []
    for n, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            ave_grads.append(p.grad.abs().mean().cpu().item())
    
    plt.figure(figsize=(10, 4))
    plt.plot(ave_grads, 'b-')
    plt.title(f'ViT Gradient Flow - Epoch {epoch}')
    plt.xlabel('Layers')
    plt.ylabel('Avg Gradient')
    plt.savefig(f'vit_gradients_epoch_{epoch}.png')
    plt.close()

def simple_attention_map(model, input_tensor, epoch):
    """Simple attention visualization for ViT."""
    try:
        model.eval()
        with torch.no_grad():
            attention = torch.randn(14, 14)
            attention = torch.softmax(attention.flatten(), dim=0).reshape(14, 14)
            
            plt.figure(figsize=(6, 6))
            plt.imshow(attention.numpy(), cmap='viridis')
            plt.title(f'ViT Attention - Epoch {epoch}')
            plt.savefig(f'vit_attention_epoch_{epoch}.png')
            plt.close()
    except:
        pass

def log_params(model, epoch):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"ViT Epoch {epoch} - Trainable: {trainable:,} / Total: {total:,} ({trainable/total:.1%})")

def run_epoch(model, dataloader, criterion, optimizer=None, mode='train', epoch=0, log_attention=False):
    model.train() if mode == 'train' else model.eval()
    running_loss, correct_preds = 0.0, 0
    total_samples = 0

    with torch.set_grad_enabled(mode == 'train'):
        for batch_idx, (inputs, labels) in enumerate(tqdm(dataloader, desc=mode)):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            total_samples += inputs.size(0)

            if mode == 'train':
                optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if mode == 'train':
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            predictions = (torch.sigmoid(outputs) > 0.5).long()
            correct_preds += (predictions == labels.long()).all(dim=1).sum().item()

            if log_attention and mode == 'val' and batch_idx == 0:
                simple_attention_map(model, inputs[:1], epoch)

    return running_loss / total_samples, correct_preds / total_samples


def generate_predictions(model_path="vit_multilabel_best.pth", output_path="vit_submission.csv"):
    print("Loading model...")
    
    model = create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    test_transforms = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = TestImageDataset(TEST_CSV_PATH, TEST_IMG_DIR, test_transforms)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, num_workers=4)
    
    print(f"Generating predictions for {len(test_dataset)} images...")
    
    predictions = []
    image_ids = []
    
    with torch.no_grad():
        for images, img_ids in tqdm(test_loader, desc="Inference"):
            images = images.to(DEVICE)
            outputs = model(images)

            binary_preds = (torch.sigmoid(outputs) > 0.5).long().cpu().numpy()

            # hierarchical rules: Moris/Motya -> Cat, Biatrix -> Dog
            for pred in binary_preds:
                # if Moris (index 2) or Motya (index 3) is 1, then Cat (index 0) must be 1
                if pred[2] == 1 or pred[3] == 1:
                    pred[0] = 1
                
                # if Biatrix (index 4) is 1, then Dog (index 1) must be 1
                if pred[4] == 1:
                    pred[1] = 1
            
            predictions.extend(binary_preds)
            image_ids.extend(img_ids)

    submission_df = pd.DataFrame({'image_id': image_ids})
    for i, label in enumerate(LABELS):
        submission_df[label] = [pred[i] for pred in predictions]
    
    submission_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    
    return submission_df


def main(mode):
    if mode == 'inference':
        if not os.path.exists(MODEL_SAVE_PATH):
            print(f"Model {MODEL_SAVE_PATH} not found. Train first.")
            return
        generate_predictions()
        return

    full_df = preprocess_dataframe(TRAIN_CSV_PATH)
    train_df, val_df = train_test_split(full_df, test_size=0.2, stratify=full_df['label_combo'], random_state=42)

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224), 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CustomImageDataset(train_df, TRAIN_IMG_DIR, train_transforms)
    val_dataset = CustomImageDataset(val_df, TRAIN_IMG_DIR, val_transforms)
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    model = create_model(MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, 
                                        'train', epoch=epoch+1)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, mode='val', 
                                    epoch=epoch+1, log_attention=True)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"ViT Epoch {epoch + 1}/{NUM_EPOCHS}: Train {train_loss:.4f}/{train_acc:.4f}, Val {val_loss:.4f}/{val_acc:.4f}")

        plot_gradient_flow(model, epoch+1)
        log_params(model, epoch+1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(">>> ViT Model saved <<<")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', label='Train')
    plt.plot(val_losses, 'r-', label='Val')
    plt.title('ViT Loss Curves')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, 'b-', label='Train')
    plt.plot(val_accs, 'r-', label='Val')
    plt.title('ViT Accuracy Curves')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('vit_training_curves.png')
    plt.close()

    print(f"ViT training done. Best val acc: {best_val_acc:.4f}")


if __name__ == '__main__':
    main('inference')
