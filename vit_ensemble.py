import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from timm import create_model
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd
import os
from PIL import Image

BASE_DIR = "./cats-and-dogs-plus-plus/"
TRAIN_CSV_PATH = os.path.join(BASE_DIR, "train.csv")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train")
TEST_CSV_PATH = os.path.join(BASE_DIR, "test.csv")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENSEMBLE_SAVE_PATH = "vit_ensemble_best.pth"

MODEL_NAME = 'vit_base_patch16_224'
NUM_CLASSES = 5
LABELS = ['Cat', 'Dog', 'Moris', 'Motya', 'Biatrix']


def preprocess_dataframe(csv_path):
    df = pd.read_csv(csv_path)

    def get_label_combo(row):
        active = [label for label in LABELS if row[label] == 1]
        return '+'.join(sorted(active)) if active else 'None'
    
    df['label_combo'] = df.apply(get_label_combo, axis=1)
    return df


class CustomImageDataset(torch.utils.data.Dataset):
    
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


class TestImageDataset(torch.utils.data.Dataset):
    
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


def create_vit_model():
    model = create_model(MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES)
    return model.to(DEVICE)


def load_trained_vit_model(model_path):
    model = create_vit_model()
    try:
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print(f"✅ Loaded ViT model from {model_path}")
    except Exception as e:
        print(f"⚠️ Could not load model from {model_path}: {e}")
    return model


class ViTEnsemble:
    
    def __init__(self, models):
        self.models = models
        for model in self.models:
            model.eval()
    
    def save_ensemble(self, filepath):
        ensemble_data = {
            'num_models': len(self.models),
            'model_states': [model.state_dict() for model in self.models],
            'model_architecture': MODEL_NAME
        }
        torch.save(ensemble_data, filepath)
        print(f"✅ ViT Ensemble saved to {filepath}")
    
    @classmethod
    def load_ensemble(cls, filepath):
        ensemble_data = torch.load(filepath, map_location=DEVICE)
        
        models = []
        for state_dict in ensemble_data['model_states']:
            model = create_vit_model()
            model.load_state_dict(state_dict)
            models.append(model)
        
        print(f"✅ Loaded ViT ensemble with {len(models)} models from {filepath}")
        return cls(models)
    
    def predict(self, dataloader, threshold=0.5):
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc="ViT Ensemble Prediction"):
                inputs = inputs.to(DEVICE)
                
                model_probs = []
                for model in self.models:
                    outputs = model(inputs)
                    probs = torch.sigmoid(outputs)
                    model_probs.append(probs)
                
                ensemble_probs = torch.stack(model_probs).mean(dim=0)
                all_probs.append(ensemble_probs.cpu().numpy())
                all_labels.append(labels.numpy())
        
        probabilities = np.vstack(all_probs)
        true_labels = np.vstack(all_labels)
        predictions = (probabilities > threshold).astype(int)
        
        return predictions, probabilities, true_labels
    
    def predict_test(self, dataloader, threshold=0.5):
        all_probs = []
        all_image_ids = []
        
        with torch.no_grad():
            for inputs, image_ids in tqdm(dataloader, desc="ViT Test Inference"):
                inputs = inputs.to(DEVICE)
                
                model_probs = []
                for model in self.models:
                    outputs = model(inputs)
                    probs = torch.sigmoid(outputs)
                    model_probs.append(probs)
                
                ensemble_probs = torch.stack(model_probs).mean(dim=0)
                all_probs.append(ensemble_probs.cpu().numpy())
                all_image_ids.extend(image_ids)
        
        probabilities = np.vstack(all_probs)
        predictions = (probabilities > threshold).astype(int)
        
        return predictions, probabilities, all_image_ids
    
    def evaluate(self, dataloader):
        predictions, probabilities, true_labels = self.predict(dataloader)
        
        exact_match_acc = np.mean(np.all(predictions == true_labels, axis=1))
        
        per_class_f1 = []
        per_class_precision = []
        per_class_recall = []
        
        for i in range(NUM_CLASSES):
            f1 = f1_score(true_labels[:, i], predictions[:, i], zero_division=0)
            precision = precision_score(true_labels[:, i], predictions[:, i], zero_division=0)
            recall = recall_score(true_labels[:, i], predictions[:, i], zero_division=0)
            
            per_class_f1.append(f1)
            per_class_precision.append(precision)
            per_class_recall.append(recall)
        
        macro_f1 = np.mean(per_class_f1)
        macro_precision = np.mean(per_class_precision)
        macro_recall = np.mean(per_class_recall)
        
        micro_f1 = f1_score(true_labels.flatten(), predictions.flatten(), zero_division=0)
        micro_precision = precision_score(true_labels.flatten(), predictions.flatten(), zero_division=0)
        micro_recall = recall_score(true_labels.flatten(), predictions.flatten(), zero_division=0)
        
        hamming_loss = np.mean(predictions != true_labels)
        subset_accuracy = exact_match_acc
        
        return {
            'exact_match_accuracy': exact_match_acc,
            'subset_accuracy': subset_accuracy,
            'hamming_loss': hamming_loss,
            'macro_f1': macro_f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'micro_f1': micro_f1,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'per_class_f1': dict(zip(LABELS, per_class_f1)),
            'per_class_precision': dict(zip(LABELS, per_class_precision)),
            'per_class_recall': dict(zip(LABELS, per_class_recall)),
            'num_models': len(self.models)
        }


def generate_predictions(model_path=ENSEMBLE_SAVE_PATH, output_path="vit_ensemble_submission.csv"):
    print("Loading ViT ensemble model...")
    
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found. Train first.")
        return
    
    ensemble = ViTEnsemble.load_ensemble(model_path)
    
    test_df = pd.read_csv(TEST_CSV_PATH)
    print(f"Test dataset size: {len(test_df)} images")
    
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = TestImageDataset(TEST_CSV_PATH, TEST_IMG_DIR, test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    print(f"Generating predictions for {len(test_dataset)} images...")
    predictions, probabilities, image_ids = ensemble.predict_test(test_loader, threshold=0.5)
    
    submission_df = pd.DataFrame({'image_id': image_ids})
    
    for i, class_name in enumerate(LABELS):
        submission_df[class_name] = predictions[:, i]
    
    for idx in range(len(submission_df)):
        if submission_df.loc[idx, 'Moris'] == 1 or submission_df.loc[idx, 'Motya'] == 1:
            submission_df.loc[idx, 'Cat'] = 1
        
        if submission_df.loc[idx, 'Biatrix'] == 1:
            submission_df.loc[idx, 'Dog'] = 1
    
    submission_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    
    print(f"\nSubmission Statistics:")
    print(f"Total images: {len(submission_df)}")
    for class_name in LABELS:
        positive_count = submission_df[class_name].sum()
        percentage = (positive_count / len(submission_df)) * 100
        print(f"   {class_name}: {positive_count} positive predictions ({percentage:.1f}%)")
    
    print(f"\nSample predictions (first 5 rows):")
    print(submission_df.head())
    
    return submission_df


def main(mode):
    if mode == 'inference':
        if not os.path.exists(ENSEMBLE_SAVE_PATH):
            print(f"Model {ENSEMBLE_SAVE_PATH} not found. Train first.")
            return
        generate_predictions()
        return
    
    print("🚀 Creating ViT Ensemble")
    
    full_df = preprocess_dataframe(TRAIN_CSV_PATH)
    train_df, val_df = train_test_split(full_df, test_size=0.2, stratify=full_df['label_combo'], random_state=42)
    
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_dataset = CustomImageDataset(val_df, TRAIN_IMG_DIR, val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    models = [
        load_trained_vit_model('vit_multilabel_best.pth'),
        load_trained_vit_model('vit_multilabel_best.pth')
    ]
    
    ensemble = ViTEnsemble(models)
    
    print(f"📊 Evaluating ViT ensemble with {len(models)} models...")
    results = ensemble.evaluate(val_loader)
    
    print(f"\n✅ VIT ENSEMBLE RESULTS:")
    print(f"{'='*50}")
    print(f"Overall Metrics:")
    print(f"   Exact Match Accuracy: {results['exact_match_accuracy']:.4f}")
    print(f"   Subset Accuracy:      {results['subset_accuracy']:.4f}")
    print(f"   Hamming Loss:         {results['hamming_loss']:.4f}")
    
    print(f"\nMacro Averages:")
    print(f"   Macro F1:        {results['macro_f1']:.4f}")
    print(f"   Macro Precision: {results['macro_precision']:.4f}")
    print(f"   Macro Recall:    {results['macro_recall']:.4f}")
    
    print(f"\nMicro Averages:")
    print(f"   Micro F1:        {results['micro_f1']:.4f}")
    print(f"   Micro Precision: {results['micro_precision']:.4f}")
    print(f"   Micro Recall:    {results['micro_recall']:.4f}")
    
    print(f"\nPer-Class F1 Scores:")
    for class_name, f1_score in results['per_class_f1'].items():
        precision = results['per_class_precision'][class_name]
        recall = results['per_class_recall'][class_name]
        print(f"   {class_name:8}: F1={f1_score:.4f}, P={precision:.4f}, R={recall:.4f}")
    
    print(f"\nEnsemble Configuration:")
    print(f"   Number of Models: {results['num_models']}")
    print(f"   Voting Method: Soft Voting (average probabilities)")
    print(f"   Model Architecture: {MODEL_NAME}")
    
    ensemble.save_ensemble(ENSEMBLE_SAVE_PATH)
    
    print("\n🔄 Testing ensemble loading...")
    loaded_ensemble = ViTEnsemble.load_ensemble(ENSEMBLE_SAVE_PATH)
    
    print("✅ ViT Ensemble loading test successful!")
    
    return ensemble, results


if __name__ == '__main__':
    main('train')
