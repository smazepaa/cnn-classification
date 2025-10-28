import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from timm import create_model
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import pandas as pd
import os

from resnet import CustomImageDataset, preprocess_dataframe, CustomResidualBlock

BASE_DIR = "./cats-and-dogs-plus-plus/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENSEMBLE_SAVE_PATH = "resnet_ensemble_best.pth"

class TestImageDataset(torch.utils.data.Dataset):
    def __init__(self, test_df, image_dir, transform=None):
        self.test_df = test_df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.test_df)
    
    def __getitem__(self, idx):
        image_id = self.test_df.iloc[idx]['image_id']
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")

        from PIL import Image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, image_id

def create_stratified_split(df, test_size=0.2, random_state=42):
    label_cols = ['Biatrix', 'Cat', 'Dog', 'Moris', 'Motya']
    df['label_count'] = df[label_cols].sum(axis=1)

    def get_label_group(count):
        if count == 0:
            return '0_labels'
        elif count == 1:
            return '1_label'
        else:
            return '2+_labels'
    
    df['label_group'] = df['label_count'].apply(get_label_group)
    df['stratify_key'] = df['label_group'] + '_' + df['label_combo']

    train_df, val_df = train_test_split(
        df, 
        test_size=test_size, 
        stratify=df['stratify_key'], 
        random_state=random_state
    )

    print("\n📊 Label Distribution:")
    print("Training set:")
    print(train_df['label_group'].value_counts().sort_index())
    print("\nValidation set:")
    print(val_df['label_group'].value_counts().sort_index())
    
    return train_df, val_df

def create_resnet_model(model_name='resnet50', use_custom_block=True):
    model = create_model(model_name, pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    feature_dim = model.fc.in_features if hasattr(model, 'fc') else 2048
    model.fc = nn.Linear(feature_dim, 5)  # 5 classes

    if use_custom_block:
        model.custom_block = CustomResidualBlock(feature_dim)

        def new_forward(x):
            features = model.forward_features(x)
            features = model.custom_block(features)
            pooled = model.global_pool(features)
            return model.fc(pooled.flatten(1))

        model.forward = new_forward

    return model.to(DEVICE)

def load_existing_model():
    model = create_resnet_model('resnet50', use_custom_block=True)
    try:
        state_dict = torch.load('resnet50_multilabel_best.pth', map_location=DEVICE)
        model.load_state_dict(state_dict, strict=False)
        print("✅ Loaded existing ResNet50 model")
    except Exception as e:
        print(f"⚠️ Could not load existing model: {e}")
    return model

class ResNetEnsemble:
    def __init__(self, models):
        self.models = models
        for model in self.models:
            model.eval()
    
    def save_ensemble(self, filepath):
        ensemble_data = {
            'num_models': len(self.models),
            'model_states': [model.state_dict() for model in self.models],
            'model_architecture': 'resnet50_with_custom_block'
        }
        torch.save(ensemble_data, filepath)
        print(f"✅ Ensemble saved to {filepath}")
    
    @classmethod
    def load_ensemble(cls, filepath):
        ensemble_data = torch.load(filepath, map_location=DEVICE)

        models = []
        for state_dict in ensemble_data['model_states']:
            model = create_resnet_model('resnet50', use_custom_block=True)
            model.load_state_dict(state_dict, strict=False)
            models.append(model)
        
        print(f"✅ Loaded ensemble with {len(models)} models from {filepath}")
        return cls(models)
    
    def predict(self, dataloader, threshold=0.5):
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc="Ensemble Prediction"):
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
            for inputs, image_ids in tqdm(dataloader, desc="Test Inference"):
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

        class_names = ['Biatrix', 'Cat', 'Dog', 'Moris', 'Motya']
        exact_match_acc = np.mean(np.all(predictions == true_labels, axis=1))

        per_class_f1 = []
        per_class_precision = []
        per_class_recall = []
        
        for i in range(5):
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
            'per_class_f1': dict(zip(class_names, per_class_f1)),
            'per_class_precision': dict(zip(class_names, per_class_precision)),
            'per_class_recall': dict(zip(class_names, per_class_recall)),
            'num_models': len(self.models)
        }


def generate_predictions(model_path=ENSEMBLE_SAVE_PATH, output_path="ensemble_submission.csv"):
    print("Loading ensemble model...")
    
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found. Train first.")
        return
    
    ensemble = ResNetEnsemble.load_ensemble(model_path)
    test_df = pd.read_csv(BASE_DIR + "test.csv")
    print(f"Test dataset size: {len(test_df)} images")

    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = TestImageDataset(test_df, BASE_DIR + "test", test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    print(f"Generating predictions for {len(test_dataset)} images...")
    predictions, probabilities, image_ids = ensemble.predict_test(test_loader, threshold=0.5)
    
    # Create submission dataframe with correct column order
    # Model was trained with: ['Biatrix', 'Cat', 'Dog', 'Moris', 'Motya'] (indices 0,1,2,3,4)
    # But submission needs: ['Cat', 'Dog', 'Moris', 'Motya', 'Biatrix']
    model_class_order = ['Biatrix', 'Cat', 'Dog', 'Moris', 'Motya']
    submission_class_order = ['Cat', 'Dog', 'Moris', 'Motya', 'Biatrix']
    
    submission_df = pd.DataFrame({'image_id': image_ids})

    for submission_idx, class_name in enumerate(submission_class_order):
        model_idx = model_class_order.index(class_name)
        submission_df[class_name] = predictions[:, model_idx]

    for idx in range(len(submission_df)):
        if submission_df.loc[idx, 'Moris'] == 1 or submission_df.loc[idx, 'Motya'] == 1:
            submission_df.loc[idx, 'Cat'] = 1
        if submission_df.loc[idx, 'Biatrix'] == 1:
            submission_df.loc[idx, 'Dog'] = 1
    
    submission_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

    print(f"\nSubmission Statistics:")
    print(f"Total images: {len(submission_df)}")
    for class_name in submission_class_order:
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

    print("🚀 Creating ResNet Ensemble")
    full_df = preprocess_dataframe(BASE_DIR + "train.csv")
    train_df, val_df = create_stratified_split(full_df, test_size=0.2, random_state=42)
    
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_dataset = CustomImageDataset(val_df, BASE_DIR + "train", val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    models = [
        load_existing_model(),
        load_existing_model()
    ]
    
    ensemble = ResNetEnsemble(models)

    print(f"📊 Evaluating ensemble with {len(models)} models...")
    results = ensemble.evaluate(val_loader)
    
    print(f"\n✅ ENSEMBLE RESULTS:")
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

    ensemble.save_ensemble(ENSEMBLE_SAVE_PATH)

    print("\n🔄 Testing ensemble loading...")
    loaded_ensemble = ResNetEnsemble.load_ensemble(ENSEMBLE_SAVE_PATH)
    print("✅ Ensemble loading test successful!")
    
    return ensemble, results

if __name__ == '__main__':
    main('inference')
