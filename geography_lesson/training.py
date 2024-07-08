import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
import time
from tqdm import tqdm, trange
from pytorch_pretrained_vit import ViT

from datasets import CountriesDataset
from device import get_device

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class ImageClassifier:
    def __init__(self, num_classes, image_size=256, patches=16):
        self.device = get_device()
        
        self.model = ViT('B_16',
                         image_size=image_size,
                         pretrained=True,
                         num_classes=num_classes,
                         patches=patches,)
        self.model.float()
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=10)
        self.scaler = GradScaler() if self.device.type == 'cuda' else None
    
    def print_model_params(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    def load_model(self, path):
        try:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model.eval()
            print(f"Model loaded from {path}")
        except (RuntimeError, FileNotFoundError):
            print("No valid saved model found. Starting from scratch.")

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def train_epoch(self, dataloader, epoch, num_epochs):
        self.model.train()
        total_loss = 0.0
        
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False) as t:
            for inputs, labels in t:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                if self.device.type == 'cuda':
                    with autocast():
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                
                total_loss += loss.item()
                t.set_postfix(loss=f"{loss.item():.4f}")
        
        return total_loss / len(dataloader)

    def validate(self, dataloader, epoch, num_epochs):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            with tqdm(dataloader, desc=f"Validation {epoch+1}/{num_epochs}", leave=False) as t:
                for inputs, labels in t:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    total_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                    t.set_postfix(loss=f"{loss.item():.4f}")
        
        accuracy = correct / total
        avg_loss = total_loss / len(dataloader)
        return avg_loss, accuracy

    def predict(self, input_tensor):
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor.to(self.device))
        return output

def get_data_loaders(batch_size=32, image_size=256):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    augmenter = transforms.AugMix()

    training_set = CountriesDataset(train=True, transform=transform, augmenter=augmenter, aug_p=0.8)
    validation_set = CountriesDataset(train=False, transform=transform)

    training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    return training_loader, validation_loader, training_set.label_dict

def train(classifier, train_loader, val_loader, num_epochs=50, patience=5):
    best_vloss = float('inf')
    no_improve = 0
    train_losses = []
    val_losses = []
    val_accuracies = []

    total_start_time = time.time()

    with trange(num_epochs, desc="Training Progress") as pbar:
        for epoch in pbar:
            epoch_start_time = time.time()
            
            train_loss = classifier.train_epoch(train_loader, epoch, num_epochs)
            val_loss, val_accuracy = classifier.validate(val_loader, epoch, num_epochs)
            classifier.scheduler.step()

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            epoch_time = time.time() - epoch_start_time
            total_time = time.time() - total_start_time
            avg_epoch_time = total_time / (epoch + 1)
            estimated_time_left = avg_epoch_time * (num_epochs - epoch - 1)

            pbar.set_postfix({
                'Train Loss': f"{train_loss:.4f}",
                'Val Loss': f"{val_loss:.4f}",
                'Val Acc': f"{val_accuracy:.4f}",
                'Epoch Time': f"{epoch_time:.2f}s",
                'ETA': f"{estimated_time_left:.2f}s"
            })

            if val_loss < best_vloss:
                best_vloss = val_loss
                classifier.save_model("models/best_model.pth")
                no_improve = 0
            else:
                no_improve += 1

            if no_improve == patience:
                print("Early stopping")
                break

    total_time = time.time() - total_start_time
    print(f"Total training time: {total_time:.2f}s")

    return train_losses, val_losses, val_accuracies

def plot_training_results(train_losses, val_losses, val_accuracies):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.legend()
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.tight_layout()
    plt.show()

def create_confusion_matrix(classifier, val_loader, classes):
    classifier.model.eval()
    heatmap_grid = np.zeros((len(classes), len(classes)), dtype=np.float32)

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = classifier.predict(inputs)
            preds = torch.argmax(torch.softmax(outputs, 1), 1)
            for pred, label in zip(preds, labels):
                heatmap_grid[pred][label] += 1

    plt.figure(figsize=(12, 10))
    plt.imshow(heatmap_grid, cmap="RdPu")
    plt.colorbar()
    plt.xticks(range(len(classes)), classes, rotation=90)
    plt.yticks(range(len(classes)), classes)
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

def main():
    batch_size = 32
    image_size = 256
    train_loader, val_loader, label_dict = get_data_loaders(batch_size, image_size)
    classes = list(label_dict.values())
    num_classes = len(classes)

    print(f"Number of classes: {num_classes}")
    print(f"Images in training set: {len(train_loader.dataset)}")
    print(f"Images in validation set: {len(val_loader.dataset)}")

    classifier = ImageClassifier(num_classes, image_size, 16)
    classifier.print_model_params()
    classifier.load_model("models/newest_model")

    train_losses, val_losses, val_accuracies = train(classifier, train_loader, val_loader)
    plot_training_results(train_losses, val_losses, val_accuracies)
    create_confusion_matrix(classifier, val_loader, classes)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    multiprocessing.set_start_method('spawn')
    main()