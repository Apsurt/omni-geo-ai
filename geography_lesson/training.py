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
import sys
import multiprocessing as mp
from functools import partial

from datasets import CountriesDataset
from device import get_device

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class ImageClassifier:
    def __init__(self, num_classes, image_size=256, patches=16, debug=False):
        self.device = get_device()
        self.debug = debug
        
        self.model = ViT('B_16',
                         image_size=image_size,
                         pretrained=True,
                         num_classes=num_classes)
        
        self.freeze_early_layers()

        self.model.float()
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.0001, weight_decay=0.01)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=10)
        self.scaler = GradScaler() if self.device.type == 'cuda' else None
    
    def print_model_params(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    def freeze_early_layers(self):
        if hasattr(self.model, 'patch_embedding'):
            for param in self.model.patch_embedding.parameters():
                param.requires_grad = False

        if hasattr(self.model, 'transformer'):
            transformer = self.model.transformer
            if hasattr(transformer, 'blocks'):
                num_blocks = len(transformer.blocks)
                for block in transformer.blocks[:num_blocks//2]:
                    for param in block.parameters():
                        param.requires_grad = False

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
            for batch_idx, (inputs, labels) in enumerate(t):
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
                
                if self.debug and batch_idx % 10 == 0:
                    print(f"\nDebug - Epoch: {epoch+1}, Batch: {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
                    print(f"Input shape: {inputs.shape}, Output shape: {outputs.shape}")
                    print(f"Labels: {labels[:5]}, Predictions: {outputs.argmax(1)[:5]}")
        
        return total_loss / len(dataloader)

    def validate(self, dataloader, epoch, num_epochs):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            with tqdm(dataloader, desc=f"Validation {epoch+1}/{num_epochs}", leave=False) as t:
                for batch_idx, (inputs, labels) in enumerate(t):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    total_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                    t.set_postfix(loss=f"{loss.item():.4f}")
                    
                    if self.debug and batch_idx % 10 == 0:
                        print(f"\nDebug - Validation - Epoch: {epoch+1}, Batch: {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
                        print(f"Accuracy: {correct/total:.4f}")
        
        accuracy = correct / total
        avg_loss = total_loss / len(dataloader)
        return avg_loss, accuracy

    def predict(self, input_tensor):
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor.to(self.device))
        return output

def get_data_loaders(batch_size=32, image_size=256, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    augmenter = transforms.AugMix()

    training_set = CountriesDataset(train=True, transform=transform, augmenter=augmenter, aug_p=0.8)
    validation_set = CountriesDataset(train=False, transform=transform)

    training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return training_loader, validation_loader, training_set.label_dict

def train(classifier, train_loader, val_loader, num_epochs=50, patience=5):
    best_vloss = float('inf')
    no_improve = 0
    train_losses = []
    val_losses = []
    val_accuracies = []

    total_start_time = time.time()

    try:
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            train_loss = classifier.train_epoch(train_loader, epoch, num_epochs)
            val_loss, val_accuracy = classifier.validate(val_loader, epoch, num_epochs)
            classifier.scheduler.step()

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

            if val_loss < best_vloss:
                best_vloss = val_loss
                classifier.save_model("models/best_model.pth")
                no_improve = 0
            else:
                no_improve += 1

            if no_improve == patience:
                print("Early stopping")
                break

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current model state...")
        classifier.save_model("models/interrupted_model.pth")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        classifier.save_model("models/error_model.pth")
    finally:
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

def run_training(num_classes, image_size, batch_size, epochs, debug=False):
    train_loader, val_loader, label_dict = get_data_loaders(batch_size, image_size)
    classes = list(label_dict.values())

    print(f"Number of classes: {num_classes}")
    print(f"Images in training set: {len(train_loader.dataset)}")
    print(f"Images in validation set: {len(val_loader.dataset)}")

    classifier = ImageClassifier(num_classes, image_size, 16, debug)
    classifier.print_model_params()
    classifier.load_model("models/newest_model")

    train_losses, val_losses, val_accuracies = train(classifier, train_loader, val_loader, epochs)
    plot_training_results(train_losses, val_losses, val_accuracies)
    create_confusion_matrix(classifier, val_loader, classes)

def main():
    try:
        batch_size = 32
        image_size = 256
        epochs = 10
        num_classes = 93
        debug = True

        mp.set_start_method('spawn', force=True)

        p = mp.Process(target=run_training, args=(num_classes, image_size, batch_size, epochs, debug))
        p.start()
        p.join()
        
    except KeyboardInterrupt:
        print("\nScript interrupted by user. Exiting gracefully...")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        print("Script execution completed.")

if __name__ == '__main__':
    main()