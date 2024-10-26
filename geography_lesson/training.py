#Imports
from typing_extensions import Never
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from torchvision.transforms import autoaugment, RandomErasing
import time
from tqdm import tqdm
#https://github.com/lukemelas/PyTorch-Pretrained-ViT
from pytorch_pretrained_vit import ViT
import multiprocessing as mp

from datasets import CountriesDataset
from device import get_device

#Sets up cuda (only for devices with nvidia gpus)
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

#Main class that encapsulates the model
class ImageClassifier:
    def __init__(self, num_classes, image_size=256, patches=16, debug=False):
        #num_classess should be 93 for our dataset but it should be set automatically
        
        #gets correct device automatically
        self.device = get_device()
        self.debug = debug
        
        #https://github.com/lukemelas/PyTorch-Pretrained-ViT
        #you can change some thing here but its kind of danger zone
        #changing this leads to bugs and pain...
        self.model = ViT('B_16',
                         image_size=image_size,
                         pretrained=False,
                         num_classes=num_classes)
        
        #blocks some parameters as model is pretrained
        #remove if training model that is not pretrained
        self.freeze_early_layers()

        self.model.float()
        #send model to gpu
        self.model.to(self.device)
        
        #loss function
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        
        #optimizer that takes only parameters that are not frozen
        #you can experiment with this
        #for me adamw was to slow but good learning progress
        #sgd + nestrov momentum is 2 twice as fast but 
        #doesnt seem to learn as good as adamw
        self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.002, weight_decay=0.01, momentum=0.9, nesterov=True)
        
        #this changes the learning rate variable
        #set T_max to number of epoch that you wish to train
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=3)
    
    def print_model_params(self):
        #debug function
        #prints number of total and frozen parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    def freeze_early_layers(self):
        #some dark magic to freeze params
        
        #this freezes embeddings
        #must be as with out it it doesnt make sense
        #to use pretrained model
        if hasattr(self.model, 'patch_embedding'):
            for param in self.model.patch_embedding.parameters():
                param.requires_grad = False

        if hasattr(self.model, 'transformer'):
            transformer = self.model.transformer
            if hasattr(transformer, 'blocks'):
                num_blocks = len(transformer.blocks)
                #if you want to change what parameters are frozen
                #change the slice of the `blocks` variable.
                #for now it freezes first half of transformer blocks.
                for block in transformer.blocks[:num_blocks//2]:
                    for param in block.parameters():
                        param.requires_grad = False

    def load_model(self, path):
        #loads model
        #hard coded to models/best_model.pth later in the code
        try:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model.eval()
            print(f"Model loaded from {path}")
        except (RuntimeError, FileNotFoundError) as e:
            print("No valid saved model found. Starting from scratch.")

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def train_epoch(self, dataloader, epoch, num_epochs):
        #most important function
        #training itself
        
        #sets model to the correct mode
        self.model.train()
        total_loss = 0.0
        
        #this is only for the nice progress bars
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False) as t:
            #loops through every batch produced by the data loader
            for batch_idx, (inputs, labels) in enumerate(t):
                #data is loaded to the gpu
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                #setting correct mode or something
                #must be
                self.optimizer.zero_grad()
                
                #now here is the real work
                
                #this takes input and passes it through the model
                #to produce outputs
                outputs = self.model(inputs)
                
                #then the outputs are used to calculate the loss function
                loss = self.criterion(outputs, labels)
                loss.backward()
                
                #here the optimizer changes 
                #the parameters that it got
                #in the __init__ so frozen params
                #dont change
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                #some loss calculation for debuging
                total_loss += loss.item()
                #updating the progress bar
                t.set_postfix(loss=f"{loss.item():.4f}")
                
                #debug info that prints every 10% of the epoch
                if self.debug and batch_idx % (len(dataloader)//10) == 0:
                    print(f"\nDebug - Epoch: {epoch+1}, Batch: {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
                    print(f"Input shape: {inputs.shape}, Output shape: {outputs.shape}")
                    print(f"Labels: {labels[:5]}, Predictions: {outputs.argmax(1)[:5]}")
        
        return total_loss / len(dataloader)

    def validate(self, dataloader, epoch, num_epochs):
        #similar to train epoch but doesnt change parameters
        #and uses different dataset that the model never
        #seen before in terms of training
        self.model.eval()
        total_loss = 0.0
        correct_top1 = 0
        correct_top5 = 0
        total = 0
        n_classes = dataloader.dataset.n_classes

        #confusion matrix
        heatmap_grid = np.zeros((n_classes, n_classes), dtype=np.float32)
        
        with torch.no_grad():
            with tqdm(dataloader, desc=f"Validation {epoch+1}/{num_epochs}", leave=False) as t:
                for batch_idx, (inputs, labels) in enumerate(t):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    preds = torch.argmax(torch.softmax(outputs, 1), 1)
                    for pred, label in zip(preds, labels):
                        heatmap_grid[pred][label] += 1
                    loss = self.criterion(outputs, labels)
                    total_loss += loss.item()


                    # Top-1 Accuracy
                    _, predicted = outputs.max(1)
                    correct_top1 += predicted.eq(labels).sum().item()

                    # Top-5 Accuracy
                    _, top5_predicted = outputs.topk(5, 1, largest=True, sorted=True)
                    correct_top5 += top5_predicted.eq(labels.view(-1, 1).expand_as(top5_predicted)).sum().item()

                    total += labels.size(0)
                    
                    top1_accuracy = correct_top1 / total
                    top5_accuracy = correct_top5 / total
                    t.set_postfix(loss=f"{loss.item():.4f}")
                    
                    if self.debug and batch_idx % (len(dataloader)//10) == 0:
                        print(f"\nDebug - Validation - Epoch: {epoch+1}, Batch: {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
                        print(f"Top-1 Accuracy: {top1_accuracy:.4f}, Top-5 Accuracy: {top5_accuracy:.4f}")
        
        final_top1_accuracy = correct_top1 / total
        final_top5_accuracy = correct_top5 / total
        avg_loss = total_loss / len(dataloader)
        return avg_loss, final_top1_accuracy, heatmap_grid

    def predict(self, input_tensor):
        #used to get output
        #useless when the model is not trained
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor.to(self.device))
        return output

def get_data_loaders(batch_size=32, image_size=256, num_workers=4, debug=False, preload=False, cache_size=1000):
    #this function is important
    #it initializes the dataloaders
    
    #these are transforms used for the training dataset
    #it includes augmentation
    #https://pytorch.org/vision/stable/transforms.html 
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        autoaugment.RandAugment(num_ops=2, magnitude=7),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    #these transforms dont augment the the data
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    #creates datasets defined in geography_lesson/gatasets.py
    preload = False
    training_set = CountriesDataset(train=True, transform=train_transform, debug=debug, preload=preload, cache_size=cache_size, max_class_file_count=700)
    validation_set = CountriesDataset(train=False, transform=val_transform, debug=debug, preload=preload, cache_size=5000)

    training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return training_loader, validation_loader, training_set.label_dict

def train(classifier, train_loader, val_loader, num_epochs=5, patience=2):
    #main training loop function
    best_vloss = float('inf')
    no_improve = 0
    train_losses = []
    val_losses = []
    val_accuracies = []

    total_start_time = time.time()
    
    n_classes = train_loader.dataset.n_classes
    heatmap_grid = np.zeros((n_classes, n_classes), dtype=np.float32)

    #main loop
    try:
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            train_loss = classifier.train_epoch(train_loader, epoch, num_epochs)
            val_loss, val_accuracy, heatmap_grid = classifier.validate(val_loader, epoch, num_epochs)
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
        
        if len(val_losses) == 0:
            val_loss, val_accuracy, heatmap_grid = classifier.validate(val_loader, 0, 1)

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current model state...")
        classifier.save_model("models/interrupted_model.pth")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        classifier.save_model("models/error_model.pth")
    finally:
        total_time = time.time() - total_start_time
        print(f"Total training time: {total_time:.2f}s")

    return train_losses, val_losses, val_accuracies, heatmap_grid

def plot_training_results(train_losses, val_losses, val_accuracies):
    #matplotlib plots
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

def create_confusion_matrix(heatmap_grid, classes):
    #creates confusion matrix
    #helps monitor convergence
    #diagonal line good - horizontal, vertical or noise bad
    #also cool because helps to check if the model confuses
    #similar countries (e.g. Austria and Slovakia)
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

def run_training(image_size, batch_size, epochs, debug=False, cache_size=1000):
    #This collects all important objects and
    #starts the main loop
    train_loader, val_loader, label_dict = get_data_loaders(batch_size, image_size, debug=debug, preload=True, cache_size=cache_size)
    classes = list(label_dict.values())
    num_classes = len(train_loader.dataset.label_dict)

    print(f"Number of classes: {num_classes}")
    print(f"Images in training set: {len(train_loader.dataset)}")
    print(f"Images in validation set: {len(val_loader.dataset)}")

    classifier = ImageClassifier(num_classes, image_size, 16, debug)
    classifier.print_model_params()
    classifier.load_model("models/best_model.pth")

    train_losses, val_losses, val_accuracies, heatmap_grid = train(classifier, train_loader, val_loader, epochs)
    create_confusion_matrix(heatmap_grid, classes)
    plot_training_results(train_losses, val_losses, val_accuracies)

def main():
    #entry point
    try:
        #parameters to change
        
        #better to use 2^n
        #choose between 8, 16, 32, 64
        #rule of thumb higher batch better learning lower worse
        #however when machine is not fast enough, because of bottelnecks
        #and time that cpu spends loading data to gpu it might be better
        #to use smaller batches
        #on apple m chips larger batches might be faster because of the 
        #unified memory between cpu and gpu
        batch_size = 16
        
        #this should always be 256 because thats the image size in dataset
        #however some experiments with smaller values might yield in interesting results
        #use 2^n values
        image_size = 256
        
        #for how long the simulation should go for
        #with current settings on my machine
        #one epoch takes ~40 min
        #remember to change T_Max in scheduler
        #after changing this
        #model is save whenever one epoch
        #finishes and the loss on validation is 
        #lower than before
        #setting more epoch might lead to better
        #results thanks to the scheduler but it
        #doesnt matter.
        #Just think of when you want to use your
        #computer again and the divide by 40 min
        #and set the `epochs` to the result
        #model is also saved on error and keyboard interrupt
        #so when you dont want to wait for the end of training
        #ctrl + c and rename interrupted_model.pth to best_model.pth
        epochs = 3
        
        #this dictates how many samples will be cached in memory
        #this setting depends on your memory
        #but i dont even know if it works so...
        cache_size = 30000
        
        #verbose mode
        debug = True

        #multiprocessing because keyboard interrupt was buggy
        #this repaired it
        mp.set_start_method('spawn', force=True)

        p = mp.Process(target=run_training, args=(image_size, batch_size, epochs, debug, cache_size))
        p.start()
        p.join()
        
    except KeyboardInterrupt:
        print("\nScript interrupted by user. Exiting...")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        print("Script execution completed.")

if __name__ == '__main__':
    main()