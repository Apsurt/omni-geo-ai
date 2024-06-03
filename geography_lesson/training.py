
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from vit_pytorch import ViT
from datasets import CountriesDataset

device = torch.device("cpu")

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

batch_size = 32

training_set = CountriesDataset()
validation_set = CountriesDataset()
training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False)

classes = list(training_set.label_dict.values())
print(classes)
print(len(classes))

model = ViT(
    image_size = 1024,
    patch_size = 32,
    num_classes = len(classes),
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1,
    channels=3
)

try:
    model.load_state_dict(torch.load("models/newest_model"))
    model.eval()
    print("Using saved model")
except:
    print("Saved model invalid")

model.to(device)

loss_fn = torch.nn.CrossEntropyLoss()
loss_fn.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def train_one_epoch(epoch_index):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(training_loader):
        inputs, labels = data
        optimizer.zero_grad()
        
        print("\tCalculating output")
        outputs = model(inputs.to(device))

        print("\tCalculating loss")
        loss = loss_fn(outputs, labels.to(device))
        
        print("\tOptimizing")
        loss.backward()
        optimizer.step()

        last_loss = loss
        print('batch {} loss: {}'.format(i + 1, last_loss))

    return last_loss


epoch_number = 0
EPOCHS = 3
best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    model.train(True)
    avg_loss = train_one_epoch(epoch_number)


    running_vloss = 0.0
    model.eval()

    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs.to(device))
            vloss = loss_fn(voutputs, vlabels.to(device))
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'models/newest_model'
        torch.save(model.state_dict(), model_path)

    epoch_number += 1

#accuracy test
with torch.no_grad():
    total = 0
    correct = 0
    for i, vdata in enumerate(validation_loader):
        vinputs, vlabels = vdata
        voutputs = model(vinputs.to(device))
        preds = torch.argmax(torch.softmax(voutputs, 1), 1)
        results = preds == vlabels.to(device)
        total += len(results)
        correct += torch.sum(results)
        for idx, res in enumerate(results):
            pass
    
    acc = correct/total
    print(f"Accuracy after training is: {acc*100}%")