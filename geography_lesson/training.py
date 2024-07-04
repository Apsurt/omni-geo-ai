
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import CountriesDataset
from device import get_device
from torchvision import transforms
from vit_pytorch.deepvit import DeepViT

device = get_device()

batch_size = 32

transform = transforms.Compose([
    transforms.ConvertImageDtype(torch.float32),
    ])

augmenter = transforms.AugMix()

training_set = CountriesDataset(train=True, transform=transform, augmenter=None, aug_p=0.8)
validation_set = CountriesDataset(train=False, transform=transform)
training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False)

print(f"Images in training set: {len(training_set)}")
print(f"Images in validation set: {len(validation_set)}")

if len(training_set.label_dict) != len(validation_set.label_dict):
    raise ValueError("Classes in training set and validation set are different")

classes = list(training_set.label_dict.values())
print(f"{len(classes)} classes")

model = DeepViT(
    image_size = 512,
    patch_size = 32,
    num_classes = len(classes),
    dim = 512,
    depth = 8,
    heads = 6,
    mlp_dim = 1024,
    dropout = 0.1,
    emb_dropout = 0.1,
)


#model = ViT(
#    name = "L_32_imagenet1k",
#    pretrained = True,
#    image_size = 512,
#    patches = 16,
#    num_classes=len(classes),
#    dim = 1024,
#    num_layers = 24,
#    num_heads = 16,
#    ff_dim = 4096,
#    dropout_rate = 0.1,
#    attention_dropout_rate = 0.1,
#)

try:
    model.load_state_dict(torch.load("models/newest_model"))
    model.eval()
    print("Using saved model")
except (RuntimeError, FileNotFoundError):
    print("Saved model invalid")

model.to(device)

#teacher.to(device)
#
#distiller = DistillWrapper(
#    student = model,
#    teacher = teacher,
#    temperature = 3,
#    alpha = 0.5,
#    hard = False
#)
#
#distiller.to(device)

pp=0
for p in list(model.parameters()):
    nn=1
    for s in list(p.size()):
        nn = nn*s
    pp += nn

print(f"Number of parameters: {round(pp/1_000_000)}M")

loss_fn = torch.nn.CrossEntropyLoss()
loss_fn.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def train_one_epoch(epoch_index):
    last_loss = 0.
    sum_loss = 0.
    n = 0

    for i, data in enumerate(training_loader):
        inputs, labels = data
        optimizer.zero_grad()

        print("\tCalculating output")
        try:
            outputs = model(inputs.to(device))
        except torch.cuda.OutOfMemoryError as e:
            print(torch.cuda.memory_summary(device=None, abbreviated=False))
            raise e


        print("\tCalculating loss")
        loss = loss_fn(outputs, labels.to(device))

        print("\tOptimizing")
        loss.backward()
        optimizer.step()

        last_loss = loss
        print(f"batch {i+1} loss: {last_loss}")
        n+=1
        sum_loss += last_loss

    return sum_loss/n

avg_losses = []

epoch_number = 0
EPOCHS = 1
best_vloss = 1_000_000

for epoch in range(EPOCHS):
    print(f"EPOCH {epoch_number + 1}:")

    model.train(True)
    avg_loss = train_one_epoch(epoch_number)
    avg_losses.append(avg_loss.cpu().detach().numpy())

    running_vloss = 0.0
    model.eval()

    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            print(f"Validating {i+1}/{int(np.ceil(validation_set.n_samples/batch_size))}", end="\r")
            vinputs, vlabels = vdata
            voutputs = model(vinputs.to(device))
            vloss = loss_fn(voutputs, vlabels.to(device))
            running_vloss += vloss
        print()

    avg_vloss = running_vloss / (i + 1)
    print(f"LOSS train {avg_loss} valid {avg_vloss}")

    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = "models/newest_model"
        torch.save(model.state_dict(), model_path)

    epoch_number += 1

#accuracy test
heatmap_grid = np.zeros((len(classes), len(classes)), dtype=np.float32)
with torch.no_grad():
    total = 0
    correct = 0
    for i, vdata in enumerate(validation_loader):
        vinputs, vlabels = vdata
        voutputs = model(vinputs.to(device))
        preds = torch.argmax(torch.softmax(voutputs, 1), 1)
        for idx, pred in enumerate(preds):
            print(f"Guess: {validation_set.label_dict[pred.item()]}\nTrue:  {validation_set.label_dict[vlabels[idx].item()]}\n")
        results = preds == vlabels.to(device)
        total += len(results)
        correct += torch.sum(results)
        for idx in range(len(preds)):
            heatmap_grid[preds[idx]][[vlabels[idx]]] += 1

    acc = (correct/total).item()
    print(f"Accuracy after training: {round(acc*100, 2)}%")

x = list(range(EPOCHS))
plt.plot(x, avg_losses, "-")
plt.show()

f, ax = plt.subplots()

ax.imshow(heatmap_grid, cmap="RdPu")
ax.set_xticks(range(len(classes)))
ax.set_xticklabels(classes, rotation=90)
ax.xaxis.tick_top()
ax.set_xlabel("True")
ax.xaxis.set_label_position("top")
ax.set_yticks(range(len(classes)))
ax.set_yticklabels(classes)
ax.set_ylabel("Predict")
plt.show()
