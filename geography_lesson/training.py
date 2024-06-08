
import torch
import torchvision.transforms as transforms
from vit_pytorch.deepvit import DeepViT
from vit_pytorch.distill import DistillableViT, DistillWrapper
from pytorch_pretrained_vit import ViT
from datasets import CountriesDataset
import matplotlib.pyplot as plt

device = torch.device("mps")

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

batch_size = 64

training_set = CountriesDataset(train=True)
validation_set = CountriesDataset(train=True)
training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False)

print(f"Images in training set: {len(training_set)}")
print(f"Images in validation set: {len(validation_set)}")

classes = list(training_set.label_dict.values())
print(classes)
print(len(classes))

#model = DeepViT(
#    image_size = 512,
#    patch_size = 64,
#    num_classes = len(classes),
#    dim = 1024,
#    depth = 7,
#    heads = 15,
#    mlp_dim = 2048,
#    dropout = 0.1,
#    emb_dropout = 0.1
#)

model = DistillableViT(
    image_size = 512,
    patch_size = 32,
    num_classes = len(classes),
    dim = 1024,
    depth = 18,
    heads = 16,
    mlp_dim = 3072,
    dropout = 0.1,
    emb_dropout = 0.1
)

try:
    model.load_state_dict(torch.load("models/newest_model"))
    model.eval()
    print("Using saved model")
except (RuntimeError, FileNotFoundError):
    print("Saved model invalid")

teacher = ViT(
    name = "L_32_imagenet1k",
    pretrained = True,
    patches = 16,
    dim = 1024,
    ff_dim = 2048,
    num_heads = 8,
    num_layers = 6,
    attention_dropout_rate = 0.1,
    dropout_rate = 0.1,
    image_size = 512
)

distiller = DistillWrapper(
    student = model,
    teacher = teacher,
    temperature = 3,
    alpha = 0.5,
    hard = False
)


model.to(device)

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
        outputs = model(inputs.to(device))

        print("\tCalculating loss")
        loss = loss_fn(outputs, labels.to(device))

        print("\tOptimizing")
        loss.backward()
        optimizer.step()

        last_loss = loss
        print(f'batch {i+1} loss: {last_loss}')
        n+=1
        sum_loss += last_loss

    return sum_loss/n

avg_losses = []

epoch_number = 0
EPOCHS = 3
best_vloss = 1_000_000

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    model.train(True)
    avg_loss = train_one_epoch(epoch_number)
    avg_losses.append(avg_loss.cpu().detach().numpy())

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
        for idx, pred in enumerate(preds):
            print(f"Guess: {validation_set.label_dict[pred.item()]}\nTrue:  {validation_set.label_dict[vlabels[idx].item()]}\n")
        results = preds == vlabels.to(device)
        total += len(results)
        correct += torch.sum(results)
        for idx, res in enumerate(results):
            pass
    
    acc = (correct/total).item()
    print(f"Accuracy after training: {round(acc*100, 2)}%")

x = list(range(EPOCHS))

plt.plot(x, avg_losses, "-")
plt.show()