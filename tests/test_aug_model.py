import torch
import torchvision.transforms as transforms
from vit_pytorch import ViT
from geography_lesson import CountriesDataset, get_device
import torchtest

device = get_device()

batch_size = 32

transform = transforms.Compose([
    transforms.ConvertImageDtype(torch.float32)
    ])

augmenter = transforms.AugMix()

augmented_dataset = CountriesDataset(train=True, transform=transform, augmenter=augmenter, aug_p=1)
augmented_loader = torch.utils.data.DataLoader(augmented_dataset, batch_size=batch_size, shuffle=True)

classes = list(augmented_dataset.label_dict.values())

model = ViT(image_size=512,
            patch_size=32,
            num_classes=len(classes),
            dim=512,
            depth=4,
            heads=8,
            mlp_dim=768)

model = torch.nn.Sequential(
    model,
    torch.nn.Softmax(1)
)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True)
batch = next(iter(augmented_loader))

def test_var_change():
    torchtest.assert_vars_change(
        model=model,
        loss_fn=loss_fn,
        optim=optimizer,
        batch=batch,
        device=device)

def test_full():
    torchtest.test_suite(
        model,
        loss_fn,
        optimizer,
        batch,
        output_range=(0,1),
        test_output_range=True,
        test_nan_vals=True,
        test_inf_vals=True,
        device=device
        )
