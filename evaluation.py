import torch

from torchvision.utils import make_grid
from torchvision.utils import save_image
from lib.image import unnormalize

def plot_images(model, dataset, device, num_images, filename):
    image, mask, gt = zip(*[dataset[i] for i in range(num_images)])
    image = torch.stack(image)
    mask = torch.stack(mask)
    gt = torch.stack(gt)
    with torch.no_grad():
        output, _ = model(image.to(device), mask.to(device))
    output = output.to(torch.device('cpu'))
    output_comp = mask*image + (1 - mask)*output

    grid = make_grid(
        torch.cat((unnormalize(image), mask, unnormalize(output),
                   unnormalize(output_comp), unnormalize(gt)), dim=0), nrow=num_images)

    save_image(grid, filename)

def eval_val_loss(model, dataset, criterion, device):
    val_loss, n_batch = 0, 0
    style_loss = 0

    model.eval()
    with torch.no_grad():
        for train_data in dataset:
            img, mask, gt = [x.to(device) for x in train_data]

            output, _ = model(img, mask)
            loss_dict = criterion(img, mask, output, gt)

            for key, coef in criterion.LAMBDA_DICT.items():
                value = coef*loss_dict[key]
                val_loss += value 

                if key in ['style']:
                    style_loss += value

            n_batch += 1
    
    return val_loss/n_batch, style_loss/n_batch
