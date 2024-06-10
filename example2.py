import torch
import torch.optim as optim
import os
import imageio
import pickle
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
from loss_modules.VGGPerceptualLoss import VGGPerceptualLoss

from mdfloss import MDFLoss

# Set parameters
cuda_available = True
epochs = 600
application = 'JPEG'
image_path = './misc/mp_scene_0000_002.png'
code_file_path = './misc/code.pkl'
pred_file_path = './misc/pred_imgs.pkl'

if application =='SISR':
    path_disc = "./weights/Ds_SISR.pth"
elif application == 'Denoising':
    path_disc = "./weights/Ds_Denoising.pth"
elif application == 'JPEG':
    path_disc = "./weights/Ds_JPEG.pth"


def load_tensor_from_pickle(pickle_file_path):
    # Load the PyTorch tensor from the pickle file
    with open(pickle_file_path, 'rb') as file:
        loaded_tensor = pickle.load(file)

        #loaded_tensor = torch.tensor(loaded_tensor)
        #t = torch.load(pickle_file_path)['param']['code_']
        #print(t.keys())
        #t = t[..., ::-1, :]
        #print(t.shape)
        #print(t)
        loaded_tensor = torch.tensor(loaded_tensor)

    return loaded_tensor

#%% Read reference images
#imgr = imageio.imread(image_path)
loaded_tensor = load_tensor_from_pickle(code_file_path)[:4, ...]
loaded_tensor = loaded_tensor.reshape(4, 3, 128, 128)
imgd = loaded_tensor

# Create a noisy image 
#imgd = torch.rand((48, 3, 128, 128))
loaded_tensor = load_tensor_from_pickle(pred_file_path)[:4, ...]
loaded_tensor = loaded_tensor.reshape(4, 3, 128, 128)
imgr = loaded_tensor

wpr = (imgr == 1.0).all(dim=1).sum(dim=(1, 2)) / (128.0 * 128.0)
wpr = nn.Softmax(dim=0)(wpr)

# Save the original state
imgdo = imgd.detach().clone()

if cuda_available:
    imgr = imgr.cuda()
    imgd = imgd.cuda()

# Convert images to variables to support gradients
imgrb = Variable(imgr, requires_grad = False)
imgdb = Variable(imgd, requires_grad = True)

optimizer = optim.Adam([imgdb], lr=0.1)

# Initialise the loss
criterion = MDFLoss(path_disc, cuda_available=cuda_available)
#criterion = nn.MSELoss()
#criterion = VGGPerceptualLoss().cuda()

# Iterate over the epochs optimizing for the noisy image
patience = 15  # Number of epochs to wait for improvement
best_loss = float('inf')
epochs_without_improvement = 0

for ii in range(epochs):
    optimizer.zero_grad()
    loss = criterion(imgrb, imgdb, weights=wpr)
    print(f"Epoch: {ii}, Loss: {loss.item()}")

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    # Check for improvement
    if loss.item() < best_loss:
        best_loss = loss.item()
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
       print("Early stopping triggered")
       break

# Convert images to numpy
#imgrnp = imgr.cpu().squeeze(0).permute(1, 2, 0).clamp(0, 1).data.numpy()
#imgdnp = imgd.cpu().squeeze(0).permute(1, 2, 0).clamp(0, 1).data.numpy()
#imgdonp = imgdo.cpu().squeeze(0).permute(1, 2, 0).clamp(0, 1).data.numpy()


# List of images and their titles
#images = [imgdonp, imgdnp, imgrnp]
#titles = ['Noisy image', 'Recovered image', 'Reference image']

with open(os.path.join('output.pkl'), 'wb') as file:
    pickle.dump(imgdb, file)

# Iterate over the images and titles, and save each image with its title
#for i, (image, title) in enumerate(zip(images, titles)):
#    fig, ax = plt.subplots()
#    ax.imshow(image)
#    ax.set_title(title, fontsize=48)
#    ax.axis('off')  # Hide the axes
#    plt.savefig(f'image_{i}.png', bbox_inches='tight')  # Save the figure
#    plt.close(fig)  # Close the figure to free memory
