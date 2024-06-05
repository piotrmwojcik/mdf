import torch
import torch.optim as optim
import imageio
import pickle
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable

from mdfloss import MDFLoss

# Set parameters
cuda_available = True
epochs = 50
application = 'SISR'
image_path = './misc/mp_scene_0000_002.png'
pickle_file_path = './misc/scene_0000.pkl'

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
imgr = imageio.imread(image_path)
imgr = torch.from_numpy(imageio.core.asarray(imgr/255.0))
imgr = imgr.type(dtype=torch.float64)
imgr = imgr.permute(2,0,1)[:3, :, :]
imgr = imgr.unsqueeze(0).type(torch.FloatTensor)

# Create a noisy image 
#imgd = torch.rand(imgr.size())
loaded_tensor = load_tensor_from_pickle(pickle_file_path)
loaded_tensor = loaded_tensor.reshape(6, 3, 128, 128)
#imgd = loaded_tensor[2, :, :, :].unsqueeze(0)

# Save the original state
imgdo = imgd.detach().clone()

if cuda_available:
    imgr = imgr.cuda()
    imgd = imgd.cuda()

# Convert images to variables to support gradients
imgrb = Variable( imgr, requires_grad = False)
imgdb = Variable( imgd, requires_grad = True)

optimizer = optim.Adam([imgdb], lr=0.1)

# Initialise the loss
criterion = MDFLoss(path_disc, cuda_available=cuda_available)

# Iterate over the epochs optimizing for the noisy image
for ii in range(0,epochs):
    
    optimizer.zero_grad()
    loss = criterion(imgrb,imgdb) 
    print("Epoch: ",ii," loss: ", loss.item())
    loss.backward()
    optimizer.step()
 
    

# Convert images to numpy
imgrnp = imgr.cpu().squeeze(0).permute(1, 2, 0).clamp(0, 1).data.numpy()
imgdnp = imgd.cpu().squeeze(0).permute(1, 2, 0).clamp(0, 1).data.numpy()
imgdonp = imgdo.cpu().squeeze(0).permute(1, 2, 0).clamp(0, 1).data.numpy()


# List of images and their titles
images = [imgdonp, imgdnp, imgrnp]
titles = ['Noisy image', 'Recovered image', 'Reference image']

# Iterate over the images and titles, and save each image with its title
for i, (image, title) in enumerate(zip(images, titles)):
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title(title, fontsize=48)
    ax.axis('off')  # Hide the axes
    plt.savefig(f'image_{i}.png', bbox_inches='tight')  # Save the figure
    plt.close(fig)  # Close the figure to free memory