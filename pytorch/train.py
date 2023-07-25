import os
import torch
import torchvision.datasets as datasets
from tqdm import tqdm
from torch import nn, optim
from model import VariationalAutoEncoder
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import argparse
import matplotlib.pyplot as plt

# configuration
parser = argparse.ArgumentParser(description='Variational AutoEncoder')
parser.add_argument('--weights_file', type=str, default="weights.pth", help='file to save the weights')
parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='device to use (cuda or cpu)')
parser.add_argument('--input_dim', type=int, default=784, help='input dimension')
parser.add_argument('--h_dim', type=int, default=200, help='hidden dimension')
parser.add_argument('--z_dim', type=int, default=20, help='latent dimension')
parser.add_argument('--num_epochs', type=int, default=20, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--lr_rate', type=float, default=3e-4, help='learning rate')

args = parser.parse_args()

weights_file = args.weights_file
device = torch.device(args.device)
INPUT_DIM = args.input_dim
H_DIM = args.h_dim
Z_DIM = args.z_dim
NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
LR_RATE = args.lr_rate

# Dataset Loading
dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
model = VariationalAutoEncoder(INPUT_DIM, H_DIM, Z_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR_RATE)
loss_fn = nn.BCELoss(reduction="sum")

if os.path.isfile(weights_file):
    model.load_weights(weights_file)
    print(f"Loaded weights from: {weights_file}")

loss_list = []  # create an empty list to store the loss values

for epoch in range(NUM_EPOCHS):
    loop = tqdm(enumerate(train_loader))
    
    for i, (x, _) in loop:
        x = x.to(device).view(x.shape[0], INPUT_DIM)
        x_reconstructed, mu, sigma = model(x)

        # compute loss
        reconstruction_loss = loss_fn(x_reconstructed, x)
        kl_div = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
        
        # backprop
        loss = reconstruction_loss + kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())
        
        loss_list.append(loss.item())  # append the loss value to the list

# moving average 
# window size for moving average
N = 100

# compute moving averages using list comprehension and built-in Python functions
moving_averages = [sum(loss_list[i - N + 1: i + 1]) / N for i in range(N - 1, len(loss_list))]


# plot the loss values
plt.plot(moving_averages)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.savefig('loss.png')
plt.show()

model = model.to("cpu")

def inference(digit, num_examples=1):
    images = []
    idx = 0
    for x, y in dataset:
        if y == idx:
            images.append(x)
            idx += 1
        if idx == 10:
            break

    encodings_digit = []
    for d in range(10):
        with torch.no_grad():
            mu, sigma = model.encode(images[d].view(1, INPUT_DIM))
        encodings_digit.append((mu, sigma))

    mu, sigma = encodings_digit[digit]
    for example in range(num_examples):
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        out = model.decode(z)
        out = out.view(-1, 1, 28, 28)
        save_image(out, f"images/generated_{digit}_ex{example}.png")

        incoming = images[digit]
        save_image(incoming, f"images/incoming_{digit}_ex{example}.png")

for idx in range(10):
    inference(idx, num_examples=1)

model.save_weights(weights_file)
