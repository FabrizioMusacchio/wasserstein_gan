# %% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
from torch import nn
import ot
import os
if not os.path.exists('GAN_demo_images'):
    os.makedirs('GAN_demo_images')
# %% GENERATE DATA
torch.manual_seed(1)
sigma = 0.1
n_dims = 2
n_features = 2

def get_data(n_samples):
    # Generates a 2D dataset of samples forming a circle with noise.
    
    # for reproducibility:
    torch.manual_seed(1)
    
    c = torch.rand(size=(n_samples, 1))
    angle = c * 2 * np.pi
    x = torch.cat((torch.cos(angle), torch.sin(angle)), 1)
    x += torch.randn(n_samples, 2) * sigma
    data_sample_name = 'circle'
    return x, data_sample_name

def get_data(n_samples):
    # Generates a 2D dataset of samples forming a cross with noise.
    
    # for reproducibility:
    torch.manual_seed(1)
    
    # set the thickness of the cross:
    thickness = 0.2

    # half samples from vertical line, half from horizontal:
    x_vert = torch.randn(n_samples // 2, 2) * sigma
    x_vert[:, 0] *= thickness  # For vertical line, x-coordinate is always within the range [-thickness/2, thickness/2]

    x_horiz = torch.randn(n_samples // 2, 2) * sigma
    x_horiz[:, 1] *= thickness  # For horizontal line, y-coordinate is always within the range [-thickness/2, thickness/2]

    x = torch.cat((x_vert, x_horiz), 0)
    data_sample_name = 'cross'
    return x, data_sample_name

def get_data(n_samples):
    # Generates a 2D dataset of samples forming a sine wave with noise.
    
    # for reproducibility:
    torch.manual_seed(1)
    
    x = torch.linspace(-np.pi, np.pi, n_samples).view(-1, 1)
    y = torch.sin(x) + sigma * torch.randn(n_samples, 1)
    data = torch.cat((x, y), 1)
    data_sample_name = 'sine'
    return data, data_sample_name


# plot the distributions
plt.figure(figsize=(5, 5))
x, data_sample_name = get_data(500)
plt.figure(1)
plt.scatter(x[:, 0], x[:, 1], label='Data samples from $\mu_d$', alpha=0.5)
plt.title('Data distribution')
plt.legend()
plt.tight_layout()
plt.savefig(f'GAN_demo_images/{data_sample_name}_data_distribution.png', dpi=200)
plt.show()
# %% GENERATOR MODEL
# define the MLP model
class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(n_features, 200)
        self.fc2 = nn.Linear(200, 500)
        self.fc3 = nn.Linear(500, n_dims)
        self.relu = torch.nn.ReLU()  # instead of Heaviside step fn

    def forward(self, x):
        output = self.fc1(x)
        output = self.relu(output)  # instead of Heaviside step fn
        output = self.fc2(output)
        output = self.relu(output)
        output = self.fc3(output)
        return output
# %% TRAINING THE MODEL
G = Generator()
optimizer = torch.optim.RMSprop(G.parameters(), lr=0.00019, eps=1e-5)

# number of iteration and size of the batches:
n_iter = 200  # set to 200 for doc build but 1000 is better ;)
size_batch = 500

# generate static samples to see their trajectory along training:
n_visu = 100
xnvisu = torch.randn(n_visu, n_features)
xvisu = torch.zeros(n_iter, n_visu, n_dims)

ab = torch.ones(size_batch) / size_batch
losses = []


for i in range(n_iter):
    # generate noise samples:
    xn = torch.randn(size_batch, n_features)

    # generate data samples:
    xd,_ = get_data(size_batch)

    # generate sample along iterations:
    xvisu[i, :, :] = G(xnvisu).detach()

    # generate samples and compte distance matrix:
    xg = G(xn)
    M = ot.dist(xg, xd)

    loss = ot.emd2(ab, ab, M)
    losses.append(float(loss.detach()))

    if i % 10 == 0:
        print("Iter: {:3d}, loss={}".format(i, losses[-1]))

    loss.backward()
    optimizer.step()

    del M

# plot the loss (Wasserstein distance) along iterations:
plt.figure(figsize=(6, 4))
plt.semilogy(losses)
plt.grid()
plt.title('Wasserstein distance')
plt.xlabel("Iterations")
plt.tight_layout()
plt.savefig(f'GAN_demo_images/{data_sample_name}_losses.png', dpi=200)
plt.show()
# %% PLOT TRAJECTORIES OF GENERATED SAMPLES ALONG ITERATIONS
plt.figure(3, (10, 10))
ivisu = [0, 10, 25, 50, 75, 125, 15, 175, 199]

for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.scatter(xd[:, 0], xd[:, 1], label='Data samples from $\mu_d$', alpha=0.1)
    plt.scatter(xvisu[ivisu[i], :, 0], xvisu[ivisu[i], :, 1], label='Data samples from $G\#\mu_n$', alpha=0.5)
    plt.xticks(())
    plt.yticks(())
    plt.title('Iter. {}'.format(ivisu[i]))
    if i == 0:
        plt.legend()
    # get the current axes' limits
    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()
plt.tight_layout()
plt.savefig(f'GAN_demo_images/{data_sample_name}_trajectories.png', dpi=200)

xy_lim_max = np.max(np.abs(np.concatenate((xlim, ylim))))
# %% ANIMATE TRAJECTORIES OF GENERATED SAMPLES ALONG ITERATION

plt.figure(4, (6, 6))

def _update_plot(i):
    plt.clf()
    plt.scatter(xd[:, 0], xd[:, 1], label='Data samples from $\mu_d$', alpha=0.1)
    plt.scatter(xvisu[i, :, 0], xvisu[i, :, 1], label='Data samples from $G\#\mu_n$', alpha=0.5)
    plt.xticks(())
    plt.yticks(())
    plt.xlim([-xy_lim_max, xy_lim_max])
    plt.ylim([-xy_lim_max, xy_lim_max])
    plt.title('Iter. {}'.format(i))
    plt.legend(loc='upper left')
    plt.tight_layout()
    return 1

i = 0
plt.scatter(xd[:, 0], xd[:, 1], label='Data samples from $\mu_d$', alpha=0.1)
plt.scatter(xvisu[i, :, 0], xvisu[i, :, 1], label='Data samples from $G\#\mu_n$', alpha=0.5)
plt.xticks(())
plt.yticks(())
plt.xlim([-xy_lim_max, xy_lim_max])
plt.ylim([-xy_lim_max, xy_lim_max])
plt.legend(loc='upper right')
plt.title('Iter. {}'.format(ivisu[i]))

ani = animation.FuncAnimation(plt.gcf(), _update_plot, n_iter, interval=100, repeat_delay=2000)
#save animation as gif:
ani.save(f'GAN_demo_images/{data_sample_name}_animation.gif', writer='imagemagick', fps=60)
# %% GENERATE AND VISUALIZE DATA
size_batch = 500
xd, _ = get_data(size_batch)
xn = torch.randn(size_batch, 2)
x = G(xn).detach().numpy()

plt.figure(5, figsize=(5, 5))
plt.scatter(xd[:, 0], xd[:, 1], label='Data samples from $\mu_d$', alpha=0.5)
plt.scatter(x[:, 0], x[:, 1], label='Data samples from $G\#\mu_n$', alpha=0.5)
plt.title('Sources and Target distributions')
plt.xlim([-xy_lim_max, xy_lim_max])
plt.ylim([-xy_lim_max, xy_lim_max])
plt.legend()
plt.tight_layout()
plt.savefig(f'GAN_demo_images/{data_sample_name}_sources_and_target.png', dpi=200)
plt.show()