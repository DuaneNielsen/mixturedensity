import torch
import matplotlib.pyplot as plt
import numpy as np
from model import MDNPerceptron
import torch.nn as nn

n_samples = 1000

epsilon = torch.randn(n_samples)
x_data = torch.linspace(-10, 10, n_samples)
y_data = 7*np.sin(0.75*x_data) + 0.5*x_data + epsilon

y_data, x_data = x_data.view(-1, 1), y_data.view(-1, 1)

plt.ion()
#fig = plt.figure(figsize=(4, 10))
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(8, 4))
line1 = ax1.scatter(y_data.numpy(), x_data.numpy(), alpha=0.5, s=0.8)
line1 = ax2.scatter(y_data.numpy(), x_data.numpy(), alpha=0.5, s=0.8)
line2 = None
line3 = None
fig.canvas.draw()
fig.savefig('images/initial.png', bbox_inches='tight')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x_data = x_data.to(device)
y_data = y_data.to(device)

hidden_size = 20
model = nn.Sequential(nn.Linear(1, hidden_size),
                      nn.ReLU(),
                      nn.Linear(hidden_size, hidden_size),
                      nn.ReLU(),
                      nn.Linear(hidden_size, hidden_size),
                      nn.ReLU(),
                      nn.Linear(hidden_size, 1),
                      ).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
mseloss = nn.MSELoss()

for epoch in range(130):
    optimizer.zero_grad()
    y = model(x_data)
    loss = mseloss(x_data, y)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print('Loss: ' + str(loss.item()))
        y_pred = model(x_data)

        x_plot = x_data.data.squeeze().cpu().numpy()
        y_plot = y_pred.data.squeeze().cpu().numpy()

        if line2 is None:
            line2 = ax1.scatter(y_plot, x_plot, alpha=0.6, s=0.9)
        else:
            line2.set_offsets(np.c_[y_plot, x_plot])
        fig.canvas.draw_idle()
        fig.savefig('images/linear%04d.png' % (epoch,), bbox_inches='tight')
        plt.pause(0.1)


model = MDNPerceptron(1, 10, 5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


for epoch in range(10000):
    optimizer.zero_grad()
    pi, mu, sigma = model(x_data)
    loss = model.loss_fn(y_data, pi, mu, sigma)
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
        print('Loss: ' + str(loss.item()))

        pi, mu, sigma = model(x_data)
        y_pred = model.sample(pi, mu, sigma)

        x_plot = x_data.data.squeeze().cpu().numpy()
        y_plot = y_pred.data.squeeze().cpu().numpy()

        if line3 is None:
            line3 = ax2.scatter(y_plot, x_plot, alpha=0.6, s=0.9)
        else:
            line3.set_offsets(np.c_[y_plot, x_plot])
        fig.canvas.draw_idle()
        fig.savefig('images/mdn%04d' % (epoch, ) + '.png', bbox_inches='tight')
        plt.pause(0.1)