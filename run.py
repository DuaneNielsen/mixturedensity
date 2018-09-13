import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from model import MDNPerceptron

n_samples = 1000

epsilon = torch.randn(n_samples)
x_data = torch.linspace(-10, 10, n_samples)
y_data = 7*np.sin(0.75*x_data) + 0.5*x_data + epsilon

y_data, x_data = x_data.view(-1, 1), y_data.view(-1, 1)

model = MDNPerceptron(1, 10, 5)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
x_data = x_data.to(device)
y_data = y_data.to(device)

for epoch in range(10000):
    optimizer.zero_grad()
    pi, mu, sigma = model(x_data)
    loss = model.loss_fn(y_data, pi, mu, sigma)
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
        print('Loss: ' + str(loss.item()))

pi, mu, sigma = model(x_data)
y_predicted = model.sample(pi, mu, sigma)


plt.figure(figsize=(8, 8))
plt.scatter(x_data, y_data, alpha=0.4)
plt.scatter(x_data, y_predicted.data, alpha=0.8)
plt.show()
