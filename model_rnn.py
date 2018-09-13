import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class MDNRNN(nn.Module):
    def __init__(self, z_size, hidden_size, num_layers, n_gaussians):
        nn.Module.__init__(self)
        self.z_size = z_size
        self.n_gaussians = n_gaussians

        self.lstm = nn.LSTM(z_size, hidden_size, num_layers, batch_first=True)

        self.pi = nn.Linear(hidden_size, z_size * n_gaussians)
        self.lsfm = nn.LogSoftmax(dim=3)
        self.mu = nn.Linear(hidden_size, z_size * n_gaussians)
        self.sigma = nn.Linear(hidden_size, z_size * n_gaussians)


    """Computes MDN parameters for each timestep
    z - (batch size, episode length, latent size)
    pi, mu, sigma - (batch size, episode length, n_gaussians)
    """
    def forward(self, z):
        episode_length = z.size(1)
        output, (hn, cn) = self.lstm(z)

        pi = self.pi(output)
        mu = self.mu(output)
        sigma = torch.exp(self.sigma(output))

        pi = pi.view(-1, episode_length, self.z_size, self.n_gaussians)
        mu = mu.view(-1, episode_length, self.z_size, self.n_gaussians)
        sigma = sigma.view(-1, episode_length, self.z_size, self.n_gaussians)

        pi = self.lsfm(pi)

        return pi, mu, sigma, (hn, cn)


    """Computes the log probability of the datapoint being
    drawn from all the gaussians parametized by the network.
    Gaussians are weighted according to the pi parameter 
    """
    def loss_fn(self, y, pi, mu, sigma):
        y = y.unsqueeze(2)
        mixture = torch.distributions.normal.Normal(mu, sigma)
        log_prob = mixture.log_prob(y)
        weighted_logprob = log_prob + pi
        weighted_prob = torch.exp(weighted_logprob)
        weighted_prob = weighted_prob + 1e-20 #todo check a good value for epsilon
        sum = torch.sum(weighted_prob, dim=[2,3]) # sum of the episode
        log_prob_loss = -torch.log(sum)
        mean = torch.mean(log_prob_loss) #mean of the batch
        return mean


if __name__ == '__main__':

    n_samples = 500
    epsilon = torch.randn(n_samples)
    x_data = torch.linspace(0, 50, n_samples)
    y_data = 7 * np.sin(0.75 * x_data) + 0.5 * x_data + epsilon

    y_data, x_data = x_data.view(-1, 1), y_data.view(-1, 1)

    plt.figure(figsize=(8, 8))
    plt.scatter(y_data, x_data, alpha=0.4)
    #plt.scatter(x_data, y_predicted.data, alpha=0.8)
    #plt.show()

    model = MDNRNN(1, 1, 1, 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    x_data = x_data.unsqueeze(0).to(device)
    y_data = y_data.unsqueeze(0).to(device)

    for epoch in range(100):
        pi, mu, sigma, finalstate = model(x_data)
        loss = model.loss_fn(y_data, pi, mu, sigma)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print('loss: ' + str(loss.item()))

    pi, mu, sigma, finalstate = model(x_data)