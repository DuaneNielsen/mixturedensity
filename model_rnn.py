import torch
import torch.nn as nn
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

    """Computes MDN parameters a mix of gassians at each timestep
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

    def sample(self, pi, mu, sigma):
        prob_pi = torch.exp(pi)
        mn = torch.distributions.multinomial.Multinomial(1, probs=prob_pi)
        mask = mn.sample().byte()
        output_shape = mu.shape[0:-1]
        mu = mu.masked_select(mask).reshape(output_shape)
        sigma = sigma.masked_select(mask).reshape(output_shape)
        mixture = torch.normal(mu, sigma)
        return mixture

    @staticmethod
    def plus_a_little_bit(value):
        return 1e-5 - (value * 1e-5)

    """Computes the log probability of the datapoint being
    drawn from all the gaussians parametized by the network.
    Gaussians are weighted according to the pi parameter
    y - the target output 
    pi - log probability over distributions in mixture given x
    mu - vector of means of distributions
    sigma - vector of standard deviation of distribution
    """
    def loss_fn(self, y, pi, mu, sigma):
        y = y.unsqueeze(2)
        mixture = torch.distributions.normal.Normal(mu, sigma)
        log_prob = mixture.log_prob(y)
        weighted_logprob = log_prob + pi
        log_sum = torch.logsumexp(weighted_logprob, dim=3)
        log_sum = torch.logsumexp(log_sum, dim=2)
        return torch.mean(-log_sum)


if __name__ == '__main__':

    n_samples = 500
    epsilon = torch.randn(n_samples)
    x_data = torch.linspace(0, 50, n_samples)
    y_data = 7 * np.sin(0.75 * x_data) + 0.5 * x_data + epsilon

    y_data, x_data = x_data.view(-1, 1), y_data.view(-1, 1)

    plt.ion()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    line1 = ax.scatter(y_data.numpy(), x_data.numpy())
    line2 = None
    fig.canvas.draw()

    model = MDNRNN(1, 256, 2, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    x_data = x_data.unsqueeze(0).to(device)
    y_data = y_data.unsqueeze(0).to(device)

    for epoch in range(10000):
        pi, mu, sigma, _ = model(x_data)
        loss = model.loss_fn(y_data, pi, mu, sigma)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print('loss: ' + str(loss.item()))
            pi, mu, sigma, _ = model(x_data)

            y_pred = model.sample(pi, mu, sigma)

            x_plot = x_data.data.squeeze().cpu().numpy()
            y_plot = y_pred.data.squeeze().cpu().numpy()

            if line2 is None:
                line2 = ax.scatter(y_plot, x_plot)
            else:
                line2.set_offsets(np.c_[y_plot, x_plot])
            fig.canvas.draw_idle()
            plt.pause(0.1)


    plt.waitforbuttonpress()