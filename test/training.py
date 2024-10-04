import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from models import CNNGenerator, CNNDiscriminator, LSTMGenerator, LSTMDiscriminator

# Define the periodogram function
def periodogram(tt):
    periodogram = torch.mean(torch.fft.fft(tt, 200).abs(), 0)
    return torch.log(periodogram)

# GAN Class (as before)
class GAN:
    def __init__(self, generator, discriminator, train_loader, device, seq_length=20):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device
        self.train_loader = train_loader
        self.seq_length = seq_length

    def train(self, nb_epoch, lr):
        criterion = nn.BCELoss()
        mse_loss = nn.MSELoss()
        optim_d = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        optim_g = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        score = []

        time_start = time.perf_counter()
        for epoch in range(nb_epoch):
            loss_g_running, loss_d_real_running, loss_d_fake_running = 0, 0, 0
            epoch_mse_scores = []
            for _, x in enumerate(self.train_loader):
                batch = x.shape[0]
                traj = x.to(self.device)

                target_ones = torch.ones((batch, 1), device=self.device)
                target_zeros = torch.zeros((batch, 1), device=self.device)

                # Train Discriminator
                pred_real = self.discriminator(traj, batch)
                loss_real = criterion(pred_real, target_ones)

                z = torch.rand((batch, self.seq_length, 1), device=self.device)
                fake_samples = self.generator(z, batch).detach()
                pred_fake = self.discriminator(fake_samples, batch)
                loss_fake = criterion(pred_fake, target_zeros)

                loss_d = (loss_real + loss_fake) / 2
                self.discriminator.zero_grad()
                loss_d.backward()
                optim_d.step()

                loss_d_real_running += loss_real.item()
                loss_d_fake_running += loss_fake.item()

                # Train Generator
                z = torch.rand((batch, self.seq_length, 1), device=self.device)
                generated = self.generator(z, batch)
                classifications = self.discriminator(generated, batch)
                
                periodogram_loss = 10 * torch.mean((periodogram(generated) - periodogram(traj)) ** 2)
                loss_g = criterion(classifications, target_ones) + periodogram_loss
                self.generator.zero_grad()
                loss_g.backward()
                optim_g.step()

                loss_g_running += loss_g.item()
                mse_score = torch.mean((periodogram(generated) - periodogram(traj)) ** 2).item()
                epoch_mse_scores.append(mse_score)

            if epoch % 1 == 0:
                traj_sim = self.test(batch)
                traj_sim_tensor = torch.tensor(traj_sim, device=self.device, dtype=traj.dtype)
                score.append(mse_loss(traj, traj_sim_tensor).item())
                print(f'Epoch [{epoch + 1}/{nb_epoch}]: G: {loss_g_running/batch}, Dr: {loss_d_real_running/batch}, Df: {loss_d_fake_running/batch}')

        self.score = score
        self.computation_time = (time.perf_counter() - time_start)

    def test(self, nb):
        z = torch.rand((nb, self.seq_length, 1), device=self.device)
        generated = self.generator(z, nb)
        return generated.detach().cpu().numpy()

    def save(self, name):
        torch.save(self.generator.state_dict(), f'./results_test/{name}_G.pt')
        torch.save(self.discriminator.state_dict(), f'./results_test/{name}_D.pt')

# Function to initialize weights
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)



# Training different model combinations
nb_epoch = 100
def train_gan_models(train_loader, device):
    # CNN-CNN
    generator = CNNGenerator()
    discriminator = CNNDiscriminator()
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    cnn_cnn = GAN(generator, discriminator, train_loader, device)
    cnn_cnn.train(nb_epoch, lr=0.0002)
    cnn_cnn.save('cnn_cnn')

    # LSTM-LSTM
    generator = LSTMGenerator().to(device)
    discriminator = LSTMDiscriminator().to(device)

    lstm_lstm = GAN(generator, discriminator, train_loader, device)
    lstm_lstm.train(nb_epoch, lr=0.0002)
    lstm_lstm.save('lstm_lstm')

    # LSTM-CNN
    generator = LSTMGenerator().to(device)
    discriminator = CNNDiscriminator().to(device)
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    lstm_cnn = GAN(generator, discriminator, train_loader, device)
    lstm_cnn.train(nb_epoch, lr=0.0002)
    lstm_cnn.save('lstm_cnn')

    # CNN-LSTM
    generator = CNNGenerator().to(device)
    discriminator = LSTMDiscriminator().to(device)
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    cnn_lstm = GAN(generator, discriminator, train_loader, device)
    cnn_lstm.train(nb_epoch, lr=0.0002)
    cnn_lstm.save('cnn_lstm')
    score = np.array([cnn_cnn.score, cnn_lstm.score, lstm_cnn.score, lstm_lstm.score])
    computation_time = np.array([cnn_cnn.computation_time, cnn_lstm.computation_time, lstm_cnn.computation_time, lstm_lstm.computation_time])

    # Save the scores and computation times as .npy files
    with open('./results_test/score_20_steps_SS.npy', 'wb') as f:
        np.save(f, score)

    with open('./results_test/computation_time_20_steps_SS.npy', 'wb') as f:
        np.save(f, computation_time)

    print("Scores and computation times saved!")