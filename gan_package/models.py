import torch
import torch.nn as nn

def weights_init(m):
    """
    Initialize model weights.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class CNNGenerator(nn.Module):
    def __init__(self, input_channels=256, output_channels=1):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.ConvTranspose1d(input_channels, 128, kernel_size=25, stride=2, padding=0, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(True),

            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(True),

            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(True),

            nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv1d(8, output_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.cnn(z)
        return out

class CNNDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv1d(8, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(16, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(128, 1, kernel_size=25, stride=2, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.cnn(x)
        return out.squeeze(1)
        
# LSTM Generator
class LSTMGenerator(nn.Module):
    def __init__(self, input_channels=256, output_channels=1, hidden_size=16):
        super(LSTMGenerator, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_channels, hidden_size, num_layers=1, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(hidden_size, output_channels),
            nn.Tanh()
        )

    def forward(self, x):
        batch_size = x.size(0)
        seq_length = x.size(1)
        device = x.device

        h_0 = torch.zeros(1, batch_size, self.hidden_size).to(device)
        c_0 = torch.zeros(1, batch_size, self.hidden_size).to(device)

        out, _ = self.lstm(x, (h_0, c_0))
        out = self.linear(out)
        return out  # Shape: (batch_size, seq_length, output_channels)

# LSTM Discriminator
class LSTMDiscriminator(nn.Module):
    def __init__(self, input_size=1, hidden_size=16):
        super(LSTMDiscriminator, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)
        device = x.device

        h_0 = torch.zeros(1, batch_size, self.hidden_size).to(device)
        c_0 = torch.zeros(1, batch_size, self.hidden_size).to(device)

        out, _ = self.lstm(x, (h_0, c_0))
        out = out[:, -1, :]  # Use output from the last time step
        out = self.linear(out)
        return out  # Shape: (batch_size, 1)

# Model factory functions
def get_generator(model_name, **kwargs):
    generators = {
        'CNNGenerator': CNNGenerator,
        'CNNGenerator': CNNGenerator,  
        'LSTMGenerator': LSTMGenerator,
        # Add other generator models here
    }
    if model_name in generators:
        return generators[model_name](**kwargs)
    else:
        raise ValueError(f"Generator model '{model_name}' not recognized.")

def get_discriminator(model_name, **kwargs):
    discriminators = {
        'CNNDiscriminator': CNNDiscriminator,
        'CNNDiscriminator': CNNDiscriminator,  
        'LSTMDiscriminator': LSTMDiscriminator,
        # Add other discriminator models here
    }
    if model_name in discriminators:
        return discriminators[model_name](**kwargs)
    else:
        raise ValueError(f"Discriminator model '{model_name}' not recognized.")


