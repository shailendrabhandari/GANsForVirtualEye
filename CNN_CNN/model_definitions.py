import torch
from torch import nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Initialize weights as per DCGAN specifications
def weights_init(m):
    """
    This function initializes the model weights randomly from a 
    Normal distribution. Follows the DCGAN paper specification.
    Source: https://arxiv.org/pdf/1511.06434.pdf
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Define periodogram function
def periodogram(tt):
    return torch.log(torch.mean(torch.fft.fft(tt, 200).abs(), 0))

# CNN Generator 1
class CNNGenerator1(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.ConvTranspose1d(input_channels, 128, kernel_size=25, stride=2, padding=0, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.ConvTranspose1d(128, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.ConvTranspose1d(32, 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(8),
            nn.ReLU(True),
            nn.ConvTranspose1d(8, output_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.cnn(z)

# CNN Discriminator 1
class CNNDiscriminator1(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(8, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(32, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(128, 1, kernel_size=25, stride=2, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.cnn(x).squeeze(1)

# CNN Generator 2
class CNNGenerator2(nn.Module):
    def __init__(self, input_channels, output_channels):
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
        return self.cnn(z)

# CNN Discriminator 2
class CNNDiscriminator2(nn.Module):
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
        return self.cnn(x).squeeze(1)
    
# Define the LSTM Generator
class LSTMGenerator(nn.Module):
    def __init__(self, input_channels=256, output_channels=200):
        super(LSTMGenerator, self).__init__()
        self.lstm = nn.LSTM(input_channels, 16, 1, batch_first=True, bias=True)
        self.linear = nn.Sequential(
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, output_channels),
            nn.Tanh()
        )

    def forward(self, x):
        batch_size = x.size(0)
        h_0 = torch.randn(1, batch_size, 16).to(device)  
        c_0 = torch.randn(1, batch_size, 16).to(device)  

        out, _ = self.lstm(x, (h_0, c_0))  
        out = self.linear(out)  
        return torch.cumsum(out, dim=1)

# Define the LSTM Discriminator
class LSTMDiscriminator(nn.Module):
    def __init__(self, input_size=200):
        super(LSTMDiscriminator, self).__init__()
        self.lstm = nn.LSTM(input_size, 16, 1, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)
        h_0 = torch.randn(1, batch_size, 16).to(device)  
        c_0 = torch.randn(1, batch_size, 16).to(device)  

        out, _ = self.lstm(x, (h_0, c_0))  
        out = self.linear(out)  
        return torch.mean(out, dim=1)  
