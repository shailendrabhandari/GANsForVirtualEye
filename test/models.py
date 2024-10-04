import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sequence_length = 20
# CNN Generator
class CNNGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(    
            nn.ConvTranspose1d(20, 10, kernel_size=5, stride=2, padding=0, bias=False),
            nn.BatchNorm1d(10),
            nn.ReLU(True),

            nn.ConvTranspose1d(10, 5, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(5),
            nn.ReLU(True),

            nn.ConvTranspose1d(5, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )
        
    def forward(self, z, batch_size):
        out = self.cnn(z)
        return out

# CNN Discriminator
class CNNDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 5, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(5),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(5, 10, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(10),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(10, 20, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(20),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(20, 1, kernel_size=5, stride=2, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, batch_size):
        out = self.cnn(x)
        return out.squeeze(1)

# LSTM Generator
class LSTMGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 16, 1, batch_first=True, bias=True)
        self.linear = nn.Sequential(
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh()
        )

    def forward(self, x, batch_size):
        h_0 = torch.autograd.Variable(torch.randn(1, batch_size, 16)).to(x.device)  # Hidden state
        c_0 = torch.autograd.Variable(torch.randn(1, batch_size, 16)).to(x.device)  # Cell state

        out, h_n = self.lstm(x, (h_0, c_0))  # LSTM forward pass
        out = self.linear(out)
        out = out.view(batch_size, 1, x.shape[1])  # Reshape to (batch_size, 1, sequence_length)
        return torch.cumsum(out, 1)  # Cumulative sum along the sequence

class LSTMDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(1, 16, 1, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh()
        )

    def forward(self, x, batch_size):
        # Dynamically calculate the sequence length based on the input tensor's size
        total_elements = x.numel() 
        sequence_length = total_elements // batch_size  
        
        # Reshape input based on the actual sequence length
        out = x.view(batch_size, sequence_length, 1)

        # LSTM forward pass
        h_0 = torch.autograd.Variable(torch.randn(1, batch_size, 16)).to(x.device)
        c_0 = torch.autograd.Variable(torch.randn(1, batch_size, 16)).to(x.device)
        out, (h_n, c_n) = self.lstm(out, (h_0, c_0))

        # Apply linear layers
        out = self.linear(out)

        # Apply sigmoid activation to classify
        out = 1 / (1 + torch.exp(-torch.mean(out, 1)))

        return out


