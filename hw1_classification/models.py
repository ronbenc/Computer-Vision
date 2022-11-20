import torch
import torch.nn as nn

class SvhnCNN(nn.Module):
    """CNN for the SVHN Datset"""
    
    def __init__(self):
        """CNN Builder."""
        super(SvhnCNN, self).__init__()
        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            # What are the dims after this layer?
            # How many weights?
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv Layer block 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05), # <- Why is this here?
            # Modified Conv Layer block 3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=
            1),
            nn.ReLU(inplace=True),
            # As we go deeper - use more channels!
            )
        self.fc_layer = nn.Sequential(
        nn.Dropout(p=0.1),
        nn.Linear(8192, 512), # <- How do we know it's 8192? Why 512 later?
        nn.ReLU(inplace=True),
        nn.Linear(512, 10) # <- Why 10 here?
        )
        
    def forward(self, x):
        """Perform forward."""

        # conv layers
        x = self.conv_layer(x)

        # flatten
        x = x.view(x.size(0), -1)

        # fc layer
        x = self.fc_layer(x)
        return x


class ResidualBlock(nn.Module):
    """
    A general purpose residual block.
    """

    def __init__(self, in_channels,
        channels,
        kernel_size: int = 3,
        batchnorm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.main_path, self.shortcut_path = None, None
        self.in_channels = in_channels
        self.channels = channels
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.kernel_size = kernel_size
        self.main_path = self._make_main_path()
        self.shortcut_path = self._make_shortcut_path()

    def _make_main_path(self):
        layers = []
        in_channels_list = [self.in_channels] + self.channels[:-1]
        out_channels_list = self.channels
        
        for i, (in_channel, out_channel) in enumerate(zip(in_channels_list, out_channels_list)):
            layers.append(nn.Conv2d(in_channel, out_channel, bias=True, kernel_size=self.kernel_size, padding='same'))
            if i == len(self.channels)-1: break
            layers.append(nn.Dropout2d(self.dropout))
            if self.batchnorm: layers.append(nn.BatchNorm2d(out_channel))
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def _make_shortcut_path(self):
        if self.in_channels == self.channels[-1]:
            layer = nn.Identity()
        else:
            layer = nn.Conv2d(self.in_channels, self.channels[-1], bias=False, kernel_size=1)

        return nn.Sequential(layer)

    def forward(self, x):
        out = self.main_path(x) + self.shortcut_path(x)
        out = torch.relu(out)
        return out

    
class RonDoronNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Sequential(
            # nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            # nn.MaxPool2d(kernel_size=2, stride=2),

            ResidualBlock(3, [64, 64], 3, dropout=0.3),
            # ResidualBlock(64, [64, 64], 3, dropout=0.3),
            ResidualBlock(64, [64, 64], 3, dropout=0.3),

            nn.MaxPool2d(kernel_size=2, stride=2),


            ResidualBlock(64, [128, 128], 3, dropout=0.3),
            # ResidualBlock(128, [128, 128], 3, dropout=0.3),
            ResidualBlock(128, [128, 128], 3, dropout=0.3),

            nn.MaxPool2d(kernel_size=2, stride=2),

            ResidualBlock(128, [256, 256], 3, dropout=0.3),
            # ResidualBlock(256, [256, 256], 3, dropout=0.3),
            ResidualBlock(256, [256, 256], 3, dropout=0.3),


            ResidualBlock(256, [512, 512], 3, dropout=0.3),
            # ResidualBlock(512, [512, 512], 3, dropout=0.3),
            ResidualBlock(512, [512, 512], 3, dropout=0.3),

            nn.MaxPool2d(kernel_size=2, stride=2),
        
            ResidualBlock(512, [1024, 1024], 3, dropout=0.3),
            # ResidualBlock(1024, [1024, 1024], 3, dropout=0.3),
            ResidualBlock(1024, [1024, 1024], 3, dropout=0.3)
        )

        y = self.conv_layer(torch.zeros((3, 32, 32)))
        n_features = torch.numel(y)
        print(n_features)

        self.fc_layer = nn.Sequential(
        nn.Dropout(p=0.1),
        nn.Linear(n_features, 512), # <- How do we know it's 8192? Why 512 later?
        nn.ReLU(inplace=True),
        nn.Linear(512, 10) # <- Why 10 here?
        )


    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x
