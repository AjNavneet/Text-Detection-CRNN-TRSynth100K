import torch.nn as nn

class ConvRNN(nn.Module):
    def __init__(self, n_classes):
        """
        Initialize the Convolutional Recurrent Neural Network (ConvRNN) model.

        :param n_classes: Number of unique characters in the dataset
        """
        super(ConvRNN, self).__init()

        # Activation function
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        # Convolutional layers
        self.conv_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn_1 = nn.BatchNorm2d(64)
        self.pool_1 = nn.MaxPool2d((2, 2))

        self.conv_2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn_2 = nn.BatchNorm2d(128)
        self.pool_2 = nn.MaxPool2d((2, 2))

        self.conv_3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn_3 = nn.BatchNorm2d(256)

        self.conv_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn_4 = nn.BatchNorm2d(256)
        self.pool_4 = nn.MaxPool2d((2, 1))

        self.conv_5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn_5 = nn.BatchNorm2d(512)

        self.conv_6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn_6 = nn.BatchNorm2d(512)
        self.pool_6 = nn.MaxPool2d((2, 1))

        self.conv_7 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn_7 = nn.BatchNorm2d(512)

        # Linear layers
        self.linear_1 = nn.Linear(1024, 64)
        self.linear_2 = nn.Linear(64, 64)

        # Recurrent layers
        self.rnn_1 = nn.GRU(64, 128, num_layers=2, bidirectional=True, dropout=0.25, batch_first=True)
        self.rnn_2 = nn.GRU(256, 64, num_layers=2, bidirectional=True, dropout=0.25, batch_first=True)

        # Output layer
        self.dense = nn.Linear(128, n_classes + 1)

    def forward(self, images):
        x = self.conv_1(images)
        x = self.bn_1(x)
        x = self.relu(x)
        x = self.pool_1(x)

        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.relu(x)
        x = self.pool_2(x)

        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.relu(x)

        x = self.conv_4(x)
        x = self.bn_4(x)
        x = self.relu(x)
        x = self.pool_4(x)

        x = self.conv_5(x)
        x = self.bn_5(x)
        x = self.relu(x)

        x = self.conv_6(x)
        x = self.bn_6(x)
        x = self.relu(x)
        x = self.pool_6(x)

        x = self.conv_7(x)
        x = self.bn_7(x)
        x = self.relu(x)

        x = x.view(-1, 512 * 2, 40)
        x = x.permute(0, 2, 1)

        x = self.linear_1(x)
        x = self.relu(x)

        x = self.linear_2(x)
        x = self.relu(x)

        x, _ = self.rnn_1(x)
        x, _ = self.rnn_2(x)

        x = self.dense(x)
        return x
