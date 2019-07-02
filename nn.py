import torch.nn as nn

class NN(nn.Module):
    def __init__(self, stackSize = 4):
        """ Neural Network Constructor 

            Parameters
            -------
            stackSize : int
                the count of frames to run the network on.
        
        """
        
        super(NN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(stackSize, 8, kernel_size=4, stride=2), nn.ReLU(), # input shape (4, 96, 96), output: (8, 47, 47)
            nn.Conv2d(8, 16, kernel_size=3, stride=2), nn.ReLU(), # input: (8, 47, 47), output: (16, 23, 23)
            nn.Conv2d(16, 32, kernel_size=3, stride=2), nn.ReLU(), # input: (16, 23, 23), output: (32, 11, 11)
            nn.Conv2d(32, 64, kernel_size=3, stride=2), nn.ReLU(), # input: (32, 11, 11), output: (64, 5, 5)
            nn.Conv2d(64, 128, kernel_size=3, stride=1), nn.ReLU(), # input: (64, 5, 5), output: (128, 3, 3)
            nn.Conv2d(128, 256, kernel_size=3, stride=1), nn.ReLU(), # input: (128, 3, 3), output: (256, 1, 1)
        )
        
        self.alpha = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
        self.beta = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
        self.fc = nn.Sequential(nn.Linear(256, 100), nn.ReLU())
        self.v = nn.Sequential(nn.Linear(256, 100), nn.ReLU(), nn.Linear(100, 1))
        self.apply(self.initWeights)

    @staticmethod
    def initWeights(m):
        """ Neural Network Weights Initialiser

            Parameters
            -------
            m : NN
                Reference to the NN module itself 
        """

        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        """ Feed forward function
        
            Parameters
            -------
            x
                The network's input

            Returns
            -------
            alpha : int
                The network's alpha value
            beta : int
                The network's beta value
            v
                The predicted value
        """

        x = self.cnn(x).view(-1, 256)
        v = self.v(x)
        x = self.fc(x)
        alpha = self.alpha(x) + 1
        beta = self.beta(x) + 1

        return (alpha, beta), v
