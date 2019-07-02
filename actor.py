import numpy as np
from nn import NN

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Beta
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import os

class Actor():
    """ Create an agent and set it up for Proximal Policy Optimization """
    maxGradientNormilization = 0.5
    clipParameter = 0.1
    bufferSize = 2500
    batchSize = 128
    epoch = 10


    def __init__(self, outputModel, inputModel, discount, framesPerState = 8, trainingMode = False, hwDevice = 'CPU'): 
        """ The actor constructor
        

            Parameters
            -------
            outputModel : string
                the path to where the output model should be saved, excluding the file extension.
            inputModel : string
                the path, including file extension to the input model.
            discount : float
                The discount factor.
            framesPerState : int
                Number of frames per state.
            trainingMode: : boolean
                Whether this Actor is used for training or predicting.
            hwDevice : string, 
                CPU or CUDA (whether to offload to GPU or use the CPU).
        """
        
        self.hardwareDevice = hwDevice
        self.trainingMode = trainingMode
        self.inputPath = inputModel
        self.outputPath = outputModel
        self.discount = discount
        self.transition = np.dtype([
            ('s', np.float64, (framesPerState, 96, 96)),
            ('matrix_a', np.float64, (3,)),
            ('coefficient', np.float64),
            ('slice_to_concat', np.float64),
            ('index_exp ', np.float64, (framesPerState, 96, 96))
        ])

        self.nn = NN().double().to(self.hardwareDevice)
        self.buffer = np.empty(self.bufferSize, dtype=self.transition)
        self.optimizer = optim.Adam(self.nn.parameters(), lr=1e-3)
        self.trainingStep = 0
        self.counter = 0

    def saveModel(self):
        """ Save a model to a pytorch PKL file
        
            Raises
            -------
            AssertionError
                Raised if the output path hasn't been provided

            Notes
            -------
            `self.outputPath` has to be provided WITHOUT a file extension.
        """
        assert self.outputPath != None, "You haven't given an output path!"

        path = f"{self.outputPath}"
        while (True):
            if (not os.path.exists(f'{path}.pkl')):
                filename = f'{self.outputPath}.pkl'
                torch.save(self.nn.state_dict(), filename)
                break
            else:
               path = f'{path}-new'
    
    def loadModel(self):
        """ Load a model from a pytorch PKL file
        
            Raises
            -------
            AssertionError
                Raised if the given model path doesn't exist in the filesystem

            Notes:
            -------
            `self.inputPath` is a path to a model file INCLUDING it's file extension (usually `.pkl`)
        """
        if not self.inputPath:
            print('No input model argument was given, starting point is now set to untrained.')
            return

        assert os.path.exists(self.inputPath), "The given model path doesn't exist!"
        self.nn.load_state_dict(torch.load(self.inputPath))

    def chooseAction(self, state):
        """
            Choose an action to perform on the track

            Parameters
            -------
            state:
                The current state of the car.

            Returns
            -------
            action : np.ndarray
                An action for the network to run on the track

            Notes
            -------
                This function is only called when the --train flag is NOT provided.
        """
        state = torch.from_numpy(state).double().to(self.hardwareDevice).unsqueeze(0)
        with torch.no_grad():
            alpha, beta = self.nn(state)[0]

        action = alpha / (alpha + beta)
        return action.squeeze().cpu().numpy()

    def chooseActionTrain(self, state):
        """ Choose an action during training mode
        
            Parameters
            -------
            state:
                The current state of the car.

            Returns
            -------
            action : np.ndarray
                The actions to run on the track
            coefficient : float
                The logarithmic probability for an action

            Notes
            -------
                This function is only called when the --train flag IS provided.
        """
        state = torch.from_numpy(state).double().to(self.hardwareDevice).unsqueeze(0)
        with torch.no_grad():
            alpha, beta = self.nn(state)[0]
        dist = Beta(alpha, beta)
        action = dist.sample()
        coefficient = dist.log_prob(action).sum(dim=1)

        action = action.squeeze().cpu().numpy()
        coefficient = coefficient.item()

        return action, coefficient

    def storeInBuffer(self, transition):
        """ Store a transition in a buffer

            Parameters
            -------
            transition : dtype=self.transition 
                A transition element which is saved to the internal memory buffer

            Returns
            -------
            Boolean
                A boolean representing whether the buffer was SUCCESFULLY stored and didn't overflow.
        """
        self.buffer[self.counter] = transition
        self.counter += 1
        
        if not self.bufferSize == self.counter:
            return False
        
        self.counter = 0
        return True

    def update(self):
        """ Run an update on the network """
        self.trainingStep += 1

        sliceToConcat = torch.tensor(self.buffer['slice_to_concat'], dtype=torch.double).to(self.hardwareDevice).view(-1, 1)
        matrixA = torch.tensor(self.buffer['matrix_a'], dtype=torch.double).to(self.hardwareDevice)
        indexExp = torch.tensor(self.buffer['index_exp'], dtype=torch.double).to(self.hardwareDevice)
        s = torch.tensor(self.buffer['s'], dtype=torch.double).to(self.hardwareDevice)

        old_coefficient = torch.tensor(self.buffer['coefficient'], dtype=torch.double).to(self.hardwareDevice).view(-1, 1)

        with torch.no_grad():
            target = sliceToConcat + self.discount * self.nn(indexExp )[1]
            advantage = target - self.nn(s)[1]

        for _ in range(self.epoch):
            for index in BatchSampler(SubsetRandomSampler(range(self.bufferSize)), self.batchSize, False):
                alpha, beta = self.nn(s[index])[0]
                distance = Beta(alpha, beta)
                coefficient = distance.log_prob(matrixA[index]).sum(dim=1, keepdim=True)
                relativeAdvantage = torch.exp(coefficient - old_coefficient[index])

                s1 = relativeAdvantage * advantage[index]
                s2 = torch.clamp(ratio, 1.0 - self.clipParameter, 1.0 + self.clipParameter) * advantage[index]
                
                # Loss on an action
                aLoss = -torch.min(s1, s2).mean()
                
                # Loss on the value
                vLoss = F.smooth_l1_loss(self.nn(s[index])[1], target[index])
                
                # Total loss calculation
                loss = aLoss + (vLoss * 2.0)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

