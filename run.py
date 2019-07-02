"""
    This documents uses Google Style docstrings.
    The Flags name has been manually added for better clarity with launch flags.


    Example
    -------
    For training one can run the following commands.

        $ python run.py --train --output-model=./mymodel
        $ python run.py --train --output-model=./mymodel --input-model=./models/mypreviousmodel.pkl

    For predicting one can run the following command

        $ python run.py --input-model=./models/mymodel.pkl



    Parameters
    -------
    --action-repeat : int, 16
        Allowed repeated actions in N frames

    --frames-per-state : int
        Number of frames per state

    --log-timing : int
        The interval to run logs on when --train is enabled

    --output-model : string
        The path to save the current training data to.

        Note
        -------
            - Requires the --train flag
            - The path WITHOUT the file extension
    
    --input-model : string
        The path to load a pretrained model from.

        Note
        -------
            - The path WITH the file extension
    
    --nn-seed : int
        An integer to seed pytorch with.

    --level-seed : int
        An integer to seed the level with.

    --discount : float
        The discount factor for the neural network

    --runs : int
        The amount of training runs

        Note
        -------
            - Only works when --train flag is provided

    --train : boolean
        Enable training if this flag is provided

        Note
        -------
            - Requires the --output-model flag   
"""

from track import Track
from helpers import str2bool
from actor import Actor
import numpy as np
from nn import NN
import argparse

import torch.nn as nn
import torch

ENVIRONMENT = 'CarRacing-v0'
" The OpenAI environment to generate "

flags = argparse.ArgumentParser(description='All launch flags for making Car Racing possible')
flags.add_argument('--input-model', type=str, default=None, help='A path to a pretrained model')

# Training flags
flags.add_argument('--action-repeat', type=int, default=16, help='Allowed repeated actions in N frames')
flags.add_argument('--frames-per-state', type=int, default=4, help='Number of images per state')
flags.add_argument('--log-timing', type=int, default=10, help='Training log timing')
flags.add_argument('--output-model', type=str, default=f"./{ENVIRONMENT}", help='A path to save a model to')
flags.add_argument('--nn-seed', type=int, default=0, help='Seed for level generation')
flags.add_argument('--level-seed', type=int, default=0, help='Seed for NN seeding')
flags.add_argument('--discount', type=float, default=0.99, help='Discount factor')
flags.add_argument('--runs', type=int, default=2500, help='The amount of training runs')
flags.add_argument(
    '--train',
    type=str2bool,
    nargs='?',
    const=True,
    default=False,
    help='Enable training mode'
)
args = flags.parse_args()

# ARGUMENTS Below are for generating documentation
# PDOC has an issue with flags being overwritten by this file.
    # class Flags():
    #     train = False
    #     runs = 2500
    #     level_seed = 1
    #     nn_seed = 0
    #     action_repeat = 16
    #     frames_per_state = 4
    #     log_timing = 10
    #     output_model = f"./{ENVIRONMENT}"
    #     input_model = None
    #     discount = 0.99
    # args = Flags()

gpuDetected = torch.cuda.is_available()
hardwareDevice = torch.device("cuda" if gpuDetected else "cpu")
" Offload computation partially to gpu if Nvidia CUDA GPU's are available "

""" Seeding for the neural network

    Note:
        Keep the torch manual seed above the if statement, manual seeding without GPU is always required for CPU computations.
"""
torch.manual_seed(args.nn_seed)
if gpuDetected:
    torch.cuda.manual_seed(args.nn_seed)

if __name__ == "__main__":
    " Create an actor with the passed arguments and hardware device " 
    actor = Actor(
        outputModel = args.output_model,
        inputModel = args.input_model,
        discount = args.discount,
        trainingMode = args.train,
        framesPerState = args.frames_per_state,
        hwDevice = hardwareDevice
    )
    actor.loadModel()
    
    track = Track(levelSeed = args.level_seed, env=ENVIRONMENT)
    " Generate a randomized track " 

    if args.train:
        trainingRecords = []
        runningScore = 0

    state = track.reset()
    runs = args.runs if args.train else 10
    for run in range(runs):
        " for every lap or run (when training) run it. "
        score = 0
        state = track.reset()

        maxTrackScore = 1000
        for _track in range(maxTrackScore):
            " For every track out of the maximum score "
            if args.train:
                action, coefficient = actor.chooseActionTrain(state)
            else:
                action = actor.chooseAction(state)

            newState, reward, done, die = track.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))

            track.render()

            if args.train and actor.storeInBuffer((state, action, coefficient, reward, newState)):
                print('updating model file')
                actor.update()

            score += reward
            state = newState

            # completed the track or died.
            if done or die:
                break
        

        if args.train:
            runningScore = runningScore * 0.99 + score * 0.01

            if run % args.log_timing == 0:
                print(f'Average score over all runs: {runningScore:.2f}')
                print(f'Scored {score:.2f}/{maxTrackScore} for lap {run}')
                actor.saveModel()
        else:
            print(f'Scored {score:.2f}/{maxTrackScore} for lap {run}')
        
