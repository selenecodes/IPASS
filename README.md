# IPASS
Making a car race around a track using OpenAI's gym circuit

This project uses Google Style docstrings.
The Flags name has been manually added for better clarity with launch flags.

## Example
For training one can run the following commands.
```bash
$ python run.py --train --output-model=./mymodel
$ python run.py --train --output-model=./mymodel --input-model=./models/mypreviousmodel.pkl
```
For predicting one can run the following command
```bash
$ python run.py --input-model=./models/mymodel.pkl
```

## Flags
**`--action-repeat`**: `int` = `16`
- Allowed repeated actions in N frames

**`--frames-per-state`**: `int` = `4`
- Number of frames per state

**`--log-timing`**: `int` = `16`
- The interval to run logs on when `–train` is enabled

**`--output-model`**: `string` = `f"./{ENVIRONMENT}"`
- The path to save the current training data to.

```
Note
-------
    - Requires the --train flag
    - The path WITHOUT the file extension
```

**`--input-model`**: `string`
- The path to load a pretrained model from.

```
Note
-------
    - The path WITH the file extension
```

**`--nn-seed`**: `int` = `0`
- An integer to seed pytorch with.

**`--level-seed`**: `int` = `0`
- An integer to seed the level with.

**`--discount`**: `float` = `0.99`
- The discount factor for the neural network

**`--runs`**: `float` = `0.99`
- The amount of training runs

**`--train`**: `boolean` = `False`
- Enable training mode

## Global variables
```python
ENVIRONMENT = 'CarRacing-v0'
```
```python
hardwareDevice = 'CPU'
```
