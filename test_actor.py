import pytest
import numpy as np
from actor import Actor

badActor = Actor(None, "./path/to/nonexistent-model.pkl")
goodActor = Actor("./models/modelForTestsOnlyDontTouchOrTrain", "./models/CarRacing-v0-rep16.pkl")

def test_saveModel():
    with pytest.raises(AssertionError):
        badActor.saveModel()
    
    goodActor.saveModel()
    assert goodActor.loadModel() == None

def test_loadModel():
    with pytest.raises(AssertionError):
        badActor.loadModel()
        
    goodActor.loadModel()
    assert goodActor.loadModel() == None
    