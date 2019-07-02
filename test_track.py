import pytest
import numpy as np
from track import Track

track = Track()

def test_rgbAsGrayScale():
    arr = [10,20,30,40,50,60,70,128,256]
    mockFrame = np.array(arr).reshape(3,3)
    "Mockframe is an array where the value of the number represents the color 0 being black, 256 being white"
    np.testing.assert_allclose(
        Track.rgbAsGrayscale(mockFrame, False),
        np.dot(mockFrame[..., :], [0.299, 0.587, 0.114])
    )
    np.testing.assert_allclose(
        Track.rgbAsGrayscale(mockFrame),
        (np.dot(mockFrame[..., :], [0.299, 0.587, 0.114]) / 128. - 1.)
    )