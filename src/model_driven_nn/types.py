"""Shared array and parameter types used by the neural-network framework."""

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int64]
type BoolArray = NDArray[np.bool_]
type ParameterPair = tuple[FloatArray, FloatArray]
