"""Shared array and parameter types used by the neural-network framework."""

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int64]
type BoolArray = NDArray[np.bool_]
type ParameterPair = tuple[FloatArray, FloatArray]
type YamlMapping = Mapping[str, object]
type Config = Mapping[str, object]
