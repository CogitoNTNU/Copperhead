from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.floating[Any]] | Any
type IntArray = NDArray[np.integer[Any]] | Any
type BoolArray = NDArray[np.bool_] | Any
type ArrayLike = NDArray[Any] | Any
type YamlMapping = Mapping[str, object]
type Config = Mapping[str, object]
