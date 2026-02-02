from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Generic, Optional, Protocol, TypeVar

import pandas as pd


InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class FitPredictModel(Protocol[InputT, OutputT]):
    def fit(self, x: InputT, y: Optional[pd.Series] = None, **kwargs: Any) -> "FitPredictModel[InputT, OutputT]": ...
    def predict(self, x: InputT, **kwargs: Any) -> OutputT: ...


@dataclass
class ModelResult:
    model: str
    outputs: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {"model": self.model, "outputs": self.outputs}


class BaseModel(Generic[InputT, OutputT]):
    name: str = "base"

    def __init__(self) -> None:
        self.is_fitted = False

    def fit(self, x: InputT, y: Optional[pd.Series] = None, **kwargs: Any) -> "BaseModel[InputT, OutputT]":
        raise NotImplementedError

    def predict(self, x: InputT, **kwargs: Any) -> OutputT:
        raise NotImplementedError

    def fit_predict(self, x: InputT, y: Optional[pd.Series] = None, **kwargs: Any) -> OutputT:
        self.fit(x, y=y, **kwargs)
        return self.predict(x, **kwargs)

    def require_fitted(self) -> None:
        if not getattr(self, "is_fitted", False):
            raise RuntimeError(f"{self.__class__.__name__} is not fitted")
