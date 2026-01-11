import dataclasses
from typing import Optional

import pandas as pd


@dataclasses.dataclass
class Window:
    data: pd.DataFrame
    train_cutoff: Optional[pd.Timestamp] = None
    distance: Optional[float] = None
