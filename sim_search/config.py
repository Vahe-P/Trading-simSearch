# Config sources (priority order): env vars with MARKET_ prefix > .env file > defaults
# Relative paths resolve from project_root (default: cwd)
from pathlib import Path
from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ForecastConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='sim_search.env',
        env_prefix='MARKET_',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore'
    )

    # Base dir for relative paths
    project_root: Path = Field(default_factory=lambda: Path.cwd())

    # Forecasting params
    window_size: int = Field(default=-1)
    forecast_horizon: int = Field(default=20, ge=1, le=100)
    n_neighbors: int = Field(default=5, ge=1, le=50)
    norm_method: str = 'log_returns'
    distance_metric: str = 'euclidean'  # euclidean, dtw, msm, etc.
    forecast_impl: str = 'avg'  # avg or weighted-avg
    feature_col: str = 'close'
    timezone: str = 'UTC'
    resample: str = ''

    # DTW distance parameters (used when distance_metric='dtw' or 'ddtw')
    dtw_window: Optional[float] = Field(default=0.2, ge=0.0, le=1.0)  # Sakoe-Chiba band: 0-1 or None
    dtw_itakura_max_slope: Optional[float] = Field(default=None, gt=0.0)  # Itakura parallelogram: >1.0 or None

    # Visualization
    neighbor_subplots: bool = True
    plot_width: int = Field(default=1200, ge=400, le=3000)
    plot_height: int = Field(default=600, ge=300, le=2000)

    # Data
    data_path: Optional[Path] = None
    max_windows: Optional[int] = Field(default=None, ge=2)

    @field_validator('data_path', mode='before')
    @classmethod
    def resolve_path(cls, v, info):
        if v is None:
            return None
        # If data_path is relative, then resolve it relative to project_root (default to cwd)
        path = Path(v)
        if not path.is_absolute():
            project_root = info.data.get('project_root', Path.cwd())
            path = project_root / path
        return path

    @field_validator('forecast_impl')
    @classmethod
    def validate_forecast_impl(cls, v):
        if v not in ['avg', 'weighted-avg']:
            raise ValueError(f"forecast_impl must be 'avg' or 'weighted-avg'")
        return v
