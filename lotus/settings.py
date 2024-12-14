import lotus.models


class Settings:
    # Models
    lm: lotus.models.LM | None = None
    rm: lotus.models.RM | None = None
    helper_lm: lotus.models.LM | None = None
    reranker: lotus.models.Reranker | None = None

    # Cache settings
    enable_cache: bool = False

    # Filter cascade settings
    cascade_IS_weight: float = 0.5
    cascade_num_calibration_quantiles: int = 50

    # Join cascade settings
    min_join_cascade_size: int = 100
    cascade_IS_max_sample_range: int = 250
    cascade_IS_random_seed: int | None = None

    def configure(self, **kwargs):
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise ValueError(f"Invalid setting: {key}")
            setattr(self, key, value)


settings = Settings()
