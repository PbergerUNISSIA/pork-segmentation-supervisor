from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    APP_NAME: str = "Pork Segmentation Supervisor"
    APP_VERSION: str = "1.0.0-MVP"
    ENVIRONMENT: str = "development"

    DATABASE_URL: str = "sqlite:///./data/supervisor.db"

    MODEL_PATH: Path = Path("./data/models/unet_int8.tflite")
    MODEL_VERSION: str = "v1.0"
    MODEL_INPUT_SIZE: int = 512
    MODEL_CONFIDENCE_THRESHOLD: float = 0.75

    RETRAINING_THRESHOLD: int = 50
    UNCERTAINTY_METHOD: str = "entropy"
    HIGH_UNCERTAINTY_THRESHOLD: float = 0.3

    UPLOAD_DIR: Path = Path("./data/uploads")
    ANNOTATIONS_DIR: Path = Path("./data/annotations")
    MODELS_DIR: Path = Path("./data/models")

    LOG_LEVEL: str = "DEBUG"
    LOG_FILE: Path = Path("./logs/supervisor.log")
    LOG_ROTATION: str = "10 MB"
    LOG_RETENTION: str = "30 days"

    STREAMLIT_SERVER_PORT: int = 8501
    STREAMLIT_SERVER_ADDRESS: str = "localhost"
    STREAMLIT_THEME_PRIMARY_COLOR: str = "#FF4B4B"
    STREAMLIT_THEME_BACKGROUND_COLOR: str = "#FFFFFF"

    MAX_UPLOAD_SIZE_MB: int = 10
    SUPPORTED_FORMATS: str = "jpg,jpeg,png,bmp"
    IMAGE_PREPROCESSING_RESIZE: bool = True

    DICE_THRESHOLD: float = 0.85
    IOU_THRESHOLD: float = 0.80
    TRACK_METRICS: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    def get_supported_formats_list(self) -> list[str]:
        return self.SUPPORTED_FORMATS.split(",")

    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    def get_model_input_shape(self) -> tuple[int, int, int]:
        return (self.MODEL_INPUT_SIZE, self.MODEL_INPUT_SIZE, 3)

settings = Settings()
