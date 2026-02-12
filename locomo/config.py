import pathlib
import os
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional

load_dotenv()

class Config:
    def __init__(self):
        self.root_dir = pathlib.Path(__file__).parent.parent
        self.data_path = self.root_dir / "data/locomo10.json"
        self.outs_dir = self.root_dir / "outputs"
        self.outs_dir.mkdir(parents=True, exist_ok=True)

        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_api_base = os.getenv("OPENAI_API_BASE")
        
        self.model_name: Optional[str] = None
        self.batch_size = 1
        self.max_context = 32768
        self.overwrite = False
        self.category = None
        self.out_file_path: Optional[Path] = None
        self.max_samples: Optional[int] = None
        self.no_context = False
        self.max_questions: Optional[int] = None
        self.seed: int = 42

        self.init_validate()

    def init_validate(self):
        if not self.data_path.exists():
            raise ValueError("data_path does not exist")

        if not self.openai_api_key:
            raise ValueError("openai_api_key is not set")
        if not self.openai_api_base:
            raise ValueError("openai_api_base is not set")
        
    def run_validate(self):
        if not self.model_name:
            raise ValueError("model_name is not set")
        else:
            self.out_file_path = self.outs_dir / f"{self.model_name}_qa.json"
        
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_context <= 0:
            raise ValueError("max_context must be positive")
        
config = Config()