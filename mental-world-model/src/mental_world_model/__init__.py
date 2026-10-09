"""心智世界模型 — 以心和心所为核心的定义驱动工程系统。"""

from mental_world_model.llm.env_file import load_project_dotenv

load_project_dotenv()

from mental_world_model.model import MentalWorldModel

__version__ = "0.1.0"
__all__ = ["MentalWorldModel", "__version__"]
