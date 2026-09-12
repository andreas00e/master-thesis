import os 
import warnings
from omegaconf import OmegaConf

def setup_environment(): 
    try: 
        OmegaConf.register_new_resolver("eval", eval)
    except ValueError: 
        pass 

    #Suppress Open MPI warnings
    os.environ["OMPI_MCA_btl"] = "^openib" 
    # Headless Offscreen Rendering 
    os.environ["MUJOCO_GL"] = "egl" # Mujoco
    os.environ["PYOPENGL_PLATFORM"] = "egl" # OpenGL 
    # Distributed Training
    os.environ["NCCL_P2P_DISABLE"]="1"
    os.environ["NCCL_IB_DISABLE"]="1"

    warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
    warnings.filterwarnings("ignore", category=UserWarning, module="r3m")
    warnings.filterwarnings("ignore", category=UserWarning, module="lightning")
    warnings.filterwarnings("ignore", category=UserWarning, module="lightning.pytorch.utilities._pytree")
    warnings.filterwarnings("ignore", message=".*incompatible copy of pydevd already imported.*")