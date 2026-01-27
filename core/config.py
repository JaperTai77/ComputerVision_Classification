from dotenv import load_dotenv
import os
import torch

load_dotenv()

class Settings:
    root_dir = os.getenv("ROOT_DIR")
    target_col_name = os.getenv("TARGET_COL_NAME")
    train_csv = os.getenv("TRAIN_CSV", "train.csv")

    resize_shape = int(os.getenv("IMAGE_SHAPE", '224'))

    saved_model_file = os.getenv("SAVED_MODEL_FILE_LOCATION", "./")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    

Config = Settings()