# this is the config file for the entire backend

import os

# Define root directory for files
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define data directory
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "application_code/data")

# Define configuration paths for yolo model
YOLO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "toolbox/pytorch_objectdetecttrack")