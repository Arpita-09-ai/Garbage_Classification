
# Path to the original folder with all images
ORIGINAL_DATASET_PATH = r"C:\Users\91799\Downloads\Project\TrashType_Image_Dataset"

# Path to the folder where the split train/val/test data will be saved
SPLIT_DATASET_PATH = r"C:\Users\91799\Downloads\Project\dataset"

# Ratios for splitting the data
SPLIT_RATIOS = (0.7, 0.2, 0.1)  # 70% train, 20% validation, 10% test

IMG_SIZE = (124, 124)
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-4 # 0.0001
NUM_CLASSES = 6 # Number of trash categories 

SAVED_MODELS_PATH = "saved_models"
MODEL_NAME = "garbage_classifier.keras"