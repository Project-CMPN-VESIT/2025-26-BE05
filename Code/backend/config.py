import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "model_features.pkl")
TRAINING_X_NPY = os.path.join(BASE_DIR, "train_X.npy")

PORT = int(os.environ.get("PORT", 10000))