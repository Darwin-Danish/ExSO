import exso_sdk.model as mdl
from exso_sdk.config import REQUIRED_COLUMNS

features = REQUIRED_COLUMNS
model = mdl.load_model(input_dim=len(features))
print("Model loaded successfully!")
