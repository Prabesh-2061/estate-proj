# fix_pickle.py
import joblib
from transformer import AreaPerBedroomTransformer

# Load the old pickle (that references __main__)
model = joblib.load("best_model.pkl")

# Save a new pickle that references transformers.AreaPerBedroomTransformer
joblib.dump(model, "best_model_s.pkl")

print("✅ Fixed model saved")
