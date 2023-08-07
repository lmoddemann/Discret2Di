import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import _pickle as pkl


def standardize_data(data, name_pkl):
    # Standardize the data by subtracting the mean and scaling to achieve unit variance.
    scaler = StandardScaler()
    scaler.fit(data)
    data = pd.DataFrame(scaler.transform(data), columns=data.columns, index=data.index)
    # save scaler
    path = Path("scaler")
    with open(Path(path, name_pkl), 'wb') as f:
        pkl.dump(scaler, f)
    return data

def load_scaler(scaler_name):
    # Loading the saved scaler from the location "scaler" using the identifier "scaler_name"
    path = Path("scaler")
    file = Path(path, scaler_name)
    if not file.exists():
        raise ValueError("No scaler found at path:", file)
    with open(file, 'rb') as f:
        scaler = pkl.load(f)
    return scaler

