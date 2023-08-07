import numpy as np
from sklearn.model_selection import train_test_split
from utils import standardize_data
import pandas as pd 
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl


class Dataset(Dataset):
    def __init__(self,
                 dataframe: pd.DataFrame
                 ):
        self.df = dataframe
        self.length = self.df.shape[0]

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        return self.df.iloc[index, :].values.astype(np.float32)

class DataModule(pl.LightningDataModule):
    # Decide on the different datasets by entering either 'SmA_normal', 
    # 'SmA_anomalyID2', BeRfiPl_ds1n', BeRfiPl_ds1c', 'SWaT_norm', 'SWaT_anom, 'Tank_normal' or 'Tank_anomaly'
    def __init__(self,
                 hparam_batch,
                 dataset_name,
                 num_workers: int=20, 
                 ):
        super().__init__()
        self.batch_size = hparam_batch
        self.num_workers = num_workers
        self.dataset_name = dataset_name

        if self.dataset_name == 'SmA_normal':
            df = pd.read_csv('preprocessed_data/SmA/id1_norm.csv').reset_index(drop=True).drop(columns=['CuStepNo ValueY'])

        elif self.dataset_name == 'SmA_anomalyID2':
            df = pd.read_csv('preprocessed_data/SmA/id2_anomaly.csv').reset_index(drop=True).drop(columns=['CuStepNo ValueY'])

        elif self.dataset_name == 'BeRfiPl_ds1n':
            df = pd.read_csv('preprocessed_data/BeRfiPl/ds1n.csv', index_col=0).reset_index(drop=True)

        elif self.dataset_name == 'BeRfiPl_ds1c':
            df = pd.read_csv('preprocessed_data/BeRfiPl/ds1c.csv', index_col=0).reset_index(drop=True)

        elif self.dataset_name == 'SWaT_norm':
            df = pd.read_csv('preprocessed_data/SWaT/swat_norm_p1.csv').reset_index(drop=True)
        
        elif self.dataset_name == 'SWaT_anom':
            df = pd.read_csv('preprocessed_data/SWaT/swat_anom_p1.csv').reset_index(drop=True)

        elif self.dataset_name == 'Tank_normal':
            df = pd.read_csv('preprocessed_data/tank_simulation/norm.csv').reset_index(drop=True)

        elif self.dataset_name == 'Tank_q1fault':
            df = pd.read_csv('preprocessed_data/tank_simulation/q1_faulty.csv').reset_index(drop=True)

        elif self.dataset_name == 'Tank_v12fault':
            df = pd.read_csv('preprocessed_data/tank_simulation/v12_faulty.csv').reset_index(drop=True)

        elif self.dataset_name == 'Tank_v23fault':
            df = pd.read_csv('preprocessed_data/tank_simulation/v23_faulty.csv').reset_index(drop=True)

        elif self.dataset_name == 'Tank_v3fault':
            df = pd.read_csv('preprocessed_data/tank_simulation/v3_faulty.csv').reset_index(drop=True)

        df_train, df_val = train_test_split(df, test_size=0.2)
        df_train_sc = pd.DataFrame(standardize_data(df_train, f'scaler_{self.dataset_name}_train.pkl'), columns=df_train.columns)
        df_val_sc = pd.DataFrame(standardize_data(df_val, 'scaler_siemens_val.pkl'), columns=df_val.columns)
        self.ds_train = Dataset(dataframe = df_train_sc)
        self.ds_val = Dataset(dataframe = df_val_sc)

    def train_dataloader(self):
        return DataLoader(self.ds_train, 
                            batch_size=self.batch_size, 
                            shuffle=False, 
                            num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.ds_val,
                            batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=self.num_workers)
    


if __name__ == '__main__':
    train = DataModule()
    train.setup()
