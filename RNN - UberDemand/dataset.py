import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class PickupDS(Dataset):
    PREV_HOURS = 6
    NUM_REGIONS = 4

    def __init__(self, csv_loc="/home/hesam/NNDL/UberPickup/UberDemand.csv", train_subset=0.8):
        super().__init__()
        csv_data = pd.read_csv(csv_loc).drop(columns=['Unnamed: 0', 'borough', 'date'])
        x = csv_data.drop(columns=['pickups']).to_numpy().astype('float32')

        self.dropped_pickUPs = StandardScaler().fit(x[:int(len(x) * train_subset)]).transform(x).astype('float32')
        pickups = []
        for i in range(4):
            x = csv_data['pickups'][i::4].to_numpy().reshape(-1, 1).astype('float32')
            pickups.append(StandardScaler().fit(x[:int(len(x) * train_subset)]).transform(x).astype('float32'))
        self.pickups = pickups

    def __getitem__(self, index):
        current_data = self.dropped_pickUPs[
                       index * self.NUM_REGIONS:index * self.NUM_REGIONS + (self.PREV_HOURS + 1) * self.NUM_REGIONS]
        x = []
        for i in range(self.PREV_HOURS):
            features = np.array([self.pickups[j][index + i] for j in range(4)]).reshape(-1)

            features = np.concatenate([features, current_data[i * self.NUM_REGIONS:(i + 1) * self.NUM_REGIONS][0, :]],
                                      axis=0)

            x.append(features)
        return np.array(x), np.array([self.pickups[j][index + self.PREV_HOURS] for j in range(4)]).reshape(-1)

    def __len__(self):
        return (len(self.dropped_pickUPs) // self.NUM_REGIONS) - self.PREV_HOURS