import torch
import os.path as osp
import pandas as pd

from torch_geometric.data import Dataset, Data

class CS6208_dataset(Dataset): 
    
    def __init__(self, root, dataframe, DLIB, data_partition):

        self.df = dataframe
        self.nodes = self.df['nodes'] #node features 
        self.edge_index = self.df['edge_index'] #edge connections
        self.edge_weight = self.df['edge_weight'] #edge magnitudes
        self.y = self.df['y'] #edge magnitudes
        self.DLIB = DLIB
        self.data_partition = data_partition
        super(CS6208_dataset, self).__init__(root, dataframe)

    @property
    def processed_file_names(self):
        return [self.DLIB + '_' + self.data_partition + '_' + f'data_{i}.pt' for i in list(self.df.index)]


    def process(self):
        for idx in self.df.index:
            
            x = torch.tensor(self.nodes[idx]).unsqueeze(-1)
            edge_index = torch.tensor(self.edge_index[idx], dtype=torch.long).T #edge connections
            edge_weight = torch.tensor(self.edge_weight[idx]).unsqueeze(-1) #edge magnitudes
            label = torch.tensor(self.y[idx]) #labels
            
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=label)
            torch.save(data, osp.join(self.processed_dir, self.DLIB + '_' + self.data_partition + '_' + f'data_{idx}.pt'))

    def len(self):
        return len(self.processed_file_names)

    def __getitem__(self, idx):
        data = torch.load(osp.join(self.processed_dir, self.DLIB + '_' + self.data_partition + '_' + f'data_{idx}.pt'))
        return data