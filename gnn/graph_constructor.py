from torch_geometric.data import Data, Dataset, InMemoryDataset
import torch
import numpy as np
import logging
from tqdm import tqdm


class InMemoryStencilGraph(InMemoryDataset):
    def __init__(self,
                 features,
                 embedding_size,
                 root,
                 h,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):

        self.features  = np.ascontiguousarray(features).astype(np.float32, copy=False)
        self.h = np.ascontiguousarray(h).astype(np.float32, copy=False)

        self.total_datapoints = features.shape[0] # num of nodes in domain
        self.max_neighbours = self.features.shape[1] # max number of neighbours
        self.embedding_size = embedding_size
        #self.transform_my_class = ToUndirected()

        # prebuild once
        self.edges_max = torch.tensor(
            [[i, 0] for i in range(1, self.max_neighbours)],
            dtype=torch.long).T

        #self.transform = ToUndirected()
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])


    @property
    def processed_file_names(self):
        return ['data.pt']


    def process(self):
        data_list = []

        for idx in tqdm(range(self.total_datapoints), desc="Processing graphs"):

            # edge features and label
            # removing the central weight node

            #num_neigh = self.distances[d_idx, 1:, 0][torch.isfinite(self.distances[d_idx, 1:, 0])]
            # removing the distance of the central node to itself (0.0)
            edge_attr = self.features[idx, 1:, :]
            distances = torch.from_numpy(self.features[idx, ...].copy()).to(torch.float32)

            # creating edge attributes
            # (distance from neighbour points to central point, and from central point to neighbour points)
            edge_attr = torch.from_numpy(edge_attr.copy()).to(torch.float32)
            rev_edge_attr = -edge_attr
            edge_attr = torch.concat((edge_attr, rev_edge_attr))

            # slice down to actual degree
            num_neigh = edge_attr.shape[0]
            edge_index = self.edges_max[:, :num_neigh].long()
            tmp = [1,0]
            rev_edge_index = edge_index[tmp, :]
            edge_index = torch.concat((edge_index, rev_edge_index), dim=1)

            x = torch.ones((self.features.shape[1], self.embedding_size), dtype=torch.float32)
            x[0, :] = 1 / (x.shape[0] ** .5)

            w_norm = torch.tensor(self.h[idx], dtype=torch.float32)

            data = Data(x=x,
                        distances=distances,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        h=w_norm)


            data_list.append(data)


        self.save(data_list, self.processed_paths[0])
