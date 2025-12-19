from gnn.graph_constructor import StencilGraph, DenseGraph
from torch_geometric.loader import DataLoader
from gnn.preproc import load_gnn
import torch
from gnn.preproc import calc_moments_torch
import os
from scipy.spatial import cKDTree
import shutil
from tqdm import tqdm
from functions.nodes import neighbour_nodes_kdtree
import numpy as np


def gnn_weights(coordinates, h, total_nodes, nodes_in_domain):

    # used to gather all features, later will all be passed to GNN
    features = []
    # needed to construct graph
    embedding_size = 256
    approximation_order = 4 # only used to check moments
    num_neighbours = 35
    tree = cKDTree(coordinates)
    model_x, _ = load_gnn('./gnn/trained_model', 13, model_class='n_gnn',
                          full_path='gnn/trained_model/attrs29_epoch1096.pth') # model for x derivative
    model_laplace, _ = load_gnn('./gnn/trained_model', 6, model_class='n_gnn',
                                full_path='gnn/trained_model/attrs30_epoch796.pth')  # model for laplace


    ref_node_ls = []
    neigh_coor_dict = {}
    norm_h = []
    h_dict = {}
    neigh_xy_dist = {}

    num_edges = 2 * (num_neighbours - 1)

    batch_size = min(nodes_in_domain, 2048)

    edge_index = torch.zeros((2, num_edges * batch_size), dtype=torch.long)
    batch = torch.zeros(num_neighbours * batch_size, dtype=torch.long)
    x = torch.zeros((num_neighbours * batch_size, 2), dtype=torch.float32)
    h_total = np.zeros(batch_size)

    idx_correct = torch.arange(1, num_neighbours)


    data_loader = []
    b = -1

    for ref_x, ref_y in coordinates:#tqdm(coordinates, desc="Creating GNN dataset " + str(total_nodes), ncols=100):
        if ref_x > 0.5 or ref_x < -0.5 or ref_y > 0.5 or ref_y < -0.5: continue

        b += 1
        n = b * num_neighbours

        ref_node = (ref_x, ref_y)
        ref_node_ls.append(ref_node)
        neigh_r_d, neigh_xy_d, neigh_coor_dict[ref_node] = neighbour_nodes_kdtree(coordinates,
                                                                                  ref_node,
                                                                                  3 * h,
                                                                                  tree,
                                                                                  max_neighbors=num_neighbours)

        neigh_xy_dist[ref_node] = neigh_xy_d

        # Computing the batches
        batch[b * num_neighbours : (b + 1) * num_neighbours] = b

        # Computing the edge indexes
        edge_index[0, b * num_edges : b * num_edges + num_edges // 2] = n + idx_correct
        edge_index[0, b * num_edges + num_edges // 2: b * num_edges + num_edges] = n

        edge_index[1, b * num_edges : b * num_edges + num_edges // 2] = n
        edge_index[1, b * num_edges + num_edges // 2: b * num_edges + num_edges] = n + idx_correct

        # stencil length
        h_total[b] = neigh_r_d[-1]

        # node features
        x[b * num_neighbours: (b + 1) * num_neighbours, :] = torch.tensor(neigh_xy_d / neigh_r_d[-1], dtype=torch.float32)

        #max_r = np.max(neigh_r_d)           # obtaining maximum radius for normalisation
        #norm_h.append(max_r)                # saving max radius to denormalize predictions
        #h_dict[ref_node] = max_r
        #features.append(neigh_xy_d/max_r)   # appending normalised features to list

        if b == batch_size - 1:
            data_loader.append([x, edge_index, batch, h_total])
            b = -1
            edge_index = torch.zeros((2, num_edges * batch_size), dtype=torch.long)
            batch = torch.zeros(num_neighbours * batch_size, dtype=torch.long)
            x = torch.zeros((num_neighbours * batch_size, 2), dtype=torch.float32)
            h_total = torch.zeros(batch_size, dtype=torch.float32)

    if b > -1:
        x = x[:(b + 1) * num_neighbours, :]
        edge_index = edge_index[:, :b * num_edges + num_edges]
        batch = batch[:(b + 1) * num_neighbours]
        h_total = h_total[:b + 1]
        data_loader.append([x, edge_index, batch, h_total])

    #features_np = np.array(features)        # converting to numpy array
    #h_np = np.array(norm_h)
    #features_np /= h
    #ds = StencilGraph(features=features_np, # need to input h as normalisation
    #                          embedding_size=embedding_size,
    #                          h=h_np)

    #ds = DenseGraph(features=features_np,
    #                embedding_size=embedding_size,
    #                h=h_np)

    #data_loader = DataLoader(ds,
    #                         batch_size=min(features_np.shape[0],1024),
    #                         shuffle=False,
    #                         num_workers=0,
    #                         drop_last=False)

    weights_x = []
    unscaled_w_x = []

    weights_laplace = []
    unscaled_w_laplace = []

    model_x.eval()
    model_laplace.eval()

    with torch.no_grad():
        for b in tqdm(data_loader, desc="Predicting GNN Weights for " + str(total_nodes), ncols=100):

            x          = b[0]
            edge_index = b[1]
            batch      = b[2]
            h          = b[3]

            out_x = model_x(x,
                            edge_index,
                            batch)

            out_laplace = model_laplace(x,
                                        edge_index,
                                        batch)

            #pred_m = calc_moments_torch(batch.distances,
            #                            out,
            #                            batch.batch,
            #                            approximation_order=approximation_order)

            # can change all these reshape to view
            pred_reshape_x = out_x.view((int(batch[-1]) + 1, -1))
            weights_x.extend(pred_reshape_x.cpu().numpy() / h[:, None])

            #pred_reshape_x = torch.reshape(out_x, (int(batch[-1]) + 1, -1))
            #weights_x.extend(pred_reshape_x.cpu().numpy() / h[:, None])
            #unscaled_w_x.extend(pred_reshape_x.detach().cpu().numpy())

            pred_reshape_laplace = out_laplace.view((int(batch[-1]) + 1, -1))
            weights_laplace.extend(pred_reshape_laplace.cpu().numpy() / (h[:, None] ** 2))
            #pred_reshape_laplace = torch.reshape(out_laplace, (int(batch[-1]) + 1, -1))
            #weights_laplace.extend(pred_reshape_laplace.cpu().numpy() / (h[:, None]**2))
            #unscaled_w_laplace.extend(pred_reshape_laplace.detach().cpu().numpy())


        # add check to labfm weights
        #weights_np = np.array(weights_x)
        #unscaled_w_np = np.array(unscaled_w_x)


    weights_x_dict = {}
    weights_laplace_dict = {}
    for key, x, lap in zip(ref_node_ls, weights_x, weights_laplace):
        weights_x_dict[key] = x
        weights_laplace_dict[key] = lap

    return weights_x_dict, weights_laplace_dict, neigh_coor_dict, h_dict, neigh_xy_dist

