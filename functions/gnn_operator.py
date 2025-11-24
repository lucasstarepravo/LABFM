from gnn.graph_constructor import StencilGraph
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


def gnn_weights(coordinates, h, total_nodes):

    # used to gather all features, later will all be passed to GNN
    features = []
    # needed to construct graph
    embedding_size = 64
    approximation_order = 4 # only used to check moments
    tree = cKDTree(coordinates)
    model_x, _ = load_gnn('./gnn/trained_model', 13, model_class='a_gnn') # model for x derivative
    model_laplace, _ = load_gnn('./gnn/trained_model', 6, model_class='a_gnn')  # model for x derivative

    ref_node_ls = []
    neigh_coor_dict = {}
    norm_h = []

    for ref_x, ref_y in tqdm(coordinates, desc="Creating GNN dataset " + str(total_nodes), ncols=100):
        if ref_x > 0.5 or ref_x < -0.5 or ref_y > 0.5 or ref_y < -0.5: continue

        ref_node = (ref_x, ref_y)
        ref_node_ls.append(ref_node)
        neigh_r_d, neigh_xy_d, neigh_coor_dict[ref_node] = neighbour_nodes_kdtree(coordinates,
                                                                                  ref_node,
                                                                                  2 * h,
                                                                                  tree,
                                                                                  max_neighbors=35)

        max_r = np.max(neigh_r_d)           # obtaining maximum radius for normalisation
        norm_h.append(max_r)                # saving max radius to denormalize predictions
        features.append(neigh_xy_d/max_r)   # appending normalised features to list


    features_np = np.array(features)        # converting to numpy array
    h_np = np.array(norm_h)
    #features_np /= h
    ds = StencilGraph(features=features_np, # need to input h as normalisation
                              embedding_size=embedding_size,
                              h=h_np)

    data_loader = DataLoader(ds,
                             batch_size=features_np.shape[0],
                             shuffle=False,
                             num_workers=0,
                             drop_last=False)

    weights_x = []
    unscaled_w_x = []

    weights_laplace = []
    unscaled_w_laplace = []

    model_x.eval()
    model_laplace.eval()

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Predicting GNN Weights for " + str(total_nodes), ncols=100):
            out_x = model_x(batch.x,
                            batch.edge_index,
                            batch.edge_attr,
                            batch.batch)

            out_laplace = model_laplace(batch.x,
                                        batch.edge_index,
                                        batch.edge_attr,
                                        batch.batch)

            #pred_m = calc_moments_torch(batch.distances,
            #                            out,
            #                            batch.batch,
            #                            approximation_order=approximation_order)

            pred_reshape_x = torch.reshape(out_x, (int(max(batch.batch)) + 1, -1))
            weights_x.extend(pred_reshape_x.detach().cpu().numpy() / batch.h[:, None])
            #unscaled_w_x.extend(pred_reshape_x.detach().cpu().numpy())

            pred_reshape_laplace = torch.reshape(out_laplace, (int(max(batch.batch)) + 1, -1))
            weights_laplace.extend(pred_reshape_laplace.detach().cpu().numpy() / (batch.h[:, None]**2))
            #unscaled_w_laplace.extend(pred_reshape_laplace.detach().cpu().numpy())


        # add check to labfm weights
        #weights_np = np.array(weights_x)
        #unscaled_w_np = np.array(unscaled_w_x)


    weights_x_dict = {}
    weights_laplace_dict = {}
    for key, x, lap in zip(ref_node_ls, weights_x, weights_laplace):
        weights_x_dict[key] = x
        weights_laplace_dict[key] = lap

    return weights_x_dict, weights_laplace_dict, neigh_coor_dict

