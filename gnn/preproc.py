from gnn.MessageGNN import MessagePassingGNN
from gnn.AttenGNN import AMessagePassingGNN
from gnn.SNA_GNN import csf_gnn
from gnn.Node_GNN import NodeMessagePassingGNN
from gnn.stencil_model import StencilGNN
from gnn.m63 import SmallGNN
import logging
import pickle as pk
from collections import OrderedDict
import os
import torch
from torch.optim import Adam
import math
from torch_geometric.nn.aggr import SumAggregation
from gnn.m65 import m65
from gnn.m67 import m67

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_gnn(path, model_class='gnn'):

    attrs = torch.load(path,
                       map_location='cpu',
                       weights_only=False)

    layers = attrs['layers']
    embedding_size = attrs['embedding_size']

    if model_class.lower() == 'gnn':
        model_instance = MessagePassingGNN(embedding_size=embedding_size,
                                           layers=layers)
    elif model_class.lower() == 'a_gnn':
        model_instance = AMessagePassingGNN(embedding_size=embedding_size,
                                           layers=layers)
    elif model_class.lower() == 'n_gnn':
        model_instance = NodeMessagePassingGNN(embedding_size=embedding_size,
                                           layers=layers)
    elif model_class.lower() == 'stencil':
        model_instance = StencilGNN(embedding_size=embedding_size,
                                    layers=layers)
    elif model_class.lower() == 'small':
        model_instance = SmallGNN(embedding_size=embedding_size,
                                    layers=layers)
    elif model_class.lower() == 'm65':
        model_instance = m65(embedding_size=embedding_size,
                                    layers=layers)
    elif model_class.lower() == 'm67':
        model_instance = m67(embedding_size=embedding_size,
                             layers=layers)
    else:
        model_instance = csf_gnn(embedding_size=embedding_size,
                                            layers=layers)

    weight_dict = OrderedDict()
    weight_dict.update(
        (k[len("module."):], v) if k.startswith("module.") else (k, v) for k, v in attrs['weights'].items())
    model_instance.load_state_dict(weight_dict)

    optimizer = Adam(model_instance.parameters())
    optimizer.load_state_dict(attrs['optimizer'])

    logger.info(f"Model loaded from {path}")

    return model_instance, optimizer

# Used in GNN to compute moments of predicted errors
def monomial_power(polynomial):
    monomial_exponent = []
    for total_polynomial in range(1, polynomial + 1):
        for i in range(total_polynomial + 1):
            monomial_exponent.append((total_polynomial - i, i))
    return monomial_exponent


def calc_moments_torch(inputs, outputs, batch, approximation_order=2):
    mon_power = monomial_power(approximation_order)
    monomial = []

    for power_x, power_y in mon_power:
        inv_factorial = 1.0 / (math.factorial(power_x) * math.factorial(power_y))
        monomial_term = inv_factorial * (inputs[:, 0] ** power_x * inputs[:, 1] ** power_y)

        monomial.append(monomial_term)

    mon = torch.stack(monomial)
    batch = batch.to(torch.long)
    outs = outputs.squeeze(1)
    weighted = mon * outs.unsqueeze(0)

    sum_aggr = SumAggregation()
    mm = []

    for i in range(mon.shape[0]):
        mm.append(sum_aggr(x=weighted[i, :], index=batch, dim=0))

    moments = torch.stack(mm)

    return moments