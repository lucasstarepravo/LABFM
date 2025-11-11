from gnn.MessageGNN import MessagePassingGNN
from gnn.AttenGNN import AMessagePassingGNN
import logging
import pickle as pk
from collections import OrderedDict
import os
import torch
from torch.optim import Adam
import math
from torch_geometric.nn.aggr import SumAggregation

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_gnn(model_path, model_id, model_class='gnn'):

    path = os.path.join(model_path, f'attrs{model_id}.pth')
    attrs = torch.load(path,
                       map_location='cpu',
                       weights_only=False)

    if model_class.lower() not in ['gnn', 'a_gnn']:
        raise ValueError("model_class must be 'gnn', or 'a_gnn' ")

    layers = attrs['layers']
    embedding_size = attrs['embedding_size']

    if model_class.lower() == 'gnn':
        model_instance = MessagePassingGNN(embedding_size=embedding_size,
                                           layers=layers)
    else:
        model_instance = AMessagePassingGNN(embedding_size=embedding_size,
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
    # Convert list of tuples to a PyTorch tensor
    return monomial_exponent # torch.tensor(monomial_exponent, dtype=torch.long, device=device)


def calc_moments_torch(inputs, outputs, batch, approximation_order=2):
    mon_power = monomial_power(approximation_order)
    monomial = []

    for power_x, power_y in mon_power:
        inv_factorial = 1.0 / (math.factorial(power_x) * math.factorial(power_y))
        monomial_term = inv_factorial * (inputs[:, 0] ** power_x * inputs[:, 1] ** power_y)

        monomial.append(monomial_term)

    mon = torch.stack(monomial)  # ensure shape (P, B)
    batch = batch.to(torch.long)
    outs = outputs.squeeze(1)  # (B,)

    weighted = mon * outs.unsqueeze(0)  # (P, B)

    sum_aggr = SumAggregation()
    mm = []

    for i in range(mon.shape[0]):
        mm.append(sum_aggr(x=weighted[i, :], index=batch, dim=0))

    moments = torch.stack(mm)

    return moments