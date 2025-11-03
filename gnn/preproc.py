from gnn.MessageGNN import MessagePassingGNN
import logging
import pickle as pk
from collections import OrderedDict
import os
import torch
from torch.optim import Adam

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_gnn(model_path, model_id, model_class='gnn'):

    path = os.path.join(model_path, f'attrs{model_id}.pth')
    attrs = torch.load(path,
                       map_location='cpu',
                       weights_only=False)

    if model_class.lower() not in ['gnn']:
        raise ValueError('model_class must be gnn')

    layers = attrs['layers']
    embedding_size = attrs['embedding_size']

    if model_class.lower() == 'gnn':
        model_instance = MessagePassingGNN(embedding_size=embedding_size,
                                           layers=layers)

    weight_dict = OrderedDict()
    weight_dict.update(
        (k[len("module."):], v) if k.startswith("module.") else (k, v) for k, v in attrs['weights'].items())
    model_instance.load_state_dict(weight_dict)

    optimizer = Adam(model_instance.parameters())
    optimizer.load_state_dict(attrs['optimizer'])

    logger.info(f"Model loaded from {path}")

    return model_instance, optimizer

