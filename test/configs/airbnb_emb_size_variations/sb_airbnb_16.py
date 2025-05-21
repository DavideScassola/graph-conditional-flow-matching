import torch

from src.models.graph.resample_connected_components import ResampleConnectedComponents
from src.preprocessors.graph_quantile_normalization import GraphQuantilesNormalizer
from src.preprocessors.simplifier import Simplifier
from src.preprocessors.multi_table_missing_values_handler import MultiTableMissingValuesHandler
from src.experiment_config import ExperimentConfig
from src.graph_data import MultiTableDatasetConfig
from src.models.diffusion.diffusion_process import GraphFlow
from src.models.diffusion.graph_flow_matching import GraphFlowMatching

from src.nn.graph_denoising_net import GraphDenoisingNetConfig
from src.nn.time_embedding import classic_embedding
from src.nn.node_conditional_denoiser import NodeConditionalMLP_DenoiserConfig
from src.nn.node_embedder import GATConfig
from src.nn.optimization import Optimization
from src.preprocessors.graph_to_one_hot import GraphToOneHot
from src.util import get_available_device
from torch.optim.lr_scheduler import CosineAnnealingLR

SEED = 1234
NAME = "airbnb-simplified_subsampled"
DATA_PATH = "syntherela_benchmark/data/original/airbnb-simplified_subsampled"

TRAIN_PROPORTION = 0.9
GENERATION_STEPS = 100
LOSSES_PER_EPOCH = 200
EPOCHS = 40
LR = 1e-3
PATIENCE = 5
ACTIVATION = torch.nn.SiLU()
OPTIMIZER_CLASS = torch.optim.RAdam
LR_SCHEDULER = CosineAnnealingLR
LR_SCHEDULER_PARAMETERS = {"T_max": EPOCHS}

NN_SCALE = 700
GRAPH_EMBEDDING_DIM = 16
NAME = NAME + f"_emb_{GRAPH_EMBEDDING_DIM}"
SIGMA_MIN = 0

OBJECTIVE = "x1-x0_prediction"

dataset = MultiTableDatasetConfig(
    path=DATA_PATH,
    train_proportion=TRAIN_PROPORTION,
    split_seed=None
)


opt = Optimization(
    epochs=EPOCHS,
    batch_size=1,
    optimizer_class=OPTIMIZER_CLASS,
    optimizer_hyperparameters={"lr": LR},
    lr_scheduler_class=LR_SCHEDULER,
    lr_scheduler_params=LR_SCHEDULER_PARAMETERS,
    patience=PATIENCE,
)

mlp = NodeConditionalMLP_DenoiserConfig(
    hidden_channels=(NN_SCALE, NN_SCALE, NN_SCALE, ),
    activation_layer=ACTIVATION,
    layer_norm=True,
    time_embedding=classic_embedding,
)

node_embedder_config = GATConfig(
    hidden_channels=100,
    embedding_dim=GRAPH_EMBEDDING_DIM,
    add_node_degrees=True,
)

model = GraphFlowMatching(
    graph_model=ResampleConnectedComponents(),
    graph_denoising_net_config=GraphDenoisingNetConfig(
        node_embedder_config=node_embedder_config,
        node_conditional_denoiser_config=mlp,
    ),
    graph_preprocessor=[
        MultiTableMissingValuesHandler(),
        Simplifier(),
        GraphToOneHot(digits_encoding_for_int=False, int_encoding='flattened_one_hot'),
        GraphQuantilesNormalizer(),
    ],
    optimization=opt,
    losses_per_epoch=LOSSES_PER_EPOCH,
    graph_flow=GraphFlow(sigma_min=SIGMA_MIN, objective=OBJECTIVE),
)

generation_options = dict(n_samples=1, steps=GENERATION_STEPS)

CONFIG = ExperimentConfig(
    name=NAME,
    dataset=dataset,
    model=model,
    generation_options=generation_options,
    seed=SEED,
    device=get_available_device(verbose=True),
    store=True,
)
