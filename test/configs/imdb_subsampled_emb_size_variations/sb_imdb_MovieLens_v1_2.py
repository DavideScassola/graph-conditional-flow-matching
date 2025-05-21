from functools import partial
import torch

from src.preprocessors.graph_quantile_normalization import GraphQuantilesNormalizer
from src.preprocessors.simplifier import Simplifier
from src.preprocessors.multi_table_missing_values_handler import MultiTableMissingValuesHandler
from src.experiment_config import ExperimentConfig
from src.graph_data import MultiTableDatasetConfig
from src.models.diffusion.diffusion_process import GraphFlow
from src.models.diffusion.graph_flow_matching import GraphFlowMatching
from src.models.graph.resample_connected_components import ResampleConnectedComponents
from src.nn.graph_denoising_net import GraphDenoisingNetConfig
from src.nn.time_embedding import classic_embedding
from src.nn.node_conditional_denoiser import NodeConditionalMLP_DenoiserConfig
from src.nn.node_embedder import HeteroGinConfig
from src.nn.optimization import Optimization
from src.preprocessors.graph_to_one_hot import GraphToOneHot
from src.util import exponential_decaying_lr, get_available_device

SEED = 1234
GRAPH_EMBEDDING_DIM = 2
NAME = f"imdb_MovieLens_v1_emb{GRAPH_EMBEDDING_DIM}"
DATA_PATH = "syntherela_benchmark/data/original/imdb_MovieLens_v1"

TRAIN_PROPORTION = 0.9
GENERATION_STEPS = 50
LOSSES_PER_EPOCH = 200
EPOCHS = 20
LR_INITIAL = 1e-2
LR_FINAL = 1e-5
PATIENCE = 5
ACTIVATION = torch.nn.SiLU()
OPTIMIZER_CLASS = torch.optim.RAdam
LR_SCHEDULER = torch.optim.lr_scheduler.LambdaLR
LR_SCHEDULER_PARAMETERS = {"lr_lambda": partial(exponential_decaying_lr, start_lr=LR_INITIAL, final_lr=LR_FINAL, n_epochs=EPOCHS)}

NN_SCALE = 200

SIGMA_MIN = 1e-3

OBJECTIVE = "x1_prediction"

dataset = MultiTableDatasetConfig(
    path=DATA_PATH,
    train_proportion=TRAIN_PROPORTION,
    split_seed=None
)


opt = Optimization(
    epochs=EPOCHS,
    batch_size=1,
    optimizer_class=OPTIMIZER_CLASS,
    optimizer_hyperparameters={"lr": LR_INITIAL},
    lr_scheduler_class=LR_SCHEDULER,
    lr_scheduler_params=LR_SCHEDULER_PARAMETERS,
    patience=PATIENCE,
)

mlp = NodeConditionalMLP_DenoiserConfig(
    hidden_channels=(NN_SCALE, NN_SCALE, ),
    activation_layer=ACTIVATION,
    layer_norm=True,
    time_embedding=classic_embedding,
)

node_embedder_config = HeteroGinConfig(
    hidden_channels=100,
    num_layers=3,
    linear_embedding_size=20,
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