from functools import partial
import torch

from src.preprocessors.float_quantization import FloatQuantization
from src.preprocessors.apply_to_all_tables import ApplyToAllTables
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
from src.nn.node_embedder import GATConfig, HeteroGinConfig
from src.nn.optimization import Optimization
from src.preprocessors.graph_to_one_hot import GraphToOneHot
from src.util import get_available_device, exponential_decaying_lr

SEED = 1234
GRAPH_EMBEDDING_DIM = 64
NAME = "walmart_subsampled" + f"_emb{GRAPH_EMBEDDING_DIM}"
DATA_PATH = "syntherela_benchmark/data/original/walmart_subsampled"

TRAIN_PROPORTION = 0.96
GENERATION_STEPS = 100
LOSSES_PER_EPOCH = 100
EPOCHS = 40
LR_INITIAL = 2e-3
LR_FINAL = 1e-5
PATIENCE = 10
ACTIVATION = torch.nn.SiLU()
OPTIMIZER_CLASS = torch.optim.RAdam
LR_SCHEDULER = torch.optim.lr_scheduler.LambdaLR
LR_SCHEDULER_PARAMETERS = {"lr_lambda": partial(exponential_decaying_lr, start_lr=LR_INITIAL, final_lr=LR_FINAL, n_epochs=EPOCHS)}

CATEGORY_CARDINALITY_THRESHOLD = 10
SIGMA_MIN = 1e-3

OBJECTIVE = "x1_prediction"

hidden_channels_dict = {'features': (300, 300, 300),
                        'stores':   (20, 20),
                        'depts':    (800, 800, 800, )}

dataset = MultiTableDatasetConfig(
    path=DATA_PATH,
    train_proportion=TRAIN_PROPORTION,
    split_seed=None,
    category_cardinality_threshold=CATEGORY_CARDINALITY_THRESHOLD
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
    hidden_channels=hidden_channels_dict,
    activation_layer=ACTIVATION,
    layer_norm=True,
    time_embedding=classic_embedding,
)

if True:
    node_embedder_config = GATConfig(
        hidden_channels=100,
        embedding_dim=GRAPH_EMBEDDING_DIM,
        add_node_degrees=True,
    )
else:
    node_embedder_config = HeteroGinConfig(
        hidden_channels=100,
        num_layers=2,
        linear_embedding_size=20,
        embedding_dim=GRAPH_EMBEDDING_DIM,
        add_node_degrees=True,
    )

model = GraphFlowMatching(
    graph_model=ResampleConnectedComponents(keep_original_graph=False),
    graph_denoising_net_config=GraphDenoisingNetConfig(
        node_embedder_config=node_embedder_config,
        node_conditional_denoiser_config=mlp,
    ),
    graph_preprocessor=[
        MultiTableMissingValuesHandler(),
        Simplifier(),
        ApplyToAllTables(FloatQuantization()),
        GraphToOneHot(digits_encoding_for_int=False, int_encoding='flattened_one_hot'),
        #GraphFeaturesNormalizer(),
        GraphQuantilesNormalizer(),
    ],
    optimization=opt,
    losses_per_epoch=LOSSES_PER_EPOCH,
    graph_flow=GraphFlow(sigma_min=SIGMA_MIN, objective=OBJECTIVE),
)

generation_options = dict(n_samples=1, steps=GENERATION_STEPS, x0_seed=None)

CONFIG = ExperimentConfig(
    name=NAME,
    dataset=dataset,
    model=model,
    generation_options=generation_options,
    seed=SEED,
    device=get_available_device(verbose=True),
    store=True,
)
