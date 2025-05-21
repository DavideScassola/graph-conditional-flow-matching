import os
import random
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from src.graph_data import (DatasetConfig, MultiTableDataset,
                            MultiTableDatasetConfig, dummy_tables)
from src.models.diffusion.diffusion_process import GraphFlow
from src.models.model import Model
from src.nn.graph_denoising_net import GraphDenoisingNetConfig
from src.nn.optimization import Optimization
from src.nn.training import nn_training
from src.preprocessors.compose import Compose
from src.report import *
from src.util import get_hetero_data_device, pickle_load, pickle_store


class GraphFlowMatching(Model):
    def __init__(
        self,
        *,
        graph_model,
        graph_preprocessor,
        graph_denoising_net_config: GraphDenoisingNetConfig,
        optimization: Optimization,
        graph_flow: GraphFlow,
        use_t_linspace_for_training: bool = True,
        losses_per_epoch: int = 100,
        subgraph_generator: Callable | None = None,
        conditioning_tables: list[str] = [],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.graph_denoising_net_config = graph_denoising_net_config
        self.conditioning_tables = conditioning_tables
        graph_denoising_net_config.conditioning_tables = conditioning_tables
        self.optimization = optimization
        self.graph_preprocessor = (
            graph_preprocessor
            if not isinstance(graph_preprocessor, list)
            else Compose(graph_preprocessor)
        )
        self.graph_model = graph_model
        self.losses_per_epoch = losses_per_epoch
        self.graph_flow = graph_flow
        self.use_t_linspace_for_training = use_t_linspace_for_training
        self.subgraph_generator = subgraph_generator

    def get_preprocessed_data(
        self,
        *,
        dataset: MultiTableDatasetConfig,
        train: bool = True,
        fit: bool | None = None,
        device: str | torch.device,
    ) -> HeteroData:

        if fit is None:
            fit = train

        mtd = dataset.get()

        if False:  # TODO: for debugging preprocessors
            check = self.graph_preprocessor.reverse_transform(
                self.graph_preprocessor.fit_transform(mtd)
            )

        if fit:
            return self.graph_preprocessor.fit_transform(mtd).to(device)
        else:
            return self.graph_preprocessor.transform(mtd).to(device)

    def ready_to_train(self, training_data) -> bool:
        return hasattr(self, "graph_denoising_net")

    def pre_training_initialization(self, training_data: HeteroData):
        self.graph_denoising_net = self.graph_denoising_net_config.build_fit(
            training_data
        ).to(get_hetero_data_device(training_data))

    def loss_generator(self, graph: HeteroData, device):

        graph.to(device)

        t_linspace = torch.linspace(
            start=0.0,
            end=1.0,
            steps=self.losses_per_epoch + 1,
            device=get_hetero_data_device(graph),
        )[:-1]

        subgraph_generator = (
            self.subgraph_generator(graph)
            if self.subgraph_generator is not None
            else None
        )

        for i in range(self.losses_per_epoch):
            subgraph = (
                next(subgraph_generator) if subgraph_generator is not None else graph
            )
            t = (
                t_linspace[i]
                if self.use_t_linspace_for_training
                else random.uniform(0.00, 1.00)
            )
            x_t, x0, t_tensor = self.graph_flow.sample(x_clean=subgraph, t=t)
            x_predicted = self.graph_denoising_net.predict_x_clean(x_t=x_t, t=t_tensor)

            yield self.graph_flow.loss(
                x_clean=subgraph,
                x_predicted=x_predicted,
                masked_subset="test",
                t=t,
                x0=x0,
            )

    def _train(self, train_data: HeteroData) -> None:
        self.graph_model.train(train_data)
        device = get_hetero_data_device(train_data)

        subgraph_generator = (
            self.subgraph_generator(train_data)
            if self.subgraph_generator is not None
            else None
        )

        self.graph_flow.conditional_tables_names += (
            list(dummy_tables(train_data)) + self.conditioning_tables
        )

        def loss(X: Tensor):
            graph = (
                next(subgraph_generator)
                if subgraph_generator is not None
                else train_data
            )
            t = X if self.use_t_linspace_for_training else torch.rand(1, device=device)

            x_t, x0, t_tensor = self.graph_flow.sample(x_clean=graph, t=t)
            x_predicted = self.graph_denoising_net.predict_x_clean(x_t=x_t, t=t_tensor)

            return self.graph_flow.loss(
                x_clean=graph, x_predicted=x_predicted, masked_subset="test", t=t, x0=x0
            )

        if self.use_t_linspace_for_training:
            X = torch.linspace(
                start=0.0,
                end=1.0,
                steps=self.losses_per_epoch + 1,
                device=device,
            )[:-1]
        else:
            X = torch.ones(size=(self.losses_per_epoch,), device=device)

        self.losses = nn_training(
            train_set=X,
            optimization=self.optimization,
            loss_function=loss,
            nn=self.graph_denoising_net,
        )

    def _generate(
        self,
        steps: int,
        device: str | torch.device,
        conditioning_dataset_config: MultiTableDatasetConfig | None = None,
        conditioning_t: float | None = None,
        x0_seed: int | None = None,
        **kwargs,
    ) -> HeteroData:

        self.graph_denoising_net.to(device)
        self.graph_denoising_net.eval()

        if conditioning_dataset_config is None:
            empty_graph = self.graph_model.generate().to(device)
            return self.ode_sampling(
                starting_graph=empty_graph, starting_t=0.0, steps=steps, x0_seed=x0_seed
            )
        else:
            conditioning_data = self.get_preprocessed_data(
                dataset=conditioning_dataset_config,
                train=False,
                fit=False,
                device=device,
            )

            if conditioning_t is None:
                raise ValueError("Conditioning time must be provided")

            x_t, x0, t_tensor = self.graph_flow.sample(
                x_clean=conditioning_data, t=torch.tensor(conditioning_t)
            )

            return self.ode_sampling(
                starting_graph=x_t,
                starting_t=conditioning_t,
                steps=steps,
                x0_seed=x0_seed,
            )

    def current_nn_device(self):
        return next(self.graph_denoising_net.parameters()).device

    def ode_sampling(
        self,
        *,
        starting_graph: HeteroData,
        steps: int,
        starting_t: float = 0.0,
        x0_seed: int | None = None,
    ) -> HeteroData:

        x_prior, x0, t_tensor = self.graph_flow.sample(
            x_clean=starting_graph, t=torch.tensor(starting_t), x0_seed=x0_seed
        )

        time_schedule = torch.linspace(
            starting_t,
            1,
            steps=steps + 1,
            device=get_hetero_data_device(x_prior),
        )[:-1]

        x_t = x_prior.clone()
        step_size = time_schedule[1] - time_schedule[0]

        with torch.no_grad():
            for t in tqdm.tqdm(time_schedule, desc="ODE sampling"):

                x1_predicted = self.graph_denoising_net.predict_x_clean(
                    x_t=x_t, t=t, softmax_discrete_features=True
                )
                v = self.graph_flow.velocity(x1=x1_predicted, t=t, xt=x_t, x0=x0)

                for name, velocity in v["x_continuous"].items():
                    x_t[name].x_continuous += velocity * step_size

                for name, velocity in v["x_discrete"].items():
                    x_t[name].x_discrete += velocity * step_size

        return x_t

    def generate(self, **kwargs) -> MultiTableDataset:
        raw_graph = self._generate(**kwargs)
        return self.graph_preprocessor.reverse_transform(raw_graph)

    def model_file(self, model_path: str):
        return f"{model_path}/graph_denoising_net.pkl"

    def _store(self, model_path: str) -> None:
        self.graph_denoising_net.to("cpu")
        pickle_store(self.graph_denoising_net, file=self.model_file(model_path))

    def store_preprocessors(self, model_path: str | Path):
        self.graph_preprocessor.store(model_path)

    def _load_(self, model_path: str) -> None:
        self.demasking_predictor = pickle_load(self.model_file(model_path))

    def specific_report_plots(self, path: Path) -> None:
        pass

    def generate_report(
        self,
        *,
        path: str | Path,
        dataset: DatasetConfig,
        generation_options: dict,
        device: str | torch.device,
    ):
        report_folder = path / Path(REPORT_FOLDER_NAME)
        os.makedirs(report_folder, exist_ok=False)

        # Store losses
        store_losses(
            folder=report_folder,
            train_losses=self.losses[0],
            validation_losses=self.losses[1],
        )
        losses_plot(
            folder=report_folder,
            train_losses=self.losses[0],
            validation_losses=self.losses[1],
        )

        mtd_original: MultiTableDataset = dataset.get()
        mtd_generated = self.generate(device=device, **generation_options)

        mtd_generated.store(path=report_folder / "generated")

        generated_features = mtd_generated.get_features(dummy_tables=False)
        original_features = mtd_original.get_features(dummy_tables=False)

        for table_name in generated_features.keys():
            table_specific_folder = report_folder / table_name
            os.makedirs(table_specific_folder, exist_ok=True)
            print(f"Comparing {table_name}...")
            compare_dataframes(
                df_generated=generated_features[table_name],
                df_original=original_features[table_name],
                path=table_specific_folder,
            )

        self.specific_report_plots(report_folder)
