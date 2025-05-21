from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData

from src.graph_data import get_graph_mask
from src.util import cross_entropy, get_hetero_data_device


class DiffusionProcess(ABC):
    @abstractmethod
    def sample(self, *, x0: Tensor, t: Tensor | None = None) -> Tensor:
        pass


class GraphFlow(DiffusionProcess):

    MIN_VARIANCE_WEIGHT = 1e-2

    def __init__(
        self,
        one_hot_prior: str = "gaussian",
        sigma_min: float = 0.0,
        weighted_loss: bool = False,
        objective: str = "x1-x0_prediction",
        conditional_tables_names: list[str] = [],
    ) -> None:
        self.one_hot_prior = one_hot_prior
        self.sigma_min = sigma_min
        self.weighted_loss = weighted_loss
        self.objective = objective
        self.conditional_tables_names = conditional_tables_names

    def conditional_flow(self, x: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        # TODO: This is probably not correct for dirichlet flows
        return (1 - (1 - self.sigma_min) * t) * x + t * x1

    def xt_based_velocity(self, *, x1: Tensor, xt: Tensor, t: Tensor) -> Tensor:
        # from CATflow paper
        return (x1 - (1 - self.sigma_min) * xt) / (1 - (1 - self.sigma_min) * t)

    def x0_based_velocity(self, *, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        # from original paper
        return x1 - (1 - self.sigma_min) * x0

    def velocity(
        self, *, x1: HeteroData, t: Tensor, xt: HeteroData, x0: HeteroData
    ) -> dict:
        v = {"x_continuous": {}, "x_discrete": {}}
        for name in x1.metadata()[0]:
            # For discrete features, follow CATFlow (x1 prediction)
            if xt[name].x_discrete.shape[1] > 0:
                v["x_discrete"][name] = (
                    self.xt_based_velocity(
                        xt=xt[name].x_discrete, x1=x1[name].x_discrete, t=t
                    )
                    if name not in self.conditional_tables_names
                    else 0.0
                )

            if xt[name].x_continuous.shape[1] > 0:
                # For continuous features, the velocity depends on the objective
                if self.objective == "x1_prediction":
                    v["x_continuous"][name] = (
                        self.xt_based_velocity(
                            xt=xt[name].x_continuous,
                            x1=x1[name].x_continuous,
                            t=t,
                        )
                        if name not in self.conditional_tables_names
                        else 0.0
                    )

                elif self.objective == "x1-x0_prediction":
                    # In this case the prediction is x1 - x0, so it is the velocity
                    v["x_continuous"][name] = (
                        x1[name].x_continuous
                        if name not in self.conditional_tables_names
                        else 0.0
                    )
                else:
                    raise ValueError(f"Objective {self.objective} not recognized")
        return v

    def sample(
        self,
        *,
        x_clean: HeteroData,
        t: Tensor | None = None,
        x0_seed: int | None = None,
    ) -> tuple[HeteroData, HeteroData, torch.Tensor]:

        device = get_hetero_data_device(x_clean)

        t_tensor = torch.rand(1, device=device) if t is None else t

        x_t = x_clean.clone()
        x_0 = self.sample_prior(x_clean, seed=x0_seed)

        for name in x_clean.x_discrete_dict.keys():
            if name not in self.conditional_tables_names:
                x_t[name].x_discrete = self.conditional_flow(
                    x1=x_clean[name].x_discrete, x=x_0[name].x_discrete, t=t_tensor
                )

        for name in x_clean.x_continuous_dict.keys():
            if name not in self.conditional_tables_names:
                x_t[name].x_continuous = self.conditional_flow(
                    x1=x_clean[name].x_continuous, x=x_0[name].x_continuous, t=t_tensor
                )

        return x_t, x_0, t_tensor

    def sample_prior(self, graph: HeteroData, seed: int | None) -> HeteroData:
        x_0 = graph.clone()
        if seed is not None:
            torch.manual_seed(seed)
        for name in graph.x_discrete_dict.keys():
            x_0[name].x_discrete = torch.randn_like(x_0[name].x_discrete)
        for name in graph.x_continuous_dict.keys():
            x_0[name].x_continuous = torch.randn_like(x_0[name].x_continuous)
        return x_0

    def default_prediction_variance(self, t: float) -> float:
        return max((1 - (1 - self.sigma_min) * t) ** 2, self.MIN_VARIANCE_WEIGHT)

    def loss(
        self,
        x_clean: HeteroData,
        x_predicted: HeteroData,
        masked_subset: str,
        t: float | Tensor,
        x0: HeteroData | None = None,
    ) -> tuple[Tensor, float]:
        device = get_hetero_data_device(x_clean)
        train_loss = torch.tensor(0.0, device=device)
        test_loss = torch.tensor(0.0, device=device)

        # TODO: check if this can be made more efficient
        for name in x_predicted.x_discrete_dict.keys():
            if torch.numel(x_clean[name].x_discrete):
                loss_mask = get_graph_mask(
                    graph=x_clean, feature_name=name, masked_subset=masked_subset
                )

                discrete_feature_slices = getattr(
                    x_clean, "discrete_feature_slices", None
                )
                if discrete_feature_slices:
                    discrete_loss = sum(
                        [
                            (
                                cross_entropy(
                                    logits=x_predicted[name].x_discrete[:, s[0] : s[1]],
                                    targets=x_clean[name]
                                    .x_discrete[:, s[0] : s[1]]
                                    .argmax(-1),
                                )
                            )
                            for s in discrete_feature_slices[name]
                        ]
                    )
                else:
                    discrete_loss = cross_entropy(
                        logits=x_predicted[name].x_discrete,
                        targets=x_clean[name].x_discrete.argmax(-1),
                    ).sum(
                        dim=1
                    )  # TODO: maybe it's not efficient to do the argmax

                train_discrete_loss = (
                    discrete_loss[loss_mask == 1]
                    if torch.any(loss_mask == 1)
                    else torch.tensor(0.0, device=device)
                )
                test_discrete_loss = (
                    discrete_loss[loss_mask == 0]
                    if torch.any(loss_mask == 0)
                    else torch.tensor(0.0, device=device)
                )
                assert torch.any(loss_mask == 0) and torch.any(loss_mask == 1)
                train_loss.add_(train_discrete_loss.mean())
                test_loss.add_(test_discrete_loss.mean())

        for name in x_predicted.x_continuous_dict.keys():
            if x_clean[name].x_continuous.shape[1] > 0:
                loss_mask = get_graph_mask(
                    graph=x_clean, feature_name=name, masked_subset=masked_subset
                )

                if self.objective == "x1_prediction":
                    target = x_clean[name].x_continuous
                elif self.objective == "x1-x0_prediction":
                    assert x0 is not None, "x0 is required for velocity parametrization"
                    target = (
                        x_clean[name].x_continuous
                        - (1 - self.sigma_min) * x0[name].x_continuous
                    )
                else:
                    raise ValueError(f"Unknown objective: {self.objective}")

                gaussian_loss = F.mse_loss(
                    target,
                    x_predicted[name].x_continuous,
                    reduction="none",
                ).sum(-1)

                if self.weighted_loss and self.objective == "x1_prediction":
                    # This is equivalent to the log-likehood of a gaussian distribution with the predicted mean and fixed variance (variance depends on t)
                    gaussian_loss = gaussian_loss / self.default_prediction_variance(t)

                train_gaussian_loss = gaussian_loss[loss_mask == 1]
                test_gaussian_loss = gaussian_loss[loss_mask == 0]

                train_loss.add_(train_gaussian_loss.mean())
                test_loss.add_(test_gaussian_loss.mean())
        return train_loss, test_loss.detach().item()
