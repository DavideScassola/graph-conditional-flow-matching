import itertools
import os
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import pymetis
import syntherela.metadata
import torch
from pandas import DataFrame
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_networkx

from src.data import DatasetConfig, generate_if_necessary
from src.util import (fake_float_to_int, find_foreign_keys, find_primary_key,
                      get_hetero_data_device, get_indices_from_column)


def load_multi_csv(
    folder_path: str | Path,
    categorize_fake_numerical: bool = False,
    remove_free_text: bool = True,
    category_cardinality_threshold: int = 40,
) -> dict:
    df_dict = {}

    if str(folder_path).endswith(".csv"):
        df = pd.read_csv(folder_path)
        # df['id'] = df.index
        df_dict[Path(folder_path).stem] = df

    else:
        for filename in os.listdir(folder_path):

            if filename.endswith(".csv"):
                name_without_extension = os.path.splitext(filename)[0]
                csv_path = os.path.join(folder_path, filename)
                df = pd.read_csv(csv_path)
                df_dict[name_without_extension] = df

    for table in df_dict.values():
        fake_float_to_int(table)
        handle_dates(table, max_number_of_unique_values=category_cardinality_threshold)
        remove_constant_columns(table)
        # remove_free_text_columns(table)
        if categorize_fake_numerical:
            make_fake_numerical_categorical(
                table, max_number_of_unique_values=category_cardinality_threshold
            )

    # show summary for each table
    for name, df in df_dict.items():
        print(name)
        print(df.info())

    return df_dict


@dataclass
class HeteroNode:
    feature: str
    id: int
    nx_id: int | None = None
    subset: str = "train"

    def is_train(self):
        return self.subset == "train"

    def is_test(self):
        return self.subset == "test"


class HeteroNodes:

    def __init__(self, df: DataFrame):
        self.df = df
        assert set(self.df.columns) == {
            "feature",
            "id",
            "nx_id",
            "subset",
        }
        self.df.loc[:, "subset"] = self.df["subset"].astype(str)
        self.df.loc[:, "feature"] = self.df["feature"].astype(str)
        self.df.loc[:, "id"] = self.df["id"].astype(int)
        self.df.loc[:, "nx_id"] = self.df["nx_id"].astype(int)

    def nx_ids(self):
        return self.df["nx_id"]

    def get_node(self, i: int) -> HeteroNode:
        d = self.df.iloc[i]
        return HeteroNode(**d.to_dict())

    def get_node_from_nx_id(self, nx_id: int) -> HeteroNode:
        d = self.df.loc[nx_id]
        return HeteroNode(**d.to_dict())

    def nodes_iterator(self):
        for i in range(len(self.df)):
            yield self.get_node(i)

    def feature_ids(self, feature: str):
        return self.df[self.df["feature"] == feature]["id"].to_list()

    def __len__(self):
        return len(self.df)

    @lru_cache(maxsize=None)
    def get_input_nodes(self, subset: str) -> "HeteroNodes":
        if subset == "train":
            return HeteroNodes(self.df[self.df["subset"] == "train"])
        if subset == "test":
            return self
        raise ValueError("subset must be 'train' or 'test'")

    @lru_cache(maxsize=None)
    def get_output_nodes(self, subset: str) -> "HeteroNodes":
        if subset == "train":
            return HeteroNodes(self.df[self.df["subset"] == "train"])
        if subset == "test":
            return HeteroNodes(self.df[self.df["subset"] == "test"])
        raise ValueError("subset must be 'train' or 'test'")

    def filter_subset(self, subset: str) -> "HeteroNodes":
        return HeteroNodes(self.df[self.df["subset"] == subset])


def filter_nodes(hetero_graph: HeteroData, nodes: HeteroNodes) -> HeteroData:
    out = HeteroData()
    for name, dict in hetero_graph.node_items():
        for attribute_name, x in dict.items():
            setattr(out[name], attribute_name, x[nodes.feature_ids(name)])
    return out


class MultiTableDataset:
    TRAIN_NAME = "train"
    TEST_NAME = "test"

    def __init__(self, dfs: Dict[str, pd.DataFrame], mask_id: int = -1, metadata=None):
        self.mask_id = mask_id
        self.tables_info = self._get_info(dfs, metadata=metadata)
        self.graph = None
        self._initialize_node_map(dfs)
        self._initialize_connections(dfs)
        self._initialize_features(dfs)
        self.original_columns = {name: df.columns for name, df in dfs.items()}
        self.hetero_data = None

    def names(self):
        return self.features.keys()

    def num_nodes(self):
        return len(self.node_map)

    def _initialize_node_map(self, dfs):
        self.node_map = pd.DataFrame(columns=["feature", "id", "nx_id", "subset"])
        for feature, df in dfs.items():
            self.node_map = pd.concat(
                [
                    self.node_map,
                    pd.DataFrame(
                        {
                            "feature": feature,
                            "id": np.arange(len(df)),
                            "nx_id": 0,
                            "subset": self.TRAIN_NAME,
                        }
                    ),
                ]
            )
        self.node_map["nx_id"] = np.arange(len(self.node_map))

        self.node_map.set_index(np.arange(len(self.node_map)), inplace=True)

        self.node_map = HeteroNodes(self.node_map)

    def _initialize_features(self, dfs):
        self.features = {}
        for name, df in dfs.items():
            ids = list(self.tables_info[name]["foreign_keys"].keys())
            pk = self.tables_info[name]["primary_key"]
            if pk is not None:
                ids.append(pk)
            self.features[name] = df.drop(ids, axis=1)
            if len(self.features[name].columns) == 0:
                self.features[name]["dummy_column"] = 0.0

    def _initialize_connections(self, dfs):
        self.connections = {}
        for df_name, df in dfs.items():

            self.connections[df_name] = pd.DataFrame(index=np.arange(len(df)))

            for secondary_key_col, foreign_info in self.tables_info[df_name][
                "foreign_keys"
            ].items():

                foreign_name = foreign_info["table"]
                foreign_column = foreign_info["key_column"]

                foreign_index = get_indices_from_column(
                    dfs[foreign_name],
                    alternative_index=df[secondary_key_col],
                    column=foreign_column,
                )

                self.connections[df_name][secondary_key_col] = foreign_index

    def get_dfs(self):
        connections = self.get_connections()

        primary_keys = {
            name: pd.DataFrame(
                {self.tables_info[name]["primary_key"]: conn.index}
                if self.tables_info[name]["primary_key"]
                else {}
            )
            for name, conn in connections.items()
        }

        dfs = {
            name: pd.concat(
                [primary_keys[name], connections[name], self.features[name]], axis=1
            ).drop("dummy_column", axis=1, errors="ignore")
            for name in self.names()
        }

        return dfs

    def get_connections(self):
        return self.connections

    def get_features(self, dummy_tables: bool, subset: str = "all"):

        if subset in ["train", "test"]:
            out = {
                name: df.iloc[
                    self.node_map.df["id"][
                        (self.node_map.df["feature"] == name)
                        & (self.node_map.df["subset"] == subset)
                    ]
                ]
                for name, df in self.features.items()
            }
        else:
            assert (
                subset == "all"
            ), f"subset must be 'train', 'test' or 'all', not {subset}"
            out = self.features

        if not dummy_tables:
            return {
                name: df for name, df in out.items() if "dummy_column" not in df.columns
            }

        return out

    def set_features(self, features):
        self.features = features

    def _get_info(
        self,
        dfs: Dict[str, pd.DataFrame],
        metadata: syntherela.metadata.Metadata | None = None,
    ) -> Dict[str, Dict[str, str]]:
        info = {}
        if isinstance(metadata, syntherela.metadata.Metadata):
            for table_name in dfs.keys():
                info[table_name] = {}
                info[table_name]["foreign_keys"] = {}
                info[table_name]["primary_key"] = metadata.get_primary_key(table_name)
            for r in metadata.relationships:
                # info[r["parent_table_name"]]["primary_key"] = r["parent_primary_key"]
                info[r["child_table_name"]]["foreign_keys"][r["child_foreign_key"]] = {
                    "table": r["parent_table_name"],
                    "key_column": r["parent_primary_key"],
                }
        elif isinstance(metadata, dict):
            return metadata
        else:
            for ref_name, df_ref in dfs.items():
                info[ref_name] = {}
                info[ref_name]["primary_key"] = find_primary_key(df_ref)
                info[ref_name]["foreign_keys"] = find_foreign_keys(
                    df_ref, {name: df for name, df in dfs.items() if name != ref_name}
                )
        return info

    def _build_graph_from_tables(self) -> nx.DiGraph:
        return to_networkx(self.toHeteroData())

    def get_networkx_graph(self):
        if self.graph is None:
            self.graph = self._build_graph_from_tables()
        return self.graph

    def nx_id_to_hd_id(self, nx_ids: list[int]) -> HeteroNodes:
        return HeteroNodes(self.node_map.df.loc[nx_ids])

    def bfs_traversal_nodes(
        self,
        *,
        children_are_roots=True,
        root_node: HeteroNode | None = None,
        randomize=False,
    ) -> HeteroNodes:

        # TODO: implement randomization

        graph = self.get_networkx_graph()

        # Maybe cache this method
        if root_node is not None:
            root = root_node
        else:
            random_index = random.randint(0, len(self.get_input_nodes().df) - 1)
            root = self.get_input_nodes().get_node(random_index)

        bfs_edges = list(nx.bfs_edges(graph, source=root.nx_id))
        traversal_nodes = [root.nx_id] + [v for u, v in bfs_edges]

        return self.nx_id_to_hd_id(traversal_nodes)

    def filter_nodes(self, nodes_to_remove):
        if nodes_to_remove is None:
            return {name: df.index for name, df in self.get_dfs().items()}
        else:
            idx = {}
            for name, df in self.get_dfs().items():
                nds = [node for node in nodes_to_remove if node.startswith(name + ":")]
                idx[name] = df.index[~df.index.isin(nds)]
            return idx

    def get_input_nodes(self, subset="train") -> HeteroNodes:
        return self.node_map.get_input_nodes(subset)

    def get_output_nodes(self, subset="train") -> HeteroNodes:
        return self.node_map.get_output_nodes(subset)

    def toHeteroData(
        self, device=None, undirected=True, only_connections: bool = False
    ) -> HeteroData:

        data = HeteroData()

        if not only_connections:
            for name in self.get_dfs().keys():
                features = self.get_features()[name]
                if isinstance(features, Tensor):
                    data[name].x = self.get_features()[name].to(device)
                else:
                    # dummy values
                    data[name].x = torch.zeros(len(features), 1, device=device)

        for name in self.get_dfs().keys():
            s = self.node_map.df["subset"][self.node_map.df["feature"] == name]
            data[name].subset = torch.tensor(subset_name_to_int(s))

        for name, connection in self.get_connections().items():
            for foreign_key_colname in connection.columns:
                index1 = torch.tensor(
                    connection.index.values, dtype=torch.long, device=device
                )
                index2 = torch.tensor(
                    connection[foreign_key_colname].values,
                    dtype=torch.long,
                    device=device,
                )

                foreign_name = self.tables_info[name]["foreign_keys"][
                    foreign_key_colname
                ]["table"]
                data[
                    name, f"{foreign_key_colname}_refers_to", foreign_name
                ].edge_index = torch.stack([index1, index2], dim=0)

                if undirected:
                    data[
                        foreign_name, f"is_referred_by_{foreign_key_colname}", name
                    ].edge_index = torch.stack([index2, index1], dim=0)

        return data.to(device) if device is not None else data

    def get_nodes(self, nx_nodes: list[int]):
        return self.node_map.df.loc[nx_nodes]

    def get_masked_hetero_data(self, nx_masked_nodes: list[int]):
        masked_hetero_data: HeteroData = self.toHeteroData().clone()

        nodes = self.get_nodes(nx_nodes=nx_masked_nodes)

        for feature in masked_hetero_data.keys():
            feature_nodes = nodes[nodes["feature"] == feature]
            masked_hetero_data[feature].x[feature_nodes] = self.mask_id

        return masked_hetero_data

    def get_boundary_nodes(self, node: HeteroNode) -> HeteroNodes:

        raise NotImplementedError
        return self.node_map.loc[self.graph.neighbors(node.nx_id)]

    def set_test_nodes(self, test_nodes: HeteroNodes | list[int]):
        self.node_map.df.loc[:, "subset"] = self.TRAIN_NAME
        test_idx = (
            test_nodes.nx_ids() if isinstance(test_nodes, HeteroNodes) else test_nodes
        )
        self.node_map.df.loc[test_idx, "subset"] = self.TEST_NAME

    def store(self, path: str | Path):
        os.makedirs(path)
        for name, df in self.get_dfs().items():

            for column in df.columns:
                if df[column].dtype == "category":
                    df[column] = df[column].astype(str).replace("nan", "")

            df.to_csv(os.path.join(path, name + ".csv"), index=False, na_rep="")


def get_neighbours(graph: HeteroData, node: HeteroNode) -> HeteroNodes:
    nx_graph = to_networkx(graph)
    neighbors = list(nx_graph.neighbors(node.nx_id))
    return nx_to_hetero_nodes(neighbors, graph)


def bfs_nodes_splitting(
    mtd: MultiTableDataset, *, train_proportion: float, children_are_roots=True
) -> Tuple[MultiTableDataset, MultiTableDataset]:

    traversal_nodes = mtd.bfs_traversal_nodes(children_are_roots=children_are_roots)

    # Split the list of nodes into two parts
    cut = int(len(traversal_nodes.df) * train_proportion)
    test_nodes = HeteroNodes(traversal_nodes.df[cut:])

    mtd.set_test_nodes(test_nodes)


def nx_to_hetero_nodes(nx_ids: list[int], graph: HeteroData) -> HeteroNodes:
    node_df = pd.DataFrame(
        columns=["feature", "id", "nx_id", "masked_loss", "masked_input"]
    )
    node_df["nx_id"] = np.arange(graph.num_nodes, dtype=int)
    node_df["masked_loss"] = False
    node_df["masked_input"] = False

    node_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "feature": feature_name,
                    "id": np.arange(len(feature.x)),
                    "nx_id": None,
                    "masked_loss": None,
                    "masked_input": None,
                }
            )
            for feature_name, feature in graph.x_dict.items()
        ]
    )

    node_df["nx_id"] = np.arange(len(node_df))
    node_df.set_index(np.arange(len(node_df)))
    return HeteroNodes(node_df.loc[nx_ids])


def shuffle_edges(graph: HeteroData) -> None:
    for feature_name, feature in graph.x_dict.items():
        perm = torch.randperm(feature.edge_index.size(1))
        feature.edge_index = feature.edge_index[:, perm]


def bfs_traversal(graph: HeteroData, starting_node: HeteroNode | None) -> HeteroNodes:
    if starting_node is None:
        starting_node = HeteroNode(feature=list(graph.keys())[0], id=0)

    # Shuffle esges in order to avoid bias while selecting nodes with same priority in BFS
    shuffle_edges(graph=graph)

    nx_graph = to_networkx(graph)
    bfs_edges = list(nx.bfs_edges(nx_graph, source=starting_node.nx_id))
    traversal_nodes = [starting_node.nx_id] + [v for u, v in bfs_edges]
    return nx_to_hetero_nodes(nx_ids=traversal_nodes, graph=graph)


def random_splitting(mtd: MultiTableDataset, *, train_proportion: float, seed: int):
    n = mtd.num_nodes()
    index = np.arange(n).tolist()
    random.seed(seed)
    random.shuffle(index)
    cut = int(train_proportion * n)
    test_index = index[cut:]
    mtd.set_test_nodes(test_index)


def per_table_random_splitting(
    mtd: MultiTableDataset, *, train_proportion: float, seed: int
):
    for name in mtd.names():
        df = mtd.node_map.df[mtd.node_map.df["feature"] == name]
        n = len(df)
        test_index = np.arange(n)
        np.random.shuffle(test_index)
        test_index = test_index[int(train_proportion * n) :]
        loc_indexes = df.iloc[test_index].index
        mtd.node_map.df.loc[loc_indexes, "subset"] = mtd.TEST_NAME


def no_splitting(mtd: MultiTableDataset, *, train_proportion: float):
    pass


def is_syntherela_dataset(path: str | Path) -> bool:
    return "syntherela" in str(path).lower()


def make_fake_numerical_categorical(
    table: pd.DataFrame, max_number_of_unique_values: int = 100
):
    def is_numerical(s: pd.Series):
        return s.dtype not in [str, "object", "category"]

    for column in table.columns:
        if (
            is_numerical(table[column])
            and len(table[column].unique()) <= max_number_of_unique_values
        ):
            table[column] = table[column].astype("category")


def remove_constant_columns(table: pd.DataFrame):
    for column in table.columns:
        if len(table[column].unique()) == 1:
            print(f"Removing constant column {column}")
            table.drop(column, axis=1, inplace=True)


def remove_free_text_columns(
    table: pd.DataFrame, max_number_of_unique_values: int = 20
):
    # TODO: this is a very naive implementation

    def is_key_column(column):
        return column[-2:].lower() == "id"  # TODO: not a real id column identifier!

    def is_free_text_column(column):
        return (
            table[column].dtype in [object, str]
            and len(table[column].unique()) > max_number_of_unique_values
        )

    for column in table.columns:
        if not is_key_column(column) and is_free_text_column(column):
            print(
                f"Removing column {column} ({len(table[column].unique())} categories)"
            )
            table.drop(column, axis=1, inplace=True)


def handle_dates(df: pd.DataFrame, max_number_of_unique_values=10):
    """
    Transforms each date in column into separate columns for years, months and days.
    """

    def is_date_column(column):
        return "date" in column.lower()  # TODO: not a real date column identifier!

    def is_timestamp_column(column):
        return "timestamp" in column.lower()

    for c in df.columns:

        if (is_date_column(c) or is_timestamp_column(c)) and len(
            df[c].unique()
        ) <= max_number_of_unique_values:
            df[c] = df[c].astype(str).astype("category")

        elif is_date_column(c):
            date = pd.to_datetime(df[c]).dt
            df[c + "_year"] = date.year.astype("category")
            df[c + "_month"] = date.month.astype("category")
            df[c + "_day"] = date.day.astype("category")
            df.drop(c, axis=1, inplace=True)

        elif is_timestamp_column(c):
            date = pd.to_datetime(df[c]).dt
            df[c + "_year"] = date.year.astype("category")
            df[c + "_month"] = date.month.astype("category")
            df[c + "_day"] = date.day.astype("category")
            df[c + "_hour"] = date.hour.astype(int)
            df[c + "_minute"] = date.minute.astype(int)
            df[c + "_second"] = date.second.astype(int)
            df.drop(c, axis=1, inplace=True)


def load_syntherela_dataset(
    path: str | Path,
    categorize_fake_numerical: bool = False,
    category_cardinality_threshold: int = 40,
) -> MultiTableDataset:
    from syntherela.data import load_tables, remove_sdv_columns
    from syntherela.metadata import Metadata

    # read data
    metadata = Metadata().load_from_json(f"{path}/metadata.json")
    tables = load_tables(f"{path}/", metadata)
    tables, metadata = remove_sdv_columns(tables, metadata)
    metadata.validate_data(tables)

    for table in tables.values():
        handle_dates(table)

        if categorize_fake_numerical:
            make_fake_numerical_categorical(
                table, max_number_of_unique_values=category_cardinality_threshold
            )

    for name, df in tables.items():
        print(name)
        print(df.info())

    return MultiTableDataset(tables, metadata=metadata)


@dataclass
class MultiTableDatasetConfig(DatasetConfig):
    split_strategy: Callable = per_table_random_splitting
    categorize_fake_numerical: bool = True
    category_cardinality_threshold: int = 40
    split_seed: int | None = None

    def get(self) -> MultiTableDataset:

        if hasattr(self, "cached_mtd"):
            return (
                self.cached_mtd
            )  # TODO: Once it was cached, but then pickling gets too heavy

        generate_if_necessary(self.path)

        if is_syntherela_dataset(self.path):
            mtd = load_syntherela_dataset(
                self.path,
                categorize_fake_numerical=self.categorize_fake_numerical,
                category_cardinality_threshold=self.category_cardinality_threshold,
            )
        else:
            mtd = MultiTableDataset(
                load_multi_csv(
                    Path(self.path),
                    categorize_fake_numerical=self.categorize_fake_numerical,
                    category_cardinality_threshold=self.category_cardinality_threshold,
                )
            )

        self.split_strategy(
            mtd, train_proportion=self.train_proportion, seed=self.split_seed
        )

        return mtd


def subset_name_to_int(set_name: str | pd.Series):
    d = {"train": 0, "test": 1}
    if isinstance(set_name, str):
        return d[set_name]
    else:
        return pd.Series(d).loc[set_name].values


def int_to_mask(*, int_encoding: Tensor, masked_subset: str | None) -> Tensor:
    """
    masked_subset is the name the set ("test", "train", etc) that will be masked (putting zeros)
    """
    if masked_subset is None:
        return torch.ones_like(int_encoding, device=int_encoding.device)
    return torch.where(
        int_encoding == subset_name_to_int(set_name=masked_subset),
        torch.zeros_like(int_encoding, device=int_encoding.device),
        torch.ones_like(int_encoding, device=int_encoding.device),
    )


def get_graph_mask(
    graph: HeteroData, feature_name: str, masked_subset: str | None
) -> Tensor:
    return int_to_mask(
        int_encoding=graph[feature_name].subset, masked_subset=masked_subset
    ).float()


def get_hetero_data_type_ranges(hetero_data: HeteroData):
    types_ranges = {}
    i = 0
    x_name = list(
        set(hetero_data.keys()).difference(
            {"edge_index", "num_nodes", "subset", "discrete_feature_slices"}
        )
    )[0]
    for feature_name in hetero_data.node_types:
        l = len(hetero_data[feature_name][x_name])
        types_ranges[feature_name] = (i, i + l)
        i += l
        hetero_data[feature_name]["num_nodes"] = l  # required for to_networkx
    return types_ranges


def fast_to_networkx(hetero_data: HeteroData):
    G = nx.Graph()

    types_ranges = get_hetero_data_type_ranges(hetero_data)

    # Add nodes
    for node_type, (start, end) in types_ranges.items():
        G.add_nodes_from(np.arange(start, end))

    # Add edges
    for edge_type in hetero_data.edge_types:
        node_type_source, _, node_type_target = edge_type
        edges = hetero_data[edge_type].edge_index.t().cpu().numpy()
        edges = edges + np.array(
            [types_ranges[node_type_source][0], types_ranges[node_type_target][0]]
        )
        G.add_edges_from(edges)

    return G


def to_subset_dict(nodes, *, types_ranges: Dict[str, Tuple[int, int]], device):
    nodes = torch.tensor(nodes, dtype=torch.long, device=device)
    return {
        k: nodes[(nodes >= v[0]) & (nodes < v[1])] - v[0]
        for k, v in types_ranges.items()
    }


def connected_components_subgraphs_generator(
    hetero_data: HeteroData, *, min_nodes: int = 1
):
    device = get_hetero_data_device(hetero_data)
    types_ranges = get_hetero_data_type_ranges(hetero_data)

    nx_graph = fast_to_networkx(hetero_data)  # to_networkx(hetero_data).to_undirected()

    connected_components = [
        list(nx_subgraph) for nx_subgraph in nx.connected_components(nx_graph)
    ]
    connected_components.reverse()

    subset_dicts = []

    while len(connected_components) > 0:
        subgraph = []
        while len(subgraph) < min_nodes and len(connected_components) > 0:
            subgraph.extend(connected_components.pop())
        subset_dicts.append(
            to_subset_dict(subgraph, types_ranges=types_ranges, device=device)
        )

    while True:
        random.shuffle(subset_dicts)
        for subset_dict in subset_dicts:
            yield hetero_data.subgraph(subset_dict)


def bfs_nodes(*, nx_graph: nx.Graph, starting_node: int, max_nodes: int):
    bfs_edges = list(
        itertools.islice(nx.bfs_edges(nx_graph, source=starting_node), max_nodes)
    )
    if len(bfs_edges) == 0:
        return np.array([starting_node])
    max_nodes = min(max_nodes, len(bfs_edges) + 1)
    traversal_nodes = np.zeros(max_nodes, dtype=int)
    traversal_nodes[0] = starting_node
    traversal_nodes[1:] = np.array(bfs_edges)[: max_nodes - 1, 1]
    return traversal_nodes


def bfs_nodes_with_jumps(
    *, nx_graph: nx.Graph, force_starting_node: int, max_nodes: int
):
    nodes = []
    unseen_nodes = np.arange(nx_graph.number_of_nodes())
    np.random.shuffle(unseen_nodes)
    t = unseen_nodes[0]
    unseen_nodes[0] = force_starting_node
    unseen_nodes = set(unseen_nodes)

    tot_nodes = 0
    while tot_nodes < max_nodes and len(unseen_nodes) > 0:
        start_node = unseen_nodes.pop()
        if start_node == force_starting_node and len(unseen_nodes) > 0:
            start_node = t
        new_nodes = bfs_nodes(
            nx_graph=nx_graph, starting_node=start_node, max_nodes=max_nodes - tot_nodes
        )
        unseen_nodes = unseen_nodes.difference(new_nodes)
        tot_nodes = tot_nodes + len(new_nodes)
        nodes.append(new_nodes)
    return np.concatenate(nodes)


def bfs_graph_generator(hetero_data: HeteroData, *, max_nodes: int):
    device = get_hetero_data_device(hetero_data)
    types_ranges = get_hetero_data_type_ranges(hetero_data)
    nx_graph = fast_to_networkx(hetero_data)
    starting_nodes = list(nx_graph.nodes)
    random.shuffle(starting_nodes)
    while True:
        for starting_node in starting_nodes:
            nodes = bfs_nodes_with_jumps(
                nx_graph=nx_graph,
                force_starting_node=starting_node,
                max_nodes=max_nodes,
            )
            yield hetero_data.subgraph(
                to_subset_dict(nodes, types_ranges=types_ranges, device=device)
            )


def metis_graph_generator(hetero_data: HeteroData, *, partitions: int):
    device = get_hetero_data_device(hetero_data)
    types_ranges = get_hetero_data_type_ranges(hetero_data)
    nx_graph = fast_to_networkx(hetero_data)
    adjacency_list = [list(nx_graph.neighbors(node)) for node in nx_graph.nodes()]

    print("\nMETIS partitioning...")
    n_cuts, membership = pymetis.part_graph(partitions, adjacency=adjacency_list)
    print(
        f"METIS: {n_cuts} cuts, {partitions} partitions of ~{len(nx_graph)//partitions} nodes"
    )
    membership = np.array(membership)
    subgraphs = [np.where(membership == i)[0] for i in range(partitions)]

    dicts = [
        to_subset_dict(subgraph, types_ranges=types_ranges, device=device)
        for subgraph in subgraphs
    ]
    while True:
        random.shuffle(dicts)
        for d in dicts:
            yield hetero_data.subgraph(d)


def get_edge_indices(hetero_data: HeteroData):
    edge_indices = []
    for edge_info in hetero_data.edge_items():
        if "refers_to" in edge_info[0][1]:
            actual_length = len(hetero_data[edge_info[0][0]].subset)
            edge_index = edge_info[1]["edge_index"]
            assert torch.all(
                edge_index[0] == torch.arange(actual_length, device=edge_index.device)
            )

            edge_indices.append(
                {
                    "connection_name": edge_info[0][1],
                    "child_table_name": edge_info[0][0],
                    "referred_table_name": edge_info[0][2],
                    "edge_index": edge_index[1],
                    "referred_table_size": len(hetero_data[edge_info[0][2]].subset),
                }
            )

    return edge_indices


def get_neighbours_count_tables(hetero_data: HeteroData, log_count: bool = False):

    edge_indices = get_edge_indices(hetero_data)

    # for each refered table, build the table where the rows are the Ids of the refered table, and the columns are the number of incoming edges from each other table
    edge_count_tables = {
        edge_index_info["referred_table_name"]: {} for edge_index_info in edge_indices
    }

    # incoming edges
    for edge_index_info in edge_indices:
        referred_table = edge_index_info["referred_table_name"]
        edge_name = f"{edge_index_info['child_table_name']}_{edge_index_info['connection_name']}"
        edge_count_tables[referred_table][edge_name] = torch.bincount(
            edge_index_info["edge_index"],
            minlength=edge_index_info["referred_table_size"],
        )

    edge_count_tables = {
        table_name: torch.stack(list(table.values()), dim=0).T.float()
        for table_name, table in edge_count_tables.items()
    }

    if log_count:
        edge_count_tables = {
            name: torch.log(table + 1) for name, table in edge_count_tables.items()
        }

    return edge_count_tables


def get_node_degree_stats(graph: HeteroData, log_count: bool = True):
    neighbour_count_tables = get_neighbours_count_tables(graph, log_count=log_count)

    degrees_stats = {}

    for name, table in neighbour_count_tables.items():

        mean_degree = table.float().mean(dim=0)
        std_degree = table.float().std(dim=0)
        degrees_stats[name] = {"mean": mean_degree, "std": std_degree}

    return degrees_stats


def dummy_tables(hetero_graph: HeteroData) -> set:

    def is_empty(x):
        return x.x_discrete.numel() == 0 and x.x_continuous.numel() == 0

    def no_variance(x):
        return x.x_discrete.std().item() == 0 and x.x_continuous.std().item() == 0

    return {
        feature_name
        for feature_name in hetero_graph.metadata()[0]
        if is_empty(hetero_graph[feature_name])
        or no_variance(hetero_graph[feature_name])
    }
