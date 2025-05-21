import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch import Tensor
from torch_geometric.data import HeteroData

from src.graph_data import MultiTableDataset
from src.preprocessors.quantizer import digits2int_series, int_series2digits
from src.util import infer_series_type

from .preprocessor import Preprocessor

INT_IS_NUMERICAL_THRESHOLD = 20
DIGITS_BASE = 2


class GraphToOneHot(Preprocessor):

    def __init__(
        self,
        digits_encoding_for_int: bool = False,
        digits_encoding_for_float: bool = False,
        int_encoding: str = "flattened_one_hot",
        one_hot_to_cat_method: str = "argmax",
    ):
        self.parameters = {}
        self.parameters["digits_encoding_for_int"] = digits_encoding_for_int
        self.parameters["digits_encoding_for_float"] = digits_encoding_for_float
        self.parameters["int_encoding"] = int_encoding

        assert one_hot_to_cat_method in [
            "argmax",
            "sample",
        ], f"one_hot_to_cat_method should be either 'argmax' or 'sample', but got {one_hot_to_cat_method}"
        self.parameters["one_hot_to_cat_method"] = one_hot_to_cat_method

        if self.parameters["digits_encoding_for_float"]:
            raise NotImplementedError(
                "Digits encoding for float is not implemented yet"
            )

    def string_to_int(self, c: pd.Series, *, classes: dict) -> Tensor:
        le = LabelEncoder()
        le.classes_ = np.array(classes)
        return torch.tensor(le.transform(c), dtype=torch.long)

    def int_array_to_string_column(self, c: np.ndarray, *, classes: dict) -> pd.Series:
        le = LabelEncoder()
        le.classes_ = np.array(classes)
        return pd.Series(le.inverse_transform(c), name="column_name")

    def get_continuous_features_tensor(
        self, df: pd.DataFrame, table_parameters: dict
    ) -> Tensor:
        numerical_columns = table_parameters["float_columns"] + (
            table_parameters["numerical_int_columns"]
            if not self.parameters["digits_encoding_for_int"]
            else []
        )
        if len(numerical_columns) == 0:
            return torch.empty(len(df), 0)
        return torch.tensor(df[numerical_columns].to_numpy(), dtype=torch.float32)

    def get_discrete_features_tensor(
        self,
        df: pd.DataFrame,
        *,
        int_encoding: str,
        table_parameters: dict,
        table_name: str,
    ) -> Tensor:
        if (
            len(
                table_parameters["categorical_columns"]
                + table_parameters["digits_encoded_columns"]
            )
            == 0
        ):
            return torch.empty(len(df), 0)
        discrete_columns = {}
        for column_name in table_parameters["string_columns"]:
            discrete_columns[column_name] = self.string_to_int(
                df[column_name],
                classes=table_parameters["columns_metadata"][column_name]["classes"],
            ).unsqueeze(1)

        for column_name in table_parameters["categorical_int_columns"]:
            discrete_columns[column_name] = self.string_to_int(
                df[column_name].astype(str),
                classes=table_parameters["columns_metadata"][column_name]["classes"],
            ).unsqueeze(1)

        for column_name in table_parameters["bool_columns"]:
            discrete_columns[column_name] = self.string_to_int(
                df[column_name].astype(str),
                classes=table_parameters["columns_metadata"][column_name]["classes"],
            ).unsqueeze(1)

        digits_encoding_used = False
        if self.parameters["digits_encoding_for_int"]:
            for column_name in table_parameters["numerical_int_columns"]:
                discrete_columns[column_name] = int_series2digits(
                    df[column_name],
                    base=DIGITS_BASE,
                )
                digits_encoding_used = True

        int_tensor = torch.cat(list(discrete_columns.values()), dim=1)

        if int_encoding == "one_hot":

            max_num_classes = max(
                [
                    len(column_metadata.get("classes", [1, 2]))
                    for column_metadata in table_parameters["columns_metadata"].values()
                ]
            )

            if digits_encoding_used:
                max_num_classes = max(max_num_classes, 10)

            return torch.nn.functional.one_hot(
                int_tensor, num_classes=max_num_classes
            ).float()

        if int_encoding == "flattened_one_hot":
            one_hots = [
                torch.nn.functional.one_hot(
                    discrete_columns[column_name].squeeze(),
                    num_classes=len(
                        table_parameters["columns_metadata"][column_name]["classes"]
                    ),
                ).float()
                for column_name in table_parameters["categorical_columns"]
            ]

            assert len(one_hots) == len(table_parameters["categorical_columns"])

            start = len(one_hots)

            # the rest are digits
            if self.parameters["digits_encoding_for_int"]:
                for i in range(start, int_tensor.size(1)):  # This could be vectorized
                    one_hots.append(
                        torch.nn.functional.one_hot(
                            int_tensor[:, i],
                            num_classes=10,  # TODO: This should be a parameter somewhere
                        ).float()
                    )

            assert len(one_hots) == int_tensor.size(1)

            return torch.cat(one_hots, dim=1)

        return int_tensor.unsqueeze(1)

    def fit(self, mtd: MultiTableDataset):
        self.parameters["table_parameters"] = {}

        self.parameters["tables_info"] = mtd.tables_info

        for table_name, df in mtd.features.items():
            self.parameters["table_parameters"][table_name] = {}
            self.parameters["table_parameters"][table_name]["types"] = df.dtypes.apply(
                str
            ).to_dict()
            self.parameters["table_parameters"][table_name][
                "columns"
            ] = df.columns.to_list()

            self.parameters["table_parameters"][table_name]["string_columns"] = []
            self.parameters["table_parameters"][table_name][
                "categorical_int_columns"
            ] = []

            self.parameters["table_parameters"][table_name]["bool_columns"] = []

            self.parameters["table_parameters"][table_name]["float_columns"] = []
            self.parameters["table_parameters"][table_name][
                "numerical_int_columns"
            ] = []

            self.parameters["table_parameters"][table_name]["columns_metadata"] = {}

            if df.columns[0] != "dummy_column":
                for column_name, series in df.items():
                    self.parameters["table_parameters"][table_name]["columns_metadata"][
                        column_name
                    ] = {}
                    series_type = infer_series_type(series)
                    self.parameters["table_parameters"][table_name][
                        f"{series_type}_columns"
                    ].append(column_name)
                    self.parameters["table_parameters"][table_name]["columns_metadata"][
                        column_name
                    ]["type"] = series_type
                    if series_type == "string":
                        self.parameters["table_parameters"][table_name][
                            "columns_metadata"
                        ][column_name]["classes"] = sorted(list(series.unique()))
                    if series_type in ["categorical_int", "bool"]:
                        self.parameters["table_parameters"][table_name][
                            "columns_metadata"
                        ][column_name]["classes"] = sorted(
                            list(series.unique().astype(str))
                        )

                    if (
                        self.parameters["digits_encoding_for_int"]
                        and series_type == "numerical_int"
                    ):
                        self.parameters["table_parameters"][table_name][
                            "columns_metadata"
                        ][column_name]["max_digits"] = int_series2digits(
                            df[column_name], base=DIGITS_BASE
                        ).shape[
                            1
                        ]

                    if (
                        self.parameters["digits_encoding_for_float"]
                        and series_type == "float"
                    ):
                        assert False, "Digits encoding for float is not implemented yet"
                        # TODO: Implement digits encoding for float
                        self.parameters["table_parameters"][table_name][
                            "columns_metadata"
                        ][column_name]["max_digits"] = ...

            self.parameters["table_parameters"][table_name]["categorical_columns"] = (
                self.parameters["table_parameters"][table_name]["string_columns"]
                + self.parameters["table_parameters"][table_name]["bool_columns"]
                + self.parameters["table_parameters"][table_name][
                    "categorical_int_columns"
                ]
            )
            self.parameters["table_parameters"][table_name][
                "digits_encoded_columns"
            ] = (
                self.parameters["table_parameters"][table_name]["numerical_int_columns"]
                if self.parameters["digits_encoding_for_int"]
                else []
            )

        def exclude_dummy_column(columns):
            return list(filter(lambda x: x != "dummy_column", columns))

        # Remembering exact columns order
        self.parameters["table_parameters"]["all_columns"] = {
            name: exclude_dummy_column(df.columns.to_list())
            for name, df in mtd.get_dfs().items()
        }

        return self.parameters

    def discrete_feature_slices(self):
        slices = {}

        for table_name in self.parameters["table_parameters"]["all_columns"].keys():
            tp = self.parameters["table_parameters"][table_name]
            slices[table_name] = []
            start = 0
            end = 0
            for column_name in tp["categorical_columns"]:
                metadata = tp["columns_metadata"][column_name]
                if "classes" in metadata:
                    end += len(metadata["classes"])
                    slices[table_name].append((start, end))
                    start = end

            for column_name in tp["digits_encoded_columns"]:

                digits = self.parameters["table_parameters"][table_name][
                    "columns_metadata"
                ][column_name]["max_digits"]

                for _ in range(digits):
                    end += 10  # TODO: This should be a paremeter somewhere
                    slices[table_name].append((start, end))
                    start = end

        return slices

    def transform(self, mtd: MultiTableDataset) -> HeteroData:
        graph = mtd.toHeteroData(only_connections=True)
        for table_name, df in mtd.features.items():
            graph[table_name].x_discrete = self.get_discrete_features_tensor(
                df,
                int_encoding=self.parameters["int_encoding"],
                table_parameters=self.parameters["table_parameters"][table_name],
                table_name=table_name,
            )
            graph[table_name].x_continuous = self.get_continuous_features_tensor(
                df, table_parameters=self.parameters["table_parameters"][table_name]
            )

        if self.parameters["int_encoding"] == "flattened_one_hot":
            graph.discrete_feature_slices = self.discrete_feature_slices()

        return graph

    def reset_types(self, df: pd.DataFrame, types: dict):
        for c in df.columns:
            df[c] = df[c].astype(types[c])

    def get_df_from_tensors(
        self,
        discrete_tensor: Tensor,
        continuous_tensor: Tensor,
        table_parameters: dict,
        table_name: str,
    ) -> pd.DataFrame:
        df = pd.DataFrame()

        if table_parameters["columns"][0] == "dummy_column":
            df["dummy_column"] = np.zeros(len(discrete_tensor), dtype=int)
            return df

        if self.parameters["one_hot_to_cat_method"] == "sample":
            raise NotImplementedError("This is experimental code")

            def one_hot_to_cat(x: Tensor) -> Tensor:
                x[x < 0] = 1e-20
                x = torch.log(x / x.sum(-1, keepdim=True))
                return torch.nn.functional.gumbel_softmax(x, tau=1.0, hard=True).argmax(
                    dim=-1
                )

        elif self.parameters["one_hot_to_cat_method"] == "argmax":

            def one_hot_to_cat(x: Tensor) -> Tensor:
                return x.argmax(dim=-1)

        else:
            raise ValueError(
                f"Unknown one_hot_to_cat_method: {self.parameters['one_hot_to_cat_method']}"
            )

        if discrete_tensor.size(1) > 0:

            if self.parameters["int_encoding"] == "flattened_one_hot":
                int_categories = np.concatenate(
                    [
                        one_hot_to_cat(discrete_tensor[:, s[0] : s[1]])
                        .unsqueeze(1)
                        .cpu()
                        .numpy()
                        for s in self.discrete_feature_slices()[table_name]
                    ],
                    axis=1,
                )

            elif self.parameters["int_encoding"] == "one_hot":
                int_categories = one_hot_to_cat(discrete_tensor).cpu().numpy()
            else:
                int_categories = discrete_tensor.cpu().numpy()

            start_index = 0

            for i, column_name in enumerate(table_parameters["string_columns"]):
                df[column_name] = self.int_array_to_string_column(
                    int_categories[:, start_index],
                    classes=table_parameters["columns_metadata"][column_name][
                        "classes"
                    ],
                )
                start_index += 1

            for i, column_name in enumerate(
                table_parameters["categorical_int_columns"]
            ):
                df[column_name] = self.int_array_to_string_column(
                    int_categories[:, start_index],
                    classes=table_parameters["columns_metadata"][column_name][
                        "classes"
                    ],
                ).astype(int)
                start_index += 1

            for i, column_name in enumerate(table_parameters["bool_columns"]):
                df[column_name] = (
                    self.int_array_to_string_column(
                        int_categories[
                            :,
                            start_index,
                        ],
                        classes=table_parameters["columns_metadata"][column_name][
                            "classes"
                        ],
                    )
                    == "True"
                )
                start_index += 1

            for i, column_name in enumerate(table_parameters["digits_encoded_columns"]):
                digits = table_parameters["columns_metadata"][column_name]["max_digits"]
                df[column_name] = digits2int_series(
                    x=torch.tensor(
                        int_categories[:, start_index : start_index + digits]
                    ),
                    base=DIGITS_BASE,
                )
                start_index += digits

        if continuous_tensor.size(1) > 0:
            for i, column_name in enumerate(table_parameters["float_columns"]):
                df[column_name] = continuous_tensor[:, i].cpu().numpy()

            if not self.parameters["digits_encoding_for_int"]:
                for i, column_name in enumerate(
                    table_parameters["numerical_int_columns"]
                ):
                    df[column_name] = np.round(
                        continuous_tensor[:, len(table_parameters["float_columns"]) + i]
                        .cpu()
                        .numpy()
                    ).astype(int)

        self.reset_types(df, table_parameters["types"])

        return df[table_parameters["columns"]]

    def reverse_transform(self, graph: HeteroData) -> MultiTableDataset:

        graph.cpu()

        dfs = {
            table_name: self.get_df_from_tensors(
                discrete_tensor=graph[table_name].x_discrete,
                continuous_tensor=graph[table_name].x_continuous,
                table_parameters=self.parameters["table_parameters"][table_name],
                table_name=table_name,
            )
            for table_name in graph.metadata()[0]
        }

        self.add_connections(dfs=dfs, graph=graph)
        self.remove_dummy_columns(dfs=dfs)
        self.sort_columns(dfs=dfs)

        mtd = MultiTableDataset(dfs, metadata=self.parameters["tables_info"])

        assert (
            mtd.tables_info == self.parameters["tables_info"]
        ), "Tables info should be the same"

        return mtd

    def add_connections(self, *, dfs: dict, graph: HeteroData) -> None:
        for table_name in dfs:

            primary_key_name = self.parameters["tables_info"][table_name]["primary_key"]
            if primary_key_name:
                dfs[table_name][primary_key_name] = np.arange(
                    len(dfs[table_name])
                ).astype(int)

            for foreign_key, referenced_table in self.parameters["tables_info"][
                table_name
            ]["foreign_keys"].items():
                edges_tensor = (
                    graph[
                        table_name,
                        f"{foreign_key}_refers_to",
                        referenced_table["table"],
                    ]["edge_index"]
                    .long()
                    .cpu()
                    .numpy()
                )
                assert edges_tensor.shape[1] == len(
                    dfs[table_name]
                ), "A foreign key should be defined for every row"
                dfs[table_name][foreign_key] = edges_tensor[
                    1
                ]  # TODO: this could be problematic when not all entries have a foreign key

    def remove_dummy_columns(self, dfs: dict) -> None:
        for name in dfs:
            if "dummy_column" in dfs[name].columns:
                # Removing the dummy column
                del dfs[name]["dummy_column"]

    def sort_columns(self, dfs: dict) -> None:
        for name in dfs:
            dfs[name] = dfs[name][
                self.parameters["table_parameters"]["all_columns"][name]
            ]
