import math

import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor


def int_series2digits(s: pd.Series, base: int = 10) -> torch.Tensor:

    assert (s >= 0).all(), "All values must non-negative"

    # Convert the series to a tensor of integers
    integers = torch.tensor(s.values, dtype=torch.int64)

    # Find the maximum number of digits needed
    max_value = integers.max().item()
    digits = math.ceil(math.log(max_value + 1, base))

    # Create a tensor of powers of the base
    powers = base ** torch.arange(digits - 1, -1, -1, dtype=torch.int64)

    # Calculate the digits matrix
    digits_matrix = (integers.unsqueeze(-1) // powers) % base

    return digits_matrix


def digits2int_series(x: torch.Tensor, base: int = 10) -> pd.Series:
    # Get the number of digits
    num_digits = x.shape[1]

    # Create a tensor of powers of the base
    powers = base ** torch.arange(
        num_digits - 1, -1, -1, device=x.device, dtype=torch.float32
    )

    # Calculate the numbers by performing matrix multiplication
    numbers = x @ powers.long()

    return pd.Series(numbers.cpu().numpy())


def float2digits(x: Tensor, base: int = 2, base10digits: int = 4) -> Tensor:
    assert x.dtype == torch.float32, "x must be float32"
    n = 10**base10digits
    digits = math.ceil(base10digits * math.log(10, base))
    integers = torch.round(x * n).type(torch.int32)

    powers = base ** torch.arange(digits, 0, -1)
    digits_matrix = (
        base * torch.remainder(integers.unsqueeze(-1), powers.reshape(1, -1)) // powers
    )
    return digits_matrix.type(torch.int64)


def digits2float(x: Tensor, base: int = 2, base10digits: int = 4) -> Tensor:
    digits = math.ceil(base10digits * math.log(10, base))
    n = 10**base10digits
    powers = base ** torch.arange(digits - 1, -1, -1, device=x.device)
    return x.float() @ (powers.float()) / n

    def __init__(
        self,
        base: int = 2,
        base10digits: int = 4,
        one_hot: bool = False,
        flatten: bool = True,
    ) -> None:
        super().__init__()
        self.base = base
        self.base10digits = base10digits
        self.one_hot = one_hot
        self.flatten = flatten
        self.hypercube_normalizer = HypercubeNormalizer()

    def fit(self, x: Tensor) -> None:
        self.hypercube_normalizer.fit(x)
        self.parameters["hypercube_normalizer"] = self.hypercube_normalizer.parameters
        if self.flatten:
            self.parameters["shape"] = list(x[0].shape)

    def transform(self, x: Tensor) -> Tensor:
        self.hypercube_normalizer.parameters = self.parameters["hypercube_normalizer"]

        digits = float2digits(
            self.hypercube_normalizer.transform(x), self.base, self.base10digits
        )

        if not self.one_hot:
            return digits.flatten(1) if self.flatten else digits

        out = (
            F.one_hot(digits, num_classes=self.base)
            if self.base > 2
            else digits.unsqueeze(-1)
        )

        return out

    def reverse_transform(self, x: Tensor) -> Tensor:
        self.hypercube_normalizer.parameters = self.parameters["hypercube_normalizer"]

        x = x.reshape([x.shape[0]] + self.parameters["shape"] + [-1])
        digits = torch.argmax(x, dim=-1) if self.base > 2 and self.one_hot else x
        return self.hypercube_normalizer.reverse_transform(
            digits2float(digits, self.base, self.base10digits)
        )
