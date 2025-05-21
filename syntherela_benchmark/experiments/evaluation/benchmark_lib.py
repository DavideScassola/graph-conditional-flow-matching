import logging
import sys

from sklearn.linear_model import LogisticRegression
from syntherela.benchmark import Benchmark
from syntherela.metrics.multi_table.detection import (
    AggregationDetection,
    ParentChildAggregationDetection,
    ParentChildDetection,
)
from syntherela.metrics.multi_table.statistical import CardinalityShapeSimilarity
from syntherela.metrics.single_column.detection import SingleColumnDetection
from syntherela.metrics.single_column.distance import (
    HellingerDistance,
    JensenShannonDistance,
    TotalVariationDistance,
    WassersteinDistance,
)
from syntherela.metrics.single_column.statistical import (
    ChiSquareTest,
    KolmogorovSmirnovTest,
)
from syntherela.metrics.single_table.detection import SingleTableDetection
from syntherela.metrics.single_table.distance import (
    MaximumMeanDiscrepancy,
    PairwiseCorrelationDifference,
)
from xgboost import XGBClassifier

FOLDER = "syntherela_benchmark"


def run_benchmark(
    *, dataset_name: str, methods: list[str], run_id: int, full_test: bool
):

    logger = logging.getLogger(f"{dataset_name}_logger")

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info(f"START LOGGING Dataset: {dataset_name}")

    xgb_cls = XGBClassifier
    xgb_args = {"seed": 0}
    rf_args = {"random_state": 0, "n_estimators": 100}
    logistic = LogisticRegression
    logistic_args = {"random_state": 0}
    single_column_metrics = [
        ChiSquareTest(),
        KolmogorovSmirnovTest(),
        TotalVariationDistance(),
        HellingerDistance(),
        JensenShannonDistance(),
        WassersteinDistance(),
        SingleColumnDetection(
            classifier_cls=xgb_cls, classifier_args=xgb_args, random_state=42
        ),
        # SingleColumnDetection(classifier_cls=rf_cls, classifier_args=rf_args),
        SingleColumnDetection(
            classifier_cls=logistic, classifier_args=logistic_args, random_state=42
        ),
    ]
    single_table_metrics = [
        MaximumMeanDiscrepancy(),
        PairwiseCorrelationDifference(),
        SingleTableDetection(
            classifier_cls=xgb_cls, classifier_args=xgb_args, random_state=42
        ),
        # SingleTableDetection(classifier_cls=rf_cls, classifier_args=rf_args),
        SingleTableDetection(
            classifier_cls=logistic, classifier_args=logistic_args, random_state=42
        ),
    ]
    multi_table_metrics = [
        CardinalityShapeSimilarity(),
        AggregationDetection(
            classifier_cls=xgb_cls, classifier_args=xgb_args, random_state=42
        ),
        # AggregationDetection(classifier_cls=rf_cls, classifier_args=rf_args),
        AggregationDetection(
            classifier_cls=logistic, classifier_args=logistic_args, random_state=42
        ),
        ParentChildDetection(
            classifier_cls=xgb_cls, classifier_args=xgb_args, random_state=42
        ),
        ParentChildDetection(
            classifier_cls=logistic, classifier_args=logistic_args, random_state=42
        ),
        ParentChildAggregationDetection(
            classifier_cls=xgb_cls, classifier_args=xgb_args, random_state=42
        ),
        ParentChildAggregationDetection(
            classifier_cls=logistic, classifier_args=logistic_args, random_state=42
        ),
    ]

    benchmark = Benchmark(
        real_data_dir=f"{FOLDER}/data/original",
        synthetic_data_dir=f"{FOLDER}/data/synthetic",
        results_dir=f"{FOLDER}/results/{run_id}",
        benchmark_name="Benchmark",
        single_column_metrics=single_column_metrics if full_test else [],
        single_table_metrics=single_table_metrics if full_test else [SingleTableDetection(
            classifier_cls=xgb_cls, classifier_args=xgb_args, random_state=42
        ),],
        multi_table_metrics=multi_table_metrics if full_test else [AggregationDetection(
            classifier_cls=xgb_cls, classifier_args=xgb_args, random_state=42
        )],
        run_id=run_id,
        sample_id="sample1",
        datasets=[dataset_name],
        methods=methods,
    )

    benchmark.run()
