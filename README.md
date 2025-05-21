## Installation
We suggest creating a Python virtual environment with Python 3.10. After activating the environment, type:
```
make install
```
to install all the requirements.

## Datasets availability
All datasets are publicly available.
Datasets are downloaded and then cached when running training.
We use datasets made available by the [SyntheRela library](https://github.com/martinjurkovic/syntherela?tab=readme-ov-file), in fact the following scripts are copied from their repository.
In order to download the data, run the command:
```
python syntherela_benchmark/download_data.py
```
In order to download precomputed results for other methods, run the command:
```
python syntherela_benchmark/download_results.py
```

## Training models
To run a specific experiment, use the command:
```
python main-train.py config.py
```
Results will be stored in the `artifacts/models` folder. Configurations can be found in the `test/configs` folder.

To compute SyntheRela's DDA metric (Discriminative Detection with Aggregation) for a given experiment, run the following script:
```
python syntherela_benchmark/benchmark_artifact.py --artifact_folder <path_of_the_experiment_folder>
```
This will save a `.json` file in the experiment folder containing a comparison of the DDA metric for the different experiments.

Alternatively, you can run all experiments as a test by running the test file `test/test_benchmark.py`:
```
python -m unittest test/test_benchmark.py
```

# Reproducing experiments

We provide scripts for reproducing the results of our article. To perform multiple trainings (multiple seeds) for each dataset, run the following command:

```
python scripts/run_all_experiments.py
```

Then, run the following command to compute the DDA metric for all these experiments:

```
python scripts/batch_sr.py
```

To aggregate the results, run the following script. This will produce a CSV file with the statistics shown in the article:

```
python scripts/aggregate_sr_results.py
```

After completing all runs, you can run the privacy check by running the following script:
```
python syntherela_benchmark/privacy_check.py
```
