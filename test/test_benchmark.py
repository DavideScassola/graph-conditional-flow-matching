# test_math_operations.py
import sys
import unittest

sys.path.append(".")
from scripts.remove_duplicate_refs import remove_duplicates_from_artifact_folder
from src.experiment_controller import run_experiment
from syntherela_benchmark.benchmark_artifact import DEFAULT_METHOD_NAME

class TestBenchmark(unittest.TestCase):
        
    def paper_benchmark(self, config_file: str):
        """
        Run the experiment with the given config file, run syntherela benchmark and check if results are SOTA
        """
        folder_path = run_experiment(config_file)
        
        from syntherela_benchmark.benchmark_artifact import run as get_benchmark_stat
        results = get_benchmark_stat(folder_path, method_name=DEFAULT_METHOD_NAME)
        self.assertLessEqual(results[DEFAULT_METHOD_NAME], results['benchmark'], msg=f"Results for {config_file} are not SOTA")

    def test_airbnb(self):
        self.paper_benchmark("test/configs/sb_airbnb.py")
        
    def test_CORA_v1(self):
        config_file = "test/configs/sb_CORA_v1.py"
        
        
        """
        Run the experiment with the given config file, run syntherela benchmark and check if results are SOTA
        """
        folder_path = run_experiment(config_file)
        from syntherela_benchmark.benchmark_artifact import run as get_benchmark_stat

        pre_results = get_benchmark_stat(folder_path, method_name=DEFAULT_METHOD_NAME)
        remove_duplicates_from_artifact_folder(folder_path, table_name="content.csv", columns=["paper_id", "word_cited_id"])
        
        results = get_benchmark_stat(folder_path, method_name=DEFAULT_METHOD_NAME+"NoDuplicates")
        self.assertLessEqual(results[DEFAULT_METHOD_NAME], results['benchmark'], msg=f"Results for {config_file} are not SOTA")

        
    def test_imdb_MovieLens_v1(self):
        self.paper_benchmark("test/configs/sb_imdb_MovieLens_v1.py")
        
    def test_rossmann_subsampled(self):
        self.paper_benchmark("test/configs/sb_rossmann_subsampled.py")

    def test_walmart_subsampled(self):
        self.paper_benchmark("test/configs/sb_walmart_subsampled.py")
        
    def test_biodegradability(self):
        self.paper_benchmark("test/configs/sb_biodegradability.py")


if __name__ == "__main__":
    unittest.main()
