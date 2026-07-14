from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()
    config.name = "fedavg_cwru"

    config.dataset = ConfigDict()
    config.dataset.name = "cwru"
    config.dataset.num_classes = 10
    config.dataset.model_dim = 5_000
    config.dataset.download = True
    config.dataset.data_root = "cache/cwru/raw"
    config.dataset.manifest_path = "configs/datasets/cwru_manifest.json"
    config.dataset.cache_dir = "cache/cwru/cache"
    config.dataset.sensor_channel = "DE"
    config.dataset.loads = [1, 2, 3]
    config.dataset.fault_diameters = [7, 14, 21]
    config.dataset.outer_race_position = "6"
    config.dataset.window_size = 100
    config.dataset.train_candidate_stride = 1
    config.dataset.test_stride = 100
    config.dataset.train_windows_per_group = 660
    config.dataset.test_windows_per_group = 25
    config.dataset.return_metadata = False
    config.dataset.return_class_mapping = False
    config.dataset.seed = 42

    config.transform = ConfigDict()
    config.transform.seed = 0
    config.transform.normalize = False
    config.transform.batch_size = None

    config.model = ConfigDict()
    config.model.learning_rate = 1e-4
    config.model.C = 500.0
    config.model.margin_width = 1
    config.model.no_margin = False
    config.model.backend = "cpp"

    config.fl = ConfigDict()
    config.fl.method = ["fedavg"]
    config.fl.num_clients = 20
    config.fl.noniid = False
    config.fl.classes_per_client = 10
    config.fl.chunks = [1, 2, 5, 10]
    config.fl.batch_size = 1_000
    config.fl.shuffle = False

    config.training = ConfigDict()
    config.training.global_epochs = 100
    config.training.local_epochs = 10
    config.training.num_experiments = 1
    config.training.eval_global_epochs = 1

    config.reproducibility = ConfigDict()
    config.reproducibility.base_seed = 0

    config.output = ConfigDict()
    config.output.results_dir = "results/fedavg"
    config.output.results_filename = "fedavg_cwru_results.pt"
    config.output.plot_filename = "fedavg_cwru_accuracy.pdf"

    config.device = "cuda"

    return config
