from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()
    config.name = "fedavg_cwru"

    # Dataset parameters
    config.dataset = ConfigDict()
    config.dataset.name = "cwru"
    config.dataset.num_classes = 10
    config.dataset.model_dim = 5_000
    config.dataset.download = True
    config.dataset.data_root = "cache/cwru/raw"
    config.dataset.manifest_path = "configs/datasets/cwru_manifest.json"
    config.dataset.cache_dir = "cache/cwru/cache"
    config.dataset.sensor_channel = "FE"
    config.dataset.loads = [1,]
    config.dataset.fault_diameters = [7, 14, 21]
    config.dataset.outer_race_position = "6"
    config.dataset.window_size = 100
    config.dataset.train_split = 0.7
    config.dataset.train_stride = 20
    config.dataset.test_stride = 20
    config.dataset.return_metadata = False
    config.dataset.return_class_mapping = False

    # HD transform parameters
    config.transform = ConfigDict()
    config.transform.seed = 0
    config.transform.normalize_input = True
    config.transform.normalize_hypervectors = True
    config.transform.batch_size = 1_000

    config.model = ConfigDict()

    # Common model parameters
    config.model.method = "onlinehd"
    config.model.learning_rate = 1e-2

    # OnlineHD parameters
    config.model.init_aggregation = "norm"

    # MMHDC parameters
    config.model.C = 500.0
    config.model.margin_width = 1
    config.model.no_margin = False
    config.model.backend = "cpp"

    # Federated learning parameters
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
    config.training.num_experiments = 10
    config.training.eval_global_epochs = 1

    config.reproducibility = ConfigDict()
    config.reproducibility.base_seed = 0

    config.output = ConfigDict()
    config.output.results_dir = "results/fedavg"
    config.output.results_filename = "results.pt"
    config.output.plot_filename = "accuracy.pdf"
    config.output.report_filename = "report.md"

    config.device = "cuda"

    return config
