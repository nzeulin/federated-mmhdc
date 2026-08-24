from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()
    config.name = "fedavg_uci_har"

    # Dataset parameters
    config.dataset = ConfigDict()
    config.dataset.name = "uci-har"
    config.dataset.num_classes = 6
    config.dataset.model_dim = 5_000
    config.dataset.download = True
    config.dataset.data_root = "cache/har"

    # HD transform parameters
    config.transform = ConfigDict()
    config.transform.seed = 0
    config.transform.normalize_input = True
    config.transform.normalize_hypervectors = False
    config.transform.batch_size = 1_000

    config.model = ConfigDict()

    # Common model parameters
    config.model.method = "onlinehd"
    config.model.learning_rate = 1e-1

    # OnlineHD parameters
    config.model.init_aggregation = "sum"

    # MMHDC parameters
    config.model.C = 500.0
    config.model.margin_width = 1
    config.model.no_margin = False
    config.model.backend = "cpp"

    # Federated learning parameters
    config.fl = ConfigDict()
    config.fl.method = ["fedavg"]
    config.fl.num_clients = 20
    config.fl.noniid = True
    config.fl.classes_per_client = 2
    config.fl.chunks = [1, 2, 5, 10]
    config.fl.batch_size = 1_000
    config.fl.shuffle = False

    config.training = ConfigDict()
    config.training.global_epochs = 100
    config.training.local_epochs = 5
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
