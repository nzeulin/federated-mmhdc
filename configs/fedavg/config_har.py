from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()
    config.name = "fedavg_uci_har"

    config.dataset = ConfigDict()
    config.dataset.name = "uci-har"
    config.dataset.num_classes = 6
    config.dataset.model_dim = 5_000
    config.dataset.download = True
    config.dataset.data_root = "cache/har"

    config.transform = ConfigDict()
    config.transform.seed = 0
    config.transform.normalize = True
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
    config.fl.classes_per_client = 6
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
    config.output.results_filename = "results.pt"
    config.output.plot_filename = "accuracy.pdf"
    config.output.report_filename = "report.md"

    config.device = "cuda"

    return config
