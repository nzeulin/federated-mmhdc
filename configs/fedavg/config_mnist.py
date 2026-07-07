from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()
    config.name = "fedavg_mnist"

    config.dataset = ConfigDict()
    config.dataset.name = "mnist"
    config.dataset.num_classes = 10
    config.dataset.model_dim = 5_000
    config.dataset.download = True
    config.dataset.data_root = "data"

    config.transform = ConfigDict()
    config.transform.seed = 0
    config.transform.normalize = True
    config.transform.batch_size = None

    config.model = ConfigDict()
    config.model.learning_rate = 1e-2
    config.model.C = float("inf")
    config.model.margin_width = 0.25
    config.model.no_margin = True
    config.model.backend = "cpp"

    config.fl = ConfigDict()
    config.fl.method = "fedavg"
    config.fl.num_clients = 20
    config.fl.noniid = False
    config.fl.classes_per_client = 10
    config.fl.chunks = 1
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
    config.output.results_filename = "fedavg_mnist_results.pt"
    config.output.plot_filename = "fedavg_mnist_accuracy.png"

    config.device = "cuda"

    return config
