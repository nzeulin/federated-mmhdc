import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


def encode_labels(
    y_train: torch.Tensor,
    y_test: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, LabelEncoder]:
    encoder = LabelEncoder()
    train_np = y_train.detach().cpu().reshape(-1).numpy()
    y_train_enc = torch.as_tensor(encoder.fit_transform(train_np), dtype=torch.long)

    if y_test is None:
        return y_train_enc, None, encoder

    test_np = y_test.detach().cpu().reshape(-1).numpy()
    y_test_enc = torch.as_tensor(encoder.transform(test_np), dtype=torch.long)
    return y_train_enc, y_test_enc, encoder


def split_iid(y: torch.Tensor, num_clients: int, seed: int) -> list[torch.Tensor]:
    y_cpu = y.detach().cpu().reshape(-1)
    if num_clients < 1:
        raise ValueError("num_clients must be at least 1.")
    if num_clients == 1:
        return [torch.arange(y_cpu.numel(), dtype=torch.long)]
    splitter = StratifiedKFold(n_splits=num_clients, shuffle=True, random_state=seed)
    dummy_x = torch.zeros(y_cpu.numel()).numpy()
    y_np = y_cpu.numpy()
    return [
        torch.as_tensor(test_idx, dtype=torch.long)
        for _, test_idx in splitter.split(dummy_x, y_np)
    ]


def split_noniid(
    y: torch.Tensor,
    num_clients: int,
    classes_per_client: int = 2,
    seed: int = 0,
) -> list[torch.Tensor]:
    y_cpu = y.detach().cpu().reshape(-1).to(dtype=torch.long)
    if num_clients < 1:
        raise ValueError("num_clients must be at least 1.")
    classes = torch.unique(y_cpu, sorted=True)
    if num_clients < classes.numel():
        raise ValueError(
            "Non-IID splitting requires num_clients >= number of classes to "
            "assign every class at least once."
        )
    if classes_per_client < 1:
        raise ValueError("classes_per_client must be at least 1.")
    if classes_per_client > classes.numel():
        raise ValueError(
            "classes_per_client cannot exceed the number of dataset classes."
        )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    num_classes = classes.numel()
    if classes_per_client == 1 and num_clients == num_classes:
        client_labels = classes[torch.randperm(num_classes, generator=generator)].unsqueeze(0)
    else:
        while True:
            client_labels = torch.stack(
                [
                    classes[
                        torch.randperm(num_classes, generator=generator)[
                            :classes_per_client
                        ]
                    ]
                    for _ in range(num_clients)
                ]
            ).T.contiguous()
            if torch.unique(client_labels).numel() == num_classes:
                break

    max_label = int(classes.max().item())
    class_counts = torch.bincount(client_labels.flatten(), minlength=max_label + 1)
    shards_by_class: dict[int, list[torch.Tensor]] = {}
    for cls in classes.tolist():
        cls_indices = torch.nonzero(y_cpu == cls, as_tuple=False).flatten()
        split_count = int(class_counts[cls].item())
        shards_by_class[int(cls)] = list(torch.tensor_split(cls_indices, split_count))

    client_indices: list[torch.Tensor] = []
    for client_id in range(num_clients):
        parts = []
        for row in range(classes_per_client):
            cls = int(client_labels[row, client_id].item())
            parts.append(shards_by_class[cls].pop())
        client_indices.append(torch.cat(parts).to(dtype=torch.long))

    return client_indices


from .fedavg import FedAvg, FedAvgResult

__all__ = ["FedAvg", "FedAvgResult", "encode_labels", "split_iid", "split_noniid"]
