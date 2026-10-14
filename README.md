# Resource-Efficient Federated Hyperdimensional Computing

This repository contains the implementation of resource-efficient federated hyperdimensional computing (RE-FHDC) algorithm for [maximum-margin HDC](https://github.com/nzeulin/mmhdc) (MM-HDC) and OnlineHD classifiers.

**How to run**

An example for MNIST dataset with i.i.d. class distribution can be run as follows:

```bash
git clone https://github.com/nzeulin/federated-mmhdc.git
cd federated-mmhdc
python -m pip install -r requirements.txt
python main.py --config configs/fedavg/iid/config_mnist.py
```

The codebase supports three datasets: MNIST, UCI HAR, and CWRU, with i.i.d. and non-i.i.d. class distributions of clients. The corresponding configuration files can be found in `configs/fedavg`. 

The implementation of "stylish" plots from the paper can be found in the branch [mmhdc/extended-repotring](https://github.com/nzeulin/federated-mmhdc/tree/extended-reporting). In other aspects, the implementation is identical to the main branch.

### Citation

If you use this repository in your research, please cite our papers:

```bibtex
@article{zeulin2026federated,
  title={Federated Hyperdimensional Computing for Resource-Constrained Industrial IoT},
  author={Zeulin, Nikita and Galinina, Olga and Himayat, Nageen and Andreev, Sergey},
  journal={accepted to IEEE Internet of Things Magazine},
  year={2026}
}

@inproceedings{zeulin2023resource,
  title={Resource-Efficient Federated Hyperdimensional Computing},
  author={Zeulin, Nikita and Galinina, Olga and Himayat, Nageen and Andreev, Sergey},
  booktitle={MLSys 2023 Federated Learning Systems Workshop (FLSys)},
  pages={1--8},
  year={2023}
}
```
