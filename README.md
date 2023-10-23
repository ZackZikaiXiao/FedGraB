# Fed-GraB: Federated Long-tailed Learning with Self-Adjusting Gradient Balancer

This is the code for paper "Fed-GraB: Federated Long-tailed Learning with Self-Adjusting Gradient Balancer".

## Parameters

| parameters        | description                                                  |
| ----------------- | ------------------------------------------------------------ |
| `rounds`          | Number of rounds in training process, option:`500`           |
| `num_users`       | Number of clients, option:`40`,`20`                          |
| `local_bs`        | Batch size for local training, option:`5`                    |
| `beta`            | Coefficient for local proximal term, option: `0.01`,`0`      |
| `model`           | neural network model, option: `resnet18`,`resnet34`,`resnet50` |
| `dataset`         | Dataset, option:`cifar10`,`cifar100`,`imagenet` and `inat`   |
| `iid`             | `Action` iid or non iid, option: `store_true`                |
| `alpha_dirichlet` | Parameter for Dirichlet distribution, option: `10`,`1`       |


## Usage

+ To train on CIFAR-10 with IID data partition and imbalanced factor 100 over 40 clients:

```
python fed_grab.py --dataset cifar10 --iid --IF 0.01 --local_bs 5 --rounds 500 --num_users 40 --beta 0 --dataset cifar10  --model resnet18 --gpu 0
```

+ To train on CIFAR-10 with non-IID data partition with imbalanced factor 100 , alpha=1 over 40 clients:

```
python fed_grab.py --dataset cifar10 -alpha_dirichlet 1 --IF 0.01 --local_bs 5 --rounds 500 --num_users 40 --beta 0 --dataset cifar10  --model resnet18 --gpu 0
```
 


