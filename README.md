# q2-time
![CI](https://github.com/adamovanja/q2-time/blob/main/.github/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/gh/adamovanja/q2-time/branch/main/graph/badge.svg?token=VQ4D7FXMCB)](https://codecov.io/gh/adamovanja/q2-time)

Longitudinal modeling approaches accounting for high-dimensional, sparse and compositional nature of microbial time-series.

## Setup
<!-- TODO: replace plugin name "q2-time" with something better-->
To install the required dependencies for this package run:
```shell
make create-env
conda activate time
make dev
```

## Background
### Why ray tune?
"By using tuning libraries such as Ray Tune we can try out combinations of hyperparameters. Using sophisticated search strategies, these parameters can be selected so that they are likely to lead to good results (avoiding an expensive exhaustive search). Also, trials that do not perform well can be preemptively stopped to reduce waste of computing resources. Lastly, Ray Tune also takes care of training these runs in parallel, greatly increasing search speed." [source](https://docs.ray.io/en/latest/tune/examples/tune-xgboost.html#tune-xgboost-ref)
