# JET
> **J**aunty **E**stimation of Hierarchical **T**ime Series Clustering

## Installation

```shell
python setup.py install
```

## Usage

_JET_ is a _Scikit-Learn_ `BaseEstimator` with `ClusterMixin` class. It has a `fit_predict` function that expects a list of time series (`List[np.ndarray]`) that will be clustered. The available time series distance measures are _Shape Based Distance_, _Move Split Merge_, and _Dynamic Time Warping_. These measure can handle only univariate time series. Therefore, JET can handle only univariate time series, too. 

```python
import numpy as np
from jet import JET, JETMetric

from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt

# generate 100 random example time series with lengths between 30 and 50
list_of_time_series = [np.random.rand(np.random.randint(30, 50)) for _ in range(100)]

jet = JET(
    n_clusters=10,                              # number of clusters to find: $c$ in paper
    n_pre_clusters=None,                        # number of pre-clusters to find: $c_{pre}$ in paper; default is $3\sqrt{n}$ (3*np.sqrt(len(X))) if None is set
    n_jobs=1,                                   # number of parallel jobs
    verbose=False,                              # output status messages
    metric=JETMetric.SHAPE_BASED_DISTANCE,      # distance metric for time series distances; Options: SHAPE_BASED_DISTANCE, MSM, DTW, or custom
    c = 700                                     # cost parameter for MSM distance metric
)

# returns cluster label for each time series
labels = jet.fit_predict(list_of_time_series)

# plot the dendrogram
dendrogram(jet._ward_clustering._linkage_matrix)
plt.show()
```

### Bring Your Own Distance Measure

You can define your own distance measure function as shown below. (This enables you to cluster also multivariate time series if you have a suitable measure!)

```python
import numpy as np
from jet import JET, JETMetric

def custom_distance_measure(x: np.ndarray, y: np.ndarray) -> float:
    min_len = min(len(x), len(y))
    distance = np.power(x[:min_len] - y[:min_len], 2)
    return distance

jet = JET(
    n_clusters=10,
    metric=JETMetric(custom_distance_measure)
)
```

## Experiments

Code for the experiments was created with [Tidewater](https://github.com/HPI-Information-Systems/tidewater) and is described in the [README](https://github.com/HPI-Information-Systems/tidewater#paper-experiments).

## Reference

> tbd
