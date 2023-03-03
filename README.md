# JET
> **J**aunty **E**stimation of Hierarchical **T**ime Series Clustering

## Installation

```shell
python setup.py install
```

## Usage

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
    metric=JETMetric.SHAPE_BASED_DISTANCE,      # distance metric for time series distances; Options: SHAPE_BASED_DISTANCE, MSM, DTW
    c: float = 700                              # cost parameter for MSM distance metric
)

# returns cluster label for each time series
labels = jet.fit_predict(list_of_time_series)

# plot the dendrogram
dendrogram(jet._ward_clustering._linkage_matrix)
plt.show()
```

## Reference

> tbd
