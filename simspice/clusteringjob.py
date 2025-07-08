import numpy as np
import hdbscan

stacked_outputs = np.load('saved_outputs//stacked_outputs_single64_full_gain01-3.npy')

# L2 normalize for cosine metric
norm_outputs = np.linalg.norm(stacked_outputs, ord=2)

for x in [5, 10, 20, 30]:
    for y in [2, 5, 10]:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=x, min_samples=y, metric='euclidean')
        clusterer.fit(stacked_outputs)
        labels = clusterer.labels_
        np.save(f'saved_outputs//labels-cos_single64_fulldata_grange01-3_minclus{x}_minsamp{y}.npy', labels)
