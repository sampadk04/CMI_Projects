# Fast_Text_based_Clustering

- We look at various techniques of clustering while using the *Jaccard Similarity* as a measure of closeness.
- We first used the built-in `KMeans` by processing the documents as binary vectors with dimension equal to the no. of words in the vocabulary. We used this for `KOS` and `NIPS`
- Since, the above techniques turned out to be too slow or unreliable (the `MiniBatchKMeans`) for `Enron`, we created 2 algorithms from scratch to tackle while considering the *Jaccard Distance*. This turned out to be more efficient and accurate with better scores and clusters.
- Based on the clustering and score obtained by the above algorithms, we decide on the appropriate no. of clusters per dataset.
- This would help us in semi-supervised learning to learn from huge partially labelled datasets.
