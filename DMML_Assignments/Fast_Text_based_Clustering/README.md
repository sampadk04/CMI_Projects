# Fast Text-based Clustering using Bag of Words

This project explores various clustering techniques applied to text data, specifically using the "Bag of Words" dataset from the UCI Machine Learning Repository. The primary focus is on using **Jaccard Similarity** as a measure of closeness between documents.

## Project Overview

The goal is to cluster documents from five text collections (Enron emails, NIPS blog entries, KOS blog entries, etc.) and determine the optimal number of clusters ($K$).

### Key Strategies
1.  **K-Means with Euclidean Distance**: Initially, the built-in `KMeans` algorithm from `sklearn` is used. Documents are treated as binary vectors (presence/absence of words). This approach was applied to the `KOS` and `NIPS` datasets.
2.  **Custom Jaccard-based Clustering**: Since standard K-Means (and `MiniBatchKMeans`) proved slow or unreliable for the larger `Enron` dataset, two custom algorithms were implemented from scratch to optimize for **Jaccard Distance**. These custom implementations yielded better efficiency and clustering scores.

## Dataset

The data used is the **Bag of Words** dataset from the UCI Machine Learning Repository:
[https://archive.ics.uci.edu/ml/datasets/Bag+of+Words](https://archive.ics.uci.edu/ml/datasets/Bag+of+Words)

The repository contains:
-   `vocab.XYZ.txt`: Vocabulary file for collection XYZ.
-   `docword.XYZ.txt`: Sparse count data (docID, wordID, count).

*Note: The datasets are compressed (`.gz`) in the `bag-of-words-data` directory.*

## Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd Fast_Text_based_Clustering
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The main analysis and code are contained in the Jupyter Notebook:
`Fast_Text_based_Clustering_using_Bag_of_Words_sklearn.ipynb`

To run the notebook:
```bash
jupyter notebook Fast_Text_based_Clustering_using_Bag_of_Words_sklearn.ipynb
```

## Results

-   The project evaluates clustering performance using the **Jaccard Score**.
-   Elbow method graphs are plotted to visualize and determine the optimal number of clusters.
-   Custom algorithms demonstrated improved performance for high-dimensional sparse data like the Enron email dataset.

## Authors

-   Sampad Kumar Kar
-   Shankar Ram Vasudevan
