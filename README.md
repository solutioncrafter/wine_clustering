# Clustering: Finding Wine Classes from Chemical Properties

## Overview

-The main goal of this analysis is to use unsupervised machine learning (clustering) to classify different types of wine according to their chemical and physical properties.
- I also aim to provide a set of quantitative scores to evaluate the correctness of the clustering analysis.
- Finally, the "predicted" clusters are compared with the ground truth classes to further validate the analysis.

## Data

## Methods 

- K-means algorithm is used to perform the clustering.
- A set of scores is defined to evaluate the best number of clusters.
- Dimensionality reduction algorithms (as PCA) are used for visualization of the clusters.
- A confusion matrix and quantitative scores are defined to evaluate how well the K-means algorithm reproduced the original (real) clusters.
  
## Results

This is a representative summary of the most relevant results. The jupyter notebook presents a complete set of results. 

### Evaluating the Optimum Number of Clusters (K-means)

- A set of evaluation scores is computed and plotted for various cluster numbers, and the K-means algorithm is used to discover the optimum value.

  - Within-cluster sum of squares (WCSS) or "Elbow Curve": Where the optimum number of clusters is at the inflection point.
  - Silhouette Score: We look for the maximum value.
  - Calinski Harabasz Score: We look for the maximum value.
  - Davies Bouldin Score: We look for the minimum value.
    
- From the evaluation curves, 3 clusters appear to be the optimal number.

![image](https://github.com/solutioncrafter/wine_clustering/assets/126869447/30c5e67e-b4b9-4253-be4e-bbc52da4f39c)

### K-Means Clustering Results

- The dimensionality of the dataset was reduced from 13 to 2 dimensions, so that the clustering results can be better visualized and inspected.

- One of the methods used to reduce the dimensionality was PCA. Here, I plot the projection to 2 dimensions with the predicted clusters represented by different colors.

![image](https://github.com/solutioncrafter/wine_clustering/assets/126869447/0a1ac9b5-0b26-43cf-bd50-018b31eca49b)

### Comparing the Predicted and Original Clusters

- A qualitative and quantitative analysis showed a strong match between the original and the "predicted" clusters, further validating the "blind cluster extraction" results.

## Conclusions

- 3 wine types (3 clusters) were successfully identified from a dataset containing 13 features of wine (chemical properties)

- K-means algorithm was used for clustering

- Various scores were used to identify the optimum number of clusters as 3

- Results were visually evaluated and confirmed by plotting a 2-D representation of the dataset and its color-coded clusters

- The "predicted" or extracted K-means clusters were compared with the ground truth class labels, showing a strong match and demonstrating the method's validity.
