# Clustering: Finding Wine Classes from Chemical Properties

## Overview

- The main goal of this analysis is to use unsupervised machine learning (clustering) to classify different types of wine according to their chemical and physical properties.
- I also aim to provide a set of quantitative scores to evaluate the correctness of the clustering analysis.
- Finally, the "predicted" clusters are compared with the ground truth classes to further validate the analysis.

## Data

- The dataset was obtained from:
  - Aeberhard,Stefan and Forina,M.. (1991). Wine. UCI Machine Learning Repository. https://doi.org/10.24432/C5PC7J.

- As described in the original repository:
    - "These data are the results of a chemical analysis of wines grown in the same region in Italy but derived from three different cultivars. The analysis determined the quantities of 13 constituents found in each of the three types of wines."

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

- One effective way to visualize how different are the features of the 3 wine classes is by creating a radar plot.

![newplot](https://github.com/solutioncrafter/wine_clustering/assets/126869447/ef580558-7bf3-4259-9dbf-22bbaa5c644a)


### Comparing the Predicted and Original Clusters

- A qualitative and quantitative analysis showed a strong match between the original and the "predicted" clusters, further validating the "blind cluster extraction" results.

## Conclusions

- 3 wine types (3 clusters) were successfully identified from a dataset containing 13 features of wine (chemical properties)

- K-means algorithm was used for clustering

- Various scores were used to identify the optimum number of clusters as 3

- Results were visually evaluated and confirmed by plotting a 2-D representation of the dataset and its color-coded clusters

- The "predicted" or extracted K-means clusters were compared with the ground truth class labels, showing a strong match and demonstrating the method's validity.
