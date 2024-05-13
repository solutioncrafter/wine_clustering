import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.preprocessing import StandardScaler


def compute_metrics(dataframe, cluster_number=range(2, 10)):
    # Initialize lists to store metrics
    wcss = []  # Within-cluster sum of squares
    silhouette_scores = []
    calinski_harabasz_scores = []
    davies_bouldin_scores = []

    # define scaler to perform standarization
    scaler = StandardScaler()

    # standarize data
    dataframe_standardized = scaler.fit_transform(dataframe)

    # Loop through different numbers of clusters
    for i in cluster_number:  # Adjust as needed

        # Fit KMeans clustering
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300,
                        n_init=10, random_state=42)
        kmeans.fit(dataframe_standardized)

        # Extract WCSS (For elbow curve)
        try:
            wcss.append(kmeans.inertia_)

        except Exception:

            print(f"Error, WCSS score omitted for point {i}")
            wcss.append(np.nan)
        # Calculate the davies bouldin score
        try:

            davies_bouldin = davies_bouldin_score(dataframe_standardized,
                                                  kmeans.labels_)
            davies_bouldin_scores.append(davies_bouldin)

        except Exception:
            print(f"Error, davies bouldin score omitted for point {i}")
            davies_bouldin_scores.append(np.nan)

        # Calculate silhouette score
        try:
            silhouette_avg = silhouette_score(dataframe_standardized,
                                              kmeans.labels_)
            silhouette_scores.append(silhouette_avg)

        except Exception:
            print(f"Error, silhouette score omitted for point {i}")
            silhouette_scores.append(np.nan)

        # Calculate calinski harabasz score
        try:
            calinski_harabasz = calinski_harabasz_score(dataframe_standardized,
                                                        kmeans.labels_)
            calinski_harabasz_scores.append(calinski_harabasz)

        except Exception:
            print(f"Error, scalinski harabasz score omitted for point {i}")
            calinski_harabasz_scores.append(np.nan)

    return (wcss,
            silhouette_scores,
            calinski_harabasz_scores,
            davies_bouldin_scores
            )


def plot_kmeans_evaluation_curves(
        dataframe,
        cluster_number_evaluation,
        cluster_number=range(2, 10)):
    # compute  metrics
    (
        wcss,
        silhouette_scores,
        calinski_harabasz_scores,
        davies_bouldin_scores
     ) = compute_metrics(dataframe, cluster_number)

    # Create figure with custom subplot height ratios
    fig = plt.figure(figsize=(10, 8))
    # Set a very small hspace to have minimal white space between plots
    gs = gridspec.GridSpec(4, 1, height_ratios=[3, 2, 2, 2], hspace=0.08)
    # Create each subplot and share the x-axis
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(cluster_number,
             wcss, label='wcss',
             color='red',
             marker='x'
             )
    ax1.set_ylabel('WCSS', color='red')
    ax1.axvline(x=cluster_number_evaluation, color='black', linestyle='--')

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(cluster_number,
             silhouette_scores,
             label='silhouette',
             color='blue',
             marker='o'
             )
    ax2.set_ylabel('Sihouette\nScore', color='blue')
    ax2.axvline(x=cluster_number_evaluation, color='black', linestyle='--')

    plt.setp(ax1.get_xticklabels(), visible=False)  # Hide x-tick labels

    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(cluster_number,
             calinski_harabasz_scores,
             label='calinski harabasz',
             color='green',
             marker='s'
             )
    ax3.set_ylabel('Calinski Harabasz\nScore', color='green')
    ax3.axvline(x=cluster_number_evaluation, color='black', linestyle='--')

    plt.setp(ax2.get_xticklabels(), visible=False)  # Hide x-tick labels

    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.plot(cluster_number,
             davies_bouldin_scores,
             label='davies bouldin',
             color='darkviolet',
             marker='P'
             )
    ax4.set_ylabel('Davies Bouldin\nScore', color='darkviolet')
    ax4.axvline(x=cluster_number_evaluation, color='black', linestyle='--')
    ax4.set_xlabel('Number of Clusters')

    plt.setp(ax3.get_xticklabels(), visible=False)  # Hide x-tick labels
    plt.show()
