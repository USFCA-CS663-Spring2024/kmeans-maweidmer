from cluster import cluster
import random
import math


class KMeans(cluster):
    """An implementation of the kmeans algorithm for clustering.
    Attributes
    ---------
    k : int
        the number of target clusters
    max_iterations : int
        the maximum number of convergence attempts
    centroids : list
        the cluster centroid values
    labels : list
        the cluster hypotheses of all the data instances

    Methods
    -------
    fit(X)
        Performs kmeans on the given data X.
    place_centroids(X)
        Randomly picks k centroids.
    assign_centroid(instance)
        Assigns a given instance to the closest centroid.
    calculate_distance(centroid, instance)
        Calculates the euclidean distance between a given centroid and data instance.
    update_centroids(X)
        Calculates new centroids by taking the average of all points assigned to each cluster.
    """

    def __init__(self, k=5, max_iterations=100):
        """
        :param k: The target number of cluster centroids, default: 5
        :param max_iterations: The maximum number of attempts to converge, default: 100
        """
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = []
        self.labels = []

    def fit(self, X):
        """Performs kmeans on the given data X.

        :param X: the given instance data
        :return: a list of the cluster hypotheses, and a list of the cluster centroid values
        """

        self.__place_centroids(X)  # initial centroid placement
        self.labels = [self.__assign_centroid(instance) for instance in X]  # assign points to initial centroids

        # loop until max iterations
        for i in range(self.max_iterations):
            # calculate new centroids
            new_centroids = self.__update_centroids(X)
            # check to see if any centroids have changed
            cent_diff = [new_cent == old_cent for (new_cent, old_cent) in zip(new_centroids, self.centroids)]
            self.centroids = new_centroids

            # assign points to updated centroids
            new_labels = [self.__assign_centroid(instance) for instance in X]
            # check to see if any points have switched clusters
            labels_diff = [new_label == old_label for (new_label, old_label) in zip(new_labels, self.labels)]
            self.labels = new_labels

            # if either the centroids or labels have converged (no change since last iteration), break
            # out of the loop
            if cent_diff.count(False) == 0:
                print('Centroids converged in ' + str(i) + ' iterations.')
                break
            if labels_diff.count(False) == 0:
                print('Labels converged in ' + str(i) + ' iterations.')
                break

        return self.labels, self.centroids

    def __place_centroids(self, X):
        """Randomly picks k centroids.

        :param X: the dataset to pick centroid cluster values for
        """

        # Randomly picks a k-sized sample of points from the dataset to act as starting centroids.
        # This was going to be starting solution until I could implement something better (like kmeans++)
        # but unfortunately I ran out of time.  This is addressed in the jupyter notebook.
        self.centroids = [list(sample) for sample in random.sample(list(X), k=self.k)]

    def __assign_centroid(self, instance):
        """Assigns a given instance to the closest centroid.

        :param instance: the data instance that is being assigned to a cluster
        :return: the index of the centroid the instance has been assigned to
        """

        # calculate the distance between the instance and every centroid
        distances = [self.__calculate_distance(centroid, instance) for centroid in self.centroids]
        # return the index of the closest (minimum distance) centroid
        return distances.index(min(distances))

    def __calculate_distance(self, centroid, instance):
        """Calculates the euclidean distance between a given centroid and data instance.

        :param centroid: the centroid to find the distance to
        :param instance: the datapoint to find the distance from
        :return: the euclidean distance between the centroid and instance
        """

        diff = [(inst_v - cent_v) ** 2 for (cent_v, inst_v) in zip(centroid, instance)]
        return math.sqrt(sum(diff))

    def __update_centroids(self, X):
        """Calculates new centroids by taking the average of all points assigned to each cluster.

        :param X: the data instances being clustered
        :return: the new centroids
        """

        # initialise the list to hold new clusters
        new_centroids = [[]] * self.k

        # for each centroid
        for i in range(self.k):
            # find all data instances assigned to the centroid
            instances = [instance for (instance, label) in zip(X, self.labels) if label == i]
            new_centroid = [0] * len(instances[0])
            # sum the feature values of all the instances
            for inst_i in range(len(instances)):
                for feat_i in range(len(instances[inst_i])):
                    new_centroid[feat_i] += instances[inst_i][feat_i]
            # calculate the average feature value for all the instances to get the new centroid
            new_centroid = [feat_val / len(instances) for feat_val in new_centroid]
            new_centroids[i] = new_centroid
        return new_centroids
