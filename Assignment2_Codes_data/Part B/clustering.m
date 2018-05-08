%% *a*
% Clean up
clc
close all
clear
% Load iris data, first two features, ignore class 
load('iris.txt');
iris = iris(:,1:2);
% Plot the data to visualise clustering 
scatter(iris(:,1), iris(:,2), 15, 'mo');
hold on;
title('Iris dataset');

%% *b*
% For k=5, initialise random centroids
K1=5;
initial_centroids1 = [
  4.68	  4.07;	
  6.17	  3.12;	
  6.52	  2.71;	
  7.39	  2.36;	
  6.22	  2.38	
];

% For k=20, initialise random centroids
K2=20;
initial_centroids2 = [ 
  4.41	  3.23;  5.37	  3.24;  5.69	  2.22;  5.69	  3.08;
  4.84	  2.86;  5.83	  2.91;  5.28	  2.36;  6.84	  2.70;
  5.47	  4.02;  5.21	  3.17;  5.94	  2.33;  4.97	  2.08;
  6.09	  2.83;  4.47	  3.41;  7.42	  3.45;  5.06	  3.73;
  7.07	  3.50;  4.94	  3.55;  7.03	  2.87;  7.62	  2.90   
];

centroids1 = initial_centroids1;
centroids2 = initial_centroids2;

% Perform k-means on the data with 10 iterations, k=5 and k=20
for i = 1:10
    idx1 = findClosestCentroids(iris, centroids1);
    idx2 = findClosestCentroids(iris, centroids2);
    centroids1 = computeCentroids(iris, idx1, K1); 
    centroids2 = computeCentroids(iris, idx2, K2);   
end 

% Plot final centroids of k=5
figure; hold on; 
plotDataPoints(iris, idx1, K1);
plot(centroids1(:,1), centroids1(:,2), 'x', ...
    'MarkerEdgeColor','k', ...
    'MarkerSize', 10, 'LineWidth', 3);
title('K-means clustering where k=5');

% Plot final centroids of k=20
figure; hold on; 
plotDataPoints(iris, idx2, K2);
plot(centroids2(:,1), centroids2(:,2), 'x', ...
    'MarkerEdgeColor','k', ...
    'MarkerSize', 10, 'LineWidth', 3);
title('K-means clustering where k=20');

%% *c*
% Compute the single, and complete linkage
sLink = linkage(iris, 'single');
cLink = linkage(iris, 'complete');

% Plot single linkage, 5 clusters
clust = cluster(sLink, 'maxclust', 5);
figure;
scatter(iris(:,1), iris(:,2), 40, clust, 'filled');
title('Single linkage agglomerative clustering, 5 clusters');

% Plot single linkage, 20 clusters
clust = cluster(sLink, 'maxclust', 20);
figure;
scatter(iris(:,1), iris(:,2), 40, clust, 'filled');
title('Single linkage agglomerative clustering, 20 clusters');

% Plot complete linkage, 5 clusters
clust = cluster(cLink, 'maxclust', 5);
figure;
scatter(iris(:,1), iris(:,2), 40, clust, 'filled');
title('Complete linkage agglomerative clustering, 5 clusters');

% Plot complete linkage, 20 clusters
clust = cluster(cLink, 'maxclust', 20);
figure;
scatter(iris(:,1), iris(:,2), 40, clust, 'filled');
title('Complete linkage agglomerative clustering, 20 clusters');

