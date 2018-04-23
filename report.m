%% CAB420 - Machine Learning - Assignment 2
%
% Alex Wilson and Christopher Ayling
%

%% Part A: SVMS and Bayes Classifiers

%% Support Vector Machines
%%
% *1.*

% Clean up
clc
clear
close all

% Load data 
load data_ps3_2.mat
C = 1000;

%Test the SVMs and compare
% Set 1
svm_test(@Klinear, 1, C, set1_train, set1_test)
svm_test(@Kpoly, 2, C, set1_train, set1_test)
svm_test(@Kgaussian, 1, C, set1_train, set1_test)
% Linear SVM had 0.0446 of test examples misclassified (the lowest of the
% three). Therefore, for set 1, linear SVM is best. 

% Set 2
svm_test(@Klinear, 1, C, set2_train, set2_test)
svm_test(@Kpoly, 2, C, set2_train, set2_test)
svm_test(@Kgaussian, 1, C, set2_train, set2_test)
% Polynomial (of degree 2) SVM had 0.011 of test examples misclassifies
% (the lowest of the three). Therefore, for set 2, polynomial SVM is best.

% Set 3
svm_test(@Klinear, 1, C, set3_train, set3_test)
svm_test(@Kpoly, 2, C, set3_train, set3_test)
svm_test(@Kgaussian, 1, C, set3_train, set3_test)
% Gaussian SVM had no misclassifications. Therefore, for set 3, Gaussian
% SVM is best. 

%%
% *2.*

%% Bayes Classifiers
%% 
% *(a)*
%% 
% *(b)*

%% Part B: PCA & Clustering

%% Eigen Faces
%% 
% *(a)*
%% 
% *(b)*
%% 
% *(c)*
%% 
% *(d)*
%% 
% *(e)*

%% Clustering
%% 
% *(a)*
%% 
% *(b)*
%% 
% *(c)*
%% 
% *(d)*


