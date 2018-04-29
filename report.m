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
clc;
clear;
close all;
load data_ps3_2.mat
C = 1000;

set(gcf,'Visible', 'off');
lin_err = svm_test2(@Klinear, 1, C, set4_train, set4_test);
poly_err = svm_test2(@Kpoly, 2, C, set4_train, set4_test);
gauss_err = svm_test2(@Kgaussian, 1.5, C, set4_train, set4_test);

% compare to log regress classifier

% print results

%% Bayes Classifiers
%% 
% *(a)*
%% 
% *(b)*

%% Part B: PCA & Clustering

%% Eigen Faces
clear;
X = load('data/faces.txt')/255;
%%
img = reshape(X(1,:), [24, 24]);
imagesc(img);
axis square;
colormap gray;

%% 
% *(a)*
[m, n] = size(X);

mu = mean(X);
X0 = bsxfun(@minus, X, mu);
sigma = std(X0);
X0 = bsxfun(@rdivide, X0, sigma);

[U, S, V] = svd(X0);

W=U*S;

%% 
% *(b)*
ks = [1:10];
for i=1:length(ks)
    X0_hat = W(:, 1:ks(i))*V(:, 1:ks(i))';
    mse(i) = mean(mean(X0-X0_hat).^2);
end

figure();
hold on;
plot(ks, mse);
title('???'); %TODO Idk what to call this yet
xlabel('K');
ylabel('MSE');
hold off;


%% 
% *(c)*
what = {};
what2 = {}
for j=1:10
    alpha = 2*median(abs(W(:, j)));
    what{j} = mu + alpha*(V(:, j)');
    what2{j} = mu - alpha*(V(:, j)');
end

img = [reshape(what{1}, [24, 24])];
figure(8);
imagesc(img);
axis square;
colormap gray;

%% 
% *(d)*
idx = [1:20];
figure; hold on; axis ij; colormap(gray);
range = max(W(idx, 1:2)) - min(W(idx, 1:2));
scale = [200 200]./range;
for i=idx, imagesc(W(i,1)*scale(1),W(i,2)*scale(2), reshape(X(i,:), 24, 24)); end;

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

clear all


