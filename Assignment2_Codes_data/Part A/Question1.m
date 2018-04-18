% Clean up
clc
clear
close all
% Load and format data 
load data_ps3_2.mat
u1 = set1_train.X(:,1); v1 = set1_train.X(:,2);
u2 = set2_train.X(:,1); v2 = set2_train.X(:,2);
u3 = set3_train.X(:,1); v3 = set3_train.X(:,2);
% Apply K lin
lin1 = Klinear(u1, v1);
lin2 = Klinear(u2, v2);
lin3 = Klinear(u3, v3);
% Train
C = 1000
svm_train(set1_train, lin1, C);