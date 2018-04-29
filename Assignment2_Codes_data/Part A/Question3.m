clc;
clear;
close all;

% Hard code in the training data and test data
x1 = [0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1];
x2 = [0; 0; 0; 0; 1; 1; 1; 1; 0; 0; 0; 1; 1; 1; 1; 1];
y = [0; 1; 1; 1; 0; 1; 1; 1; 0; 0; 0; 0; 0; 0; 1; 1];

% Find the probabilities needed for joint Bayes Classifier: 
% Find each P(y)
Py0 = size(y(y==0), 1) / size(y, 1);
Py1 = size(y(y==1), 1) / size(y, 1);

% Estimate P(x|y) using training data
Px1y0 = size(x1(x1==1 & y==0), 1) / size(y(y==0), 1);
Px1y1 = size(x1(x1==1 & y==1), 1) / size(y(y==1), 1);
Px2y0 = size(x2(x2==1 & y==0), 1) / size(y(y==0), 1);
Px2y1 = size(x2(x2==1 & y==1), 1) / size(y(y==1), 1);

% Find each P(y|testx)
testx = [0 1; 1 0; 1 1];

% Find P(0 1|y0)
Ptx1y0 = (1 - Px1y0) * (Px2y0);
Ptx1y1 = (1 - Px1y1) * (Px2y1);

% Find P(0 1) aka P(x)

