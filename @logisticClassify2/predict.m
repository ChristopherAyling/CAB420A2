function Yte = predict(obj,Xte)
% Yhat = predict(obj, X)  : make predictions on test data X
[n,d] = size(Xte);
% (1) make predictions based on the sign of wts(1) + wts(2)*x(:,1) + ...
% (2) convert predictions to saved classes: Yte = obj.classes( [1 or 2] );
wts = getWeights(obj);
Yte = zeros(n,1);
%predict each class based on if sign comes back 0 or 1
predictions = sign(wts(1)+wts(2)*Xte(:,1)+wts(3)*Xte(:,2));
%get the current class labels
Unique_class = getClasses(obj);
%replace class 1 in Unique class with every entry of zero
%replace class 2 in Unique class with every entry of one
Yte(predictions==-1) = Unique_class(1);
Yte(predictions==1) = Unique_class(2);

end

