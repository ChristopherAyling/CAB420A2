function plot2DLinear(obj, X, Y)
% plot2DLinear(obj, X,Y)
%   plot a linear classifier (data and decision boundary) when features X are 2-dim
%   wts are 1x3,  wts(1)+wts(2)*X(1)+wts(3)*X(2)
%
  [n,d] = size(X);
  if (d~=2) ; error('Sorry -- plot2DLogistic only works on 2D data...'); end;

  %%% TODO: Fill in the rest of this function... 
  %find out what classes to plot
  Unique_Class = unique(Y);
  %figure;
  hold on;
  for i = 1:length(Unique_Class)
      %plot each point for class
      if i == 1 ; color_string = 'ro' ; else ; color_string='bo'; end;
      plot(X(Y==Unique_Class(i),1),X(Y==Unique_Class(i),2),color_string)
  end
  %get the weights and plot it on the same figure
  Weights = getWeights(obj);
  %create the function for the boundary 
  %the boundary is defined as when wts(1) + wts(2)x1 + wts(3)x2 = 0 
  %so therefore if we have x2 then it can be worked as a function of x2
  %such as x1 = (wts(1) - wts(3)*x2) / wts(2)
  %to draw a decision boundary we neeed only two points the end point and
  %grab max and min for each feature
  x1 = X(:,1);
  x2 = X(:,2);
  plot_X1 =  linspace(min(x1),max(x1),n);
  plot_X2 =  linspace (min(x2),max(x2),n);   
  %work out the corrsponding y values at those given values
  plot_Y = @(x2) (Weights(1) - (Weights(3).*x2)) / (Weights(2)) ;
  boundary_Y = plot_Y(plot_X2);

  %draw the boundary on the plot
  plot(plot_X1,boundary_Y,'--g')
  legend('class '+string(Unique_Class(1)), 'class '+string(Unique_Class(2)), 'decision boundary');
  title('plot of classes and boundary');
  hold off;  
end
