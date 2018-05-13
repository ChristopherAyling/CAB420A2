function obj = train(obj, X, Y, varargin)
% obj = train(obj, Xtrain, Ytrain [, option,val, ...])  : train logistic classifier
%     Xtrain = [n x d] training data features (constant feature not included)
%     Ytrain = [n x 1] training data classes 
%     'stepsize', val  => step size for gradient descent [default 1]
%     'stopTol',  val  => tolerance for stopping criterion [0.0]
%     'stopIter', val  => maximum number of iterations through data before stopping [1000]
%     'reg', val       => L2 regularization value [0.0]
%     'init', method   => 0: init to all zeros;  1: init to random weights;  
% Output:
%   obj.wts = [1 x d+1] vector of weights; wts(1) + wts(2)*X(:,1) + wts(3)*X(:,2) + ...


  [n,d] = size(X);            % d = dimension of data; n = number of training data

  % default options:
  plotFlag = false; 
  init     = []; 
  stopIter = 1000;
  stopTol  = -1;
  reg      = 0.0;
  stepsize = 1;

  i=1;                                       % parse through various options
  while (i<=length(varargin)),
    switch(lower(varargin{i}))
    case 'plotflag',      plotFlag = varargin{i+1}; i=i+1;   % plots on (true/false)
    case 'init',      init     = varargin{i+1}; i=i+1;   % init method
    case 'stopiter',  stopIter = varargin{i+1}; i=i+1;   % max # of iterations
    case 'stoptol',   stopTol  = varargin{i+1}; i=i+1;   % stopping tolerance on surrogate loss
    case 'reg',       reg      = varargin{i+1}; i=i+1;   % L2 regularization
    case 'stepsize',  stepsize = varargin{i+1}; i=i+1;   % initial stepsize
    end;
    i=i+1;
  end;

  X1    = [ones(n,1), X];     % make a version of training data with the constant feature

  Yin = Y;                              % save original Y in case needed later
  obj.classes = unique(Yin);
  if (length(obj.classes) ~= 2) error('This logistic classifier requires a binary classification problem.'); end;
  Y(Yin==obj.classes(1)) = 0;
  Y(Yin==obj.classes(2)) = 1;           % convert to classic binary labels (0/1)

  if (~isempty(init) || isempty(obj.wts))   % initialize weights and check for correct size
    obj.wts = randn(1,d+1);
  end;
  if (any( size(obj.wts) ~= [1 d+1]) ) error('Weights are not sized correctly for these data'); end;
  wtsold = 0*obj.wts+inf;

% Training loop (SGD):
iter=1; Jsur=zeros(1,stopIter); J01=zeros(1,stopIter); done=0; sigmoid = @(z) 1./(1+exp(-z)); 
while (~done) 
  step = stepsize/iter;               % update step-size and evaluate current loss values
  %%%% TODO: compute surrogate (neg log likelihood) loss
  
  htheta = (X1)*obj.wts';
  class_response_one = (-Y).*(log(sigmoid(htheta)));
  class_response_two = (1-Y).*log(1-sigmoid(htheta));
  regularizator = reg.*(obj.wts.^2);
  %compute the surrogate loss mean over all elements
  Jsur_Matrix = class_response_one -class_response_two + regularizator ;
  %the mean of the mean
  Jsur(iter) = (1/n)*sum(Jsur_Matrix(:,1));
  disp('the surrogate losss currently is '+string(Jsur(iter)) + ' for iteration '+ string(iter));
  %calculate error rate
  J01(iter) = err(obj,X,Yin);
  
  %%%%% end of compute surrogate (neg log likeihood) loss
 
  if (plotFlag), switch d,            % Plots to help with visualization
    case 1, figure(2);  plot1DLinear(obj,X,Yin);  %  for 1D data we can display the data and the function
    case 2, figure(2); plot2DLinear(obj,X,Yin);  %  for 2D data, just the data and decision boundary
    otherwise, % no plot for higher dimensions... %  higher dimensions visualization is hard
  end; end;
  figure(1); semilogx(1:iter, Jsur(1:iter),'b-',1:iter,J01(1:iter),'g-'); drawnow;
 
  
  %create empty predictions of Y
  Yhat = zeros(1,length(Y));
  for j=1:n,
    % get current weights (paramters of theta)
    wtsold = obj.wts;
    % Compute linear responses and activation for data point j
    z =  (X1(j,:))*wtsold';
    Yhat(j) = sign(wtsold(1)+wtsold(2)*X(j,1)+wtsold(3)*X(j,2));
    sigmoid_j = sigmoid(z);
    %%% change prediction to either 0 or 1 based on neg prediction or
    %%% postivie prediction of sign 
    Yhat(Yhat > 0 ) = 1;
    Yhat(Yhat < 0 ) = 0;
    %%%% Compute gradient:
    %for each wieght work out its gradient with repsect to itself
    k1 = (1-sigmoid_j)*X1(j,:);
    k2 = (-sigmoid_j)*X1(j,:);
    grad = ((-Y(j))*k1)-((1-Y(j))*k2)+(reg*sum(2.*wtsold));
    %update the weight with new increment added
    wtsold = wtsold - step * grad;% take a step down the gradient
    %pass on gradients to obj for the next training point
    obj.wts = wtsold;    
  end
  %update y predictions for surrogate loss function on next loop
  Y = Yhat';
  disp('current weights are ' + string(obj.wts))  
  done = false;
  %%% TODO: Check for stopping conditions
    if (iter == stopIter)
        % train has done the max amount of iterations so stop
        done = 1;%finish the loop
    end
    %if surrgoate loss hasnt changed by more than stoppping tolerance stop
    if iter > 1 
        % check to make sure iteration has occured once at least
        if ( Jsur(iter-1)-Jsur(iter) < stopTol )
            %not enough change has occured to continue , found best weights
            done = 1;%finish the loop
        end
    end
  iter = iter + 1;
  
end;


