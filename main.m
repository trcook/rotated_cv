function main()
    %%% Translated from Python to MATLAB
    
    %%% slow_ft function
    function designMatrix = slow_ft(x, intercept)
        if nargin < 2
            intercept = true;
        end
        
        n = length(x);
        J = floor(n/2);
        vecs = {};
        
        for i = 0:J
            if i == 0
                vecs{end+1} = cos(2 * pi * i/n * x(:));
            elseif i/n < 0.5
                vecs{end+1} = [cos(2 * pi * i/n * x(:)), sin(2 * pi * i/n * x(:))];
            elseif abs(i/n - 0.5) < eps
                vecs{end+1} = cos(2 * pi * 1/n * x(:));
            end
        end
        designMatrix = cell2mat(vecs);
    end

    %%% hv_folder function
    function folds = hv_folder(arr, h, v)
        idx = 1:length(arr);
        fold_start = v + 1;
        fold_end = length(arr) - v;
        folds = {};
        
        for i = fold_start:fold_end
            idx_v = idx(i-v:i+v);
            hvbottom = max(i-v-h, 1);
            hvtop = min(i+v+h, length(arr));
            idx_c = [idx(1:hvbottom-1), idx(hvtop+1:end)];
            folds{end+1} = {idx_v, idx_c};
        end
    end

    %%% Data generation
    n = 100;
    sigma = 1.5;
    phi = 0;
    theta = 0;
    burnin = 200;
    rng(1);  % Set random seed
    
    x = (1:n)';
    y_trend = sin(x/n * 3 * pi) + cos(sqrt(x/n) * 2 * pi);
    
    u = sigma * randn(n + burnin, 1);
    epsilon = zeros(size(u));
    epsilon(1) = 0;
    
    for i = 2:length(u)
        epsilon(i) = phi * epsilon(i-1) + u(i-1) * theta + u(i);
    end
    
    err = epsilon(burnin+1:end);
    y = y_trend + err;

    %%% Harmonic regression
    k = 7;
    X = slow_ft(x);
    X = X(:,1:k);
    
    b = X \ y;
    yhat = X * b;
    
    figure;
    hold on;
    plot(yhat, 'LineWidth', 1.5);
    plot(y, 'Color', [0.7 0.7 0.7], 'LineWidth', 3);
    plot(y_trend, 'k--');
    legend('Predicted', 'Observed', 'True Trend');
    hold off;

    %%% Cross-validation analysis
    out = struct('k', {}, 'hv', {}, 'hv_true', {});
    
    for i = 0:49
        k = i;
        X = slow_ft(x);
        X = X(:,1:k);
        
        folder = hv_folder(y, 0, 0);
        hv = [];
        yhats = [];
        hv_true = [];
        
        for fold = folder
            test = fold{1}{1};
            train = fold{1}{2};
            
            X_train = X(train, :);
            y_train = y(train);
            b = X_train \ y_train;
            
            yhat = X(test, :) * b;
            hv = [hv; (y(test) - yhat).^2];
            hv_true = [hv_true; (y_trend(test) - yhat).^2];
        end
        
        out(end+1).k = k;
        out(end).hv = mean(hv);
        out(end).hv_true = mean(hv_true);
    end
    
    % Display results
    [~, idx] = sort([out.hv]);
    disp('Best model by CV:');
    disp(out(idx(1)));
end
