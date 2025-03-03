function main_py_to_matlab()
    % Import necessary MATLAB toolboxes
    % (MATLAB does not require explicit imports like Python)

    % Define the slow_ft function
    function vecs = slow_ft(x, intercept)
        n = length(x);
        J = floor(n/2);
        vecs = {};
        for i = 0:J
            if i == 0
                vecs{end+1} = cos(2 * pi * i/n * x);
            elseif i/n < 0.5
                vecs{end+1} = [cos(2*pi * i/n * x), sin(2*pi * i/n * x)];
            elseif isequal(i/n, 0.5)
                vecs{end+1} = cos(2*pi*1/n*x);
            end
        end
        vecs = horzcat(vecs{:});
    end

    % Define the hv_folder function
    function idx = hv_folder(arr, h, v)
        idx = {};
        idx_v = {};
        idx_c = {};
        fold_start = v;
        fold_end = length(arr) - v;
        for i = fold_start:fold_end
            idx_v{end+1} = i-v:i+v;
            idx_c{end+1} = setdiff(1:length(arr), idx_v{end});
            hvbottom = max([i-v-h, 0]);
            hvtop = min([i+v+h, length(arr)]);
            idx_c{end} = [idx_c{end}(1:hvbottom), idx_c{end}(hvtop+1:end)];
            idx{end+1} = {idx_v{end}, idx_c{end}};
        end
    end

    % Data generation
    n = 100;
    sigma = 1.5;
    phi = 0;
    theta = 0;
    burnin = 200;
    rng(1);
    x = 1:n;
    y_trend = sin(x/n * 3 * pi) + cos(sqrt(x/n) * 2 * pi);
    u = normrnd(0, sigma, n+burnin, 1);
    epsilon = zeros(n+burnin, 1);
    epsilon(1) = 0;
    for i = 2:n+burnin
        epsilon(i) = phi * epsilon(i-1) + u(i-1) * theta + u(i);
    end
    err = epsilon(burnin+1:end);
    y = y_trend + err;

    % One pass with normal harmonic regression
    k = 7;
    X = slow_ft(x, true);
    X = X(:, 1:k);
    b = mldivide(X'*X, X'*y);
    yhat = X*b;
    plot(yhat);
    hold on;
    plot(y, 'Alpha', 0.3, 'LineWidth', 3);
    plot(y_trend);

    % One pass with 'rotated' regression
    z = 1:n;
    Z = slow_ft(z, true) * (2/sqrt(n));
    zx = Z'*X;
    zy = Z'*y;
    b = mldivide(zx'*zx, zx'*zy);
    plot(zy);
    hold on;
    plot(zx*b);

    % Cross-validation
    out = cell(50, 1);
    for i = 1:50
        k = i;
        X = slow_ft(x, true);
        X = X(:, 1:k);
        folder = hv_folder(y, 0, 0);
        hv = [];
        hv_true = [];
        yhats = [];
        for j = 1:length(folder)
            test = folder{j}{1};
            train = folder{j}{2};
            b = mldivide(X(train, :)'*X(train, :), X(train, :)'*y(train));
            yhat = X(test, :)*b;
            hv = [hv, (y(test) - yhat).^2];
            hv_true = [hv_true, (y_trend(test) - yhat).^2];
            yhats = [yhats, yhat];
        end
        out{i} = struct('k', i, 'hv', mean(hv), 'hv_true', mean(hv_true), 'yhats', yhats);
    end
    out = cell2table(out);
    out = sortrows(out, 'hv');
    disp(['Rotated CV error: ', num2str(mean(out.hv))]);
    disp(['Rotated CV error to trend: ', num2str(mean(out.hv_true))]);
    plot(out.yhats);
    hold on;
    plot(Z'*y, 'Alpha', 0.3, 'LineWidth', 3);
    plot(Z'*y_trend);
end
