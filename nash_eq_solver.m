%% Subdivide 1D trait space
Xs = linspace(0,1,1000);

%% Set parameters
f_mode = 2;
fit_mode = 'cubic';
mins = NaN(size(Xs));

%% define performance functions
if f_mode == 1
     %f = @(x,y) Moran_performance_function_interp(x,y,'log_odds_moran_pd');
     f = @(x,y) performance_pd_moran_24_individuals(x,y,fit_mode);
elseif f_mode == 2
    f = @(x,y) performance_stag_moran_24_individuals(x,y,fit_mode);
elseif f_mode == 3
    f = @(x,y) Moran_performance_function_interp(x,y,'log_odds_moran_chicken_10000_midway.mat');
end

%% Generate sampled optima
for i = 1:length(Xs)
    %% define cost function
    x = Xs(i);
    cost_f = @(y) f(x,y);
    
    %% optimize
    mins(i) = fminbnd(cost_f,0,1);
end

%% find equilibrium
discrepancy = mins - Xs;
discrepancy = discrepancy*sign(discrepancy(1));
i = find(discrepancy > 0,1,'last');

equilibrium = Xs(i) - ((Xs(i+1)-Xs(i))/(discrepancy(i+1)-discrepancy(i))*discrepancy(i));


%% Plot results
figure(1)
clf
hold on
plot(Xs,mins);
plot(mins,Xs);
plot([0,1],[0,1]);
scatter(equilibrium,equilibrium,20,'k','filled');
axis square
grid on
