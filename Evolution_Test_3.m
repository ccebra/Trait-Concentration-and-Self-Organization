clear

%% clear figures
for j = 1:21
    figure(j)
    clf
end

%% settings/parameters
num_competitors = 50;% number of total competitors
num_moran = 24;% number of entrants in Moran model

num_traits = 1; % we want to generalize to more dimensions
num_frequencies = 2;% In performance function, number of trigonometric frequencies
trig_amplitude = 1;% Amplitude of trigonometric components in performance function
linear_amplitude = 1;% Amplitude of linear components in performance function
f_mode = 3;% Which of the performance functions is to be used

genetic_drift = 0.5*10^(-2);% Amount that the traits can change with reproduction
num_epochs = 50;% Number of evolutionary steps in each trial

num_experiments = 100;% Number of total trials
games_per_competitor = 100;% Number of games each competitor plays to determine fitness

num_conv_rate_boundary = 10;%Number of times we run convergence rate analysis for boundary cases
num_conv_rate_interior = 10;%Number of times we run convergence rate analysis for interior cases
% num_conv_rate = 10^6;% Number of competitors in convergence rate analysis
std_bounds = [-2.5,-1]; % bounds on range of stds to test for convergence (log base 10)
num_stds = 40;% Number of standard deviations that we run convergence rate analysis over
network_sample_size = 100; % size of network to sample per trial in convergence test
epoch_bounds = [10,10^3]; % max and min number of trials to run in convergence test
tol = 10^(-4); % desired tolerance on estimated rho in convergence test

payout = cell(2,2);
payout{1,1} = [3,3];
payout{1,2} = [0,2];
payout{2,1} = [2,0];
payout{2,2} = [1,1];

tic;

%% Save parameters to results struct
results.parameters.num_competitors = num_competitors;
results.parameters.num_traits = num_traits;

results.parameters.payout_aa = payout{1,1};
results.parameters.payout_ab = payout{1,2};
results.parameters.payout_ba = payout{2,1};
results.parameters.payout_bb = payout{2,2};

results.parameters.num_frequencies = num_frequencies;
results.parameters.trig_amplitude = trig_amplitude;
results.parameters.linear_amplitude = linear_amplitude;
results.parameters.f_mode = f_mode;

results.parameters.genetic_drift = genetic_drift;
results.parameters.num_epochs = num_epochs;
results.parameters.num_experiments = num_experiments;

results.parameters.games_per_competitor = games_per_competitor;
results.parameters.num_conv_rate_boundary = num_conv_rate_boundary;
results.parameters.num_conv_rate_interior = num_conv_rate_interior;
% results.parameters.num_conv_rate = num_conv_rate;
results.parameters.std_bounds = std_bounds;
results.parameters.network_sample_size = network_sample_size;
results.epoch_bounds = [10,10^3];
results.tol = tol;

%% generate edge to endpoint mapping
edge_to_endpoints = NaN(num_competitors*(num_competitors - 1)/2,2);
k = 0;
for i = 1:num_competitors-1
    for j = i+1:num_competitors
        k = k+1;
        edge_to_endpoints(k,:) = [i,j];
    end
end


%% preallocate (generate data array that tracks data over multiple experiments)
experiment_array = zeros(num_experiments,12); % will store experimental results at end of evolution
step_by_step = NaN(num_epochs,6,num_experiments); % will store experimental results over each step

grad = NaN(num_traits,num_experiments); % stores gradient in performance at end of evolution
Hessian.xx = cell(num_experiments,1); % stores on diagonal block of the Hessian at end of evolution
Hessian.xy = cell(num_experiments,1); % stores off diagonal block of the Hessian at end of evolution

epsilon_prediction = NaN(num_experiments,1); % stores predicted value of epsilon at end of each experiment
rhos.prediction = NaN(num_experiments,1); % stores predicted value of rho (correlation coefficient) at end of each experiment
rhos.empirical = NaN(num_experiments,1); % stores the empirical value of rho at the end of each experiment
rel_intransitivity_prediction = NaN(num_experiments,1); % stores the predicted relative intransitivity at the end of each experiment
rhos.convergence.boundary.analytic = NaN(num_experiments,num_stds); % stores predicted correlation coefficient as a function of standard deviation around final centroid
rhos.convergence.interior.analytic = NaN(num_experiments,num_stds); % stores predicted correlation coefficient as a function of standard deviation around final centroid
rhos.convergence.boundary.mean = NaN(num_experiments,num_stds); % stores empirical correlation coefficient as a function of standard deviation around final centroid
rhos.convergence.interior.mean = NaN(num_experiments,num_stds); % stores empirical correlation coefficient as a function of standard deviation around final centroid
rhos.convergence.boundary.std = NaN(num_experiments,num_stds); % stores std in empirical correlation coefficient as a function of standard deviation around final centroid
rhos.convergence.interior.std = NaN(num_experiments,num_stds); % stores std in empirical correlation coefficient as a function of standard deviation around final centroid
boundary_correlation_coefficient = NaN(num_experiments,num_stds); % stores empirical correlation coefficient as a function of standard deviation around final centroid if on boundary
interior_correlation_coefficient = NaN(num_experiments,num_stds); % stores empirical correlation coefficient as a function of standard deviation around final centroid if in interior


%% loop over experiments
for experiment = 1:num_experiments
    
    %% randomly generate traits
    %[x,y,z] = rand_pick_sphere(num_competitors,0,1); %Call rand_pick_sphere function
    competitor_traits = (rand(num_competitors,num_traits));
    
    %% generate data array that tracks transitivity/intransitivity over time (epochs)
    epoch_array = zeros(num_epochs,6);
    cluster_centroid = zeros(1,num_traits);
    evolution_over = 0;%Dummy variable to check if we can stop the evolution process
    on_boundary = 0;%Dummy variable to state if the final location is on the boundary
    final_epoch = num_epochs;%final_epoch will output the right row of the epoch_array to experiment_array, this starts at the maximal epoch and changes if evolution_over changes
    epoch = 1;
    
    %% loop over epochs
    while epoch < final_epoch && evolution_over == 0
        disp(epoch);
        
        %% Sample event outcomes
        n_events = games_per_competitor*num_competitors;
        
        competitors = NaN([n_events,2]);
        
        Z = rand([n_events,1]); % this is all the random numbers we need
        p_win = NaN([n_events,1]);
        Outcomes = zeros([n_events,1]);
        
        k = 0;
        for i = 1:num_competitors
            for events = 1:games_per_competitor
                %% count games
                k = k+1;
                
                %% pick opponent
                stop = 0;
                while stop == 0
                    j = randperm(num_competitors,1);
                    if j ~= i
                        stop = 1;
                    end
                end
                
                %% store competitors
                competitors(k,1) = i;
                competitors(k,2) = j;
                
                %% get win probability
                %Don't call Moran if identical competitors are playing one
                %another, instead flip a coin
                p_win(k) = moran(payout,competitor_traits(i),competitor_traits(j),num_moran); % p_win = logistic(performance)
                if p_win(k) == competitor_traits(i)
                    Outcomes(k) = 1;
                end
            end
            
        end
        
        %% 
        
        %% Compute win frequencies (only left competitor)
        win_freq = sum(reshape(Outcomes, [games_per_competitor,num_competitors])',2)/games_per_competitor;
        indices = (1:num_competitors)';
        winfreq_with_indices = [indices,win_freq];
        
        %% performing the HHD for a complete graph ***only works for complete graphs
        ratings = (1/num_competitors)*sum(competition,2);
        
        
        %% decompose log odds
        F_t = ratings - ratings';
        F_c = competition - F_t;
        
        %% sizes of components
        Transitivity = norm(F_t,'fro')/sqrt(2);
        Intransitivity = norm(F_c,'fro')/sqrt(2);
        
        
        %% covariance calculations
        cov_at_step = trace(cov(competitor_traits));
        
        %% choose highest-fitness competitors
        winfreq_with_indices = sortrows(winfreq_with_indices,2, 'descend');
        highest_fitness = winfreq_with_indices(1:num_competitors/10,:);
        
        %% analyze for clusters      
        AIC = zeros(1,round(num_competitors/10));
        GMModels = cell(1,round(num_competitors/10));
        options = statset('MaxIter',500);
        for k = 1:(round(num_competitors/10))%Getting an ill-conditioned covariance error
            GMModels{k} = fitgmdist(competitor_traits,k,'Options',options,'CovarianceType','full', 'RegularizationValue',genetic_drift);
            AIC(k)= GMModels{k}.AIC;
        end
        
        [minAIC,numComponents] = min(AIC);
        n_classes = numComponents;
        
        %% cluster centroid (I moved the analyze for clusters function earlier to put this)
        %if cov_at_step < 4*num_traits*genetic_drift^2 %covariance is within twice range we'd expect if it was a single point that expanded outwards
        new_cluster_centroid = mean(competitor_traits);
        if (norm(cluster_centroid - new_cluster_centroid)) < genetic_drift/5%Cluster centroid has stopped moving
            evolution_over = 1;
            final_epoch = epoch;
        end
        cluster_centroid = new_cluster_centroid;
        %end
        
        %% reproduction process
        new_competitor_traits = NaN(num_competitors,num_traits);
        parent_traits = competitor_traits(highest_fitness(:,1),:);
        for i=0:(num_competitors/10-1)
            for j=1:10
                drift_vector = genetic_drift*(randn(1,num_traits));
                child_vector = competitor_traits(highest_fitness(i+1,1),:) + drift_vector;
                %if norm(child_vector) > 1 (code for sphere only)
                %    child_vector = 1/norm(child_vector)*child_vector;
                %end
                array_max = max(abs(child_vector));
                array_min = min(abs(child_vector));
                if array_max > 1
                    child_vector = 1/array_max*child_vector;
                end
                if array_min < 0
                    child_vector = child_vector - array_min;
                end
                new_competitor_traits(j+(10*i),:) = child_vector;
            end
        end
        competitor_traits = new_competitor_traits;
        
        %% display
         if num_traits == 1
             figure(1)
             clf;
             hold on
             for class_index = 1:n_classes
                 scatter3(competitor_traits(classes == class_index,1),competitor_traits(classes == class_index,2),competitor_traits(classes == class_index,3),20,'fill')
             end
             grid on
             axis square
             xlim([-1,1])
             ylim([-1,1])
             zlim([-1,1])
             view([1,1,1])
             drawnow
         end
        
        
        %% add to arrays
        for epoch_fill = epoch:num_epochs%Fixed to now fill subsequent epochs up to max
            epoch_array(epoch_fill,1) = Transitivity;
            epoch_array(epoch_fill,2) = Intransitivity;
            epoch_array(epoch_fill,3) = Intransitivity/sqrt(Transitivity^2 + Intransitivity^2);
            epoch_array(epoch_fill,4) = cov_at_step;
            epoch_array(epoch_fill,5) = n_classes;
            epoch_array(epoch_fill,6) = 1;
        end
        epoch = epoch + 1;
        
        
    end
    step_by_step(:,:,experiment) = epoch_array;
    
    %% Flag if final centroid is on the boundary
    final_gen_traits = max(parent_traits,[],2);
    max_coordinate = quantile(final_gen_traits,0.25,1);
    if max_coordinate > 1-1.96*genetic_drift
        on_boundary = 1;
    end
    
    %% Compute gradient and Hessian at centroid
%     if f_mode == 1
%         [g,H] = Hessian_for_example_performance_4(cluster_centroid,trait_pairs,alpha,linear_amplitude,phase);
%     elseif f_mode == 3
%         [g,H] = Hessian_for_example_performance_6(cluster_centroid,trait_pairs,alpha,linear_amplitude,phase);
%     end
%     
%     grad(:,experiment) = g;
%     Hessian.xx{experiment} = H.xx;
%     Hessian.xy{experiment} = H.xy;
%     hessian_xx_norm = norm(H.xx,'fro');
%     hessian_xy_norm = norm(H.xy,'fro');
%     
%     
%     %% predict relative size of intransitivity, use to compare to empirical result for the final sample
%     Sigma = cov(competitor_traits);
%     final_std(experiment) = sqrt(trace(Sigma)/num_traits);
%     epsilon_prediction(experiment) = trace(-H.xy*Sigma*H.xy*Sigma)/(2*g'*Sigma*g + trace(H.xx*Sigma*H.xx*Sigma));
%     rhos.prediction(experiment) = 1/(2*(1 + epsilon_prediction(experiment)));
%     
%     E = num_competitors*(num_competitors - 1)/2;
%     L = E - (num_competitors - 1);
%     rel_intransitivity_prediction(experiment) = sqrt((1 - 2*rhos.prediction(experiment))*(L/E));
%     
%     
%     
%     %% empirical correlation
%     rhos.empirical(experiment) = (1 - (E/L)*(Intransitivity^2/(Transitivity^2 + Intransitivity^2)))/2;
%     
%        %% predict rho as a function of standard deviation 
%     epsilon = @(std_traits) norm(H.xy,'fro')^2./(2*(norm(g)./std_traits).^2 + norm(H.xx,'fro')^2);
%     rho = @(std_traits) 1./(2*(1 + epsilon(std_traits))); %we will want to compare this to the results from the empirical approach defined below
%     
%     
%     %% Calculate convergence rate given deviations of competitors from final cluster centroid (boundary)
%     if  num_conv_rate_boundary > 0 && on_boundary == 1 
%         boundary_std_convergence_list = 10.^(linspace(std_bounds(2),std_bounds(1),num_stds));
%         for repopulate_step = 1:length(boundary_std_convergence_list)
%             %% compute theoretical rho and epsilon
%             epsilons(experiment,repopulate_step) = epsilon(boundary_std_convergence_list(repopulate_step));
%             rhos.convergence.boundary.analytic(experiment,repopulate_step) = rho(boundary_std_convergence_list(repopulate_step)); % compare to rhos.mean from below
%             
%             if num_conv_rate_boundary > 0 %run 10 total convergence rate tests empirically
%                 %% estimate rho empirically
%                 [rhos.convergence.boundary.mean(experiment,repopulate_step),rhos.convergence.boundary.std(experiment,repopulate_step)] = estimate_rho_Gauss(f,cluster_centroid,boundary_std_convergence_list(repopulate_step)^2,tol,epoch_bounds,network_sample_size);
%                 boundary_correlation_coefficient(experiment,repopulate_step) = rhos.convergence.boundary.mean(experiment,repopulate_step);
%             end
%             
%         end
%         num_conv_rate_boundary = num_conv_rate_boundary-1;
%     end
%     
%     %% Calculate convergence rate given deviations of competitors from final cluster centroid (interior)
%     if num_conv_rate_interior > 0 && on_boundary == 0 %run 10 total convergence rate tests
%         interior_std_convergence_list = 10.^(linspace(std_bounds(2),std_bounds(1),num_stds));
%         for repopulate_step = 1:length(interior_std_convergence_list)
%             %% compute theoretical rho and epsilon
%             epsilons(experiment,repopulate_step) = epsilon(interior_std_convergence_list(repopulate_step));
%             rhos.convergence.interior.analytic(experiment,repopulate_step) = rho(interior_std_convergence_list(repopulate_step)); % compare to rhos.mean from below
%             
%             %% estimate rho empirically
%             [rhos.convergence.interior.mean(experiment,repopulate_step),rhos.convergence.interior.std(experiment,repopulate_step)] = estimate_rho_Gauss(f,cluster_centroid,interior_std_convergence_list(repopulate_step)^2,tol,epoch_bounds,network_sample_size);
%             interior_correlation_coefficient(experiment,repopulate_step) = rhos.convergence.interior.mean(experiment,repopulate_step);
%         end
%         num_conv_rate_interior = num_conv_rate_interior - 1;
%     end
%     
    %% Add parameters from this to experiments array and structs
    experiment_array(experiment,1) = epoch_array(1,3);
    experiment_array(experiment,2) = epoch_array(1,4);
    experiment_array(experiment,3) = epoch_array(2,5);
    experiment_array(experiment,4) = epoch_array(final_epoch,3);
    experiment_array(experiment,5) = epoch_array(final_epoch,4);
    experiment_array(experiment,6) = epoch_array(final_epoch,5);
    experiment_array(experiment,7) = final_epoch;
    experiment_array(experiment,8) = max_coordinate;
    experiment_array(experiment,9) = on_boundary;
    experiment_array(experiment,11) = hessian_xx_norm;
    experiment_array(experiment,12) = hessian_xy_norm;
    
    %% print
    fprintf('\n Trial %d of %d complete \n',experiment,num_experiments)
end

%% Add parameters to structs
grad_norm = vecnorm(grad);
grad_norm = grad_norm';

results.stepbysteparray = step_by_step;
results.intransitivity.initial = experiment_array(:,1);
results.covariance.initial = experiment_array(:,2);
results.clusters.initial = experiment_array(:,3);
results.intransitivity.final = experiment_array(:,4);
results.covariance.final = experiment_array(:,5);
results.clusters.final = experiment_array(:,6);
results.num.steps = experiment_array(:,7);%(Struct)
results.maxcoordinate = experiment_array(:,8);
results.norms.gradient = grad_norm;
results.norms.xxhessian = experiment_array(:,11);
results.norms.xyhessian = experiment_array(:,12);

results.analysis.grad = grad;
results.analysis.Hessian = Hessian;
results.analysis.epsilon = epsilon_prediction;
results.analysis.rho = rhos.prediction;
results.analysis.rho_empirical = rhos.empirical;
results.analysis.rel_intransitivity = rel_intransitivity_prediction; 
results.analysis.epsilon_function = epsilon;


results.convergence_test.boundary.stds = boundary_std_convergence_list;
results.convergence_test.boundary.rhos = boundary_correlation_coefficient;
results.convergence_test.boundary.rho_mean = nanmean(boundary_correlation_coefficient);
results.convergence_test.boundary.rho_predictions = rhos.convergence.boundary.analytic;
results.convergence_test.interior.stds = interior_std_convergence_list;
results.convergence_test.interior.rhos = interior_correlation_coefficient;
results.convergence_test.interior.rho_mean = nanmean(interior_correlation_coefficient);
results.convergence_test.interior.rho_predictions = rhos.convergence.interior.analytic;

%% Save step-by-step results
step_by_step_array = NaN(num_epochs,6);
step_by_step_array = nansum(step_by_step,3);
step_by_step_array(:,(1:5)) = step_by_step_array(:,(1:5))./step_by_step_array(:,6);

step_by_step_std = NaN(num_epochs,6);
step_by_step_std = std(step_by_step,[],3,'omitnan');

%% Quartiles for step-by-step
step_by_step_first_quantile = quantile(step_by_step, 0.25, 3);
step_by_step_second_quantile = quantile(step_by_step, 0.75, 3);

%% Save step-by-step array to structs
results.bystep.intransitivity.mean = step_by_step_array(:,3);
results.bystep.covariance.mean = step_by_step_array(:,4);
results.bystep.clusters.mean = step_by_step_array(:,5);
results.bystep.remaining.mean = step_by_step_array(:,6);
results.bystep.intransitivity.std = step_by_step_std(:,3);
results.bystep.covariance.std = step_by_step_std(:,4);
results.bystep.clusters.std = step_by_step_std(:,5);
results.bystep.remaining.std = step_by_step_std(:,6);
results.bystep.intransitivity.firstquant = step_by_step_first_quantile(:,3);
results.bystep.covariance.firstquant = step_by_step_first_quantile(:,4);
results.bystep.clusters.firstquant = step_by_step_first_quantile(:,5);
results.bystep.remaining.firstquant = step_by_step_first_quantile(:,6);
results.bystep.intransitivity.secondquant = step_by_step_second_quantile(:,3);
results.bystep.covariance.secondquant = step_by_step_second_quantile(:,4);
results.bystep.clusters.secondquant = step_by_step_second_quantile(:,5);
results.bystep.remaining.secondquant = step_by_step_second_quantile(:,6);

%% Find overall information about our experiment's runtime and save to struct
runtime = toc;
results.parameters.runtime = runtime;

%% save structs
save('evolution_test_1_results.mat', 'results');

%% visualize data
PlotResults('evolution_test_1_results.mat');

%% function to choose random points from a sphere
function [x,y,z] = rand_pick_sphere(n,a,b,X,Y,Z)
% Uniform points in a shell of inner radius a, outer radius b and center at
% (X,Y,Z)
% [x,y,z] = rand_pick_sphere(300,.5,.6);  % 300 points in shell between
% r = .5 and r = .6, with center at origin.
if nargin==3
    X = 0;
    Y = 0;
    Z = 0;
end
r1 = (rand(n,1)*(b^3-a^3)+a^3).^(1/3);
phi1 = acos(-1 + 2*rand(n,1));
th1 = 2*pi*rand(n,1);
% Convert to cart.
x = r1.*sin(phi1).*sin(th1) + X;
y = r1.*sin(phi1).*cos(th1) + Y;
z = r1.*cos(phi1) + Z;
end

%% Moran process function
function f = moran(payout,strategy_x,strategy_y,num_individuals)

%% populate vector with half instances of each strategy and create fitnesses array
population_strategies = NaN(num_individuals,1);
for i = 1:num_individuals/2
    population_strategies(i) = strategy_x;
    population_strategies(num_individuals/2+i) = strategy_y;
end
fitnesses = zeros(num_individuals,2);
fitnesses(:,1) = 1:num_individuals;

while range(population_strategies) ~= 0
    for i = 1:num_individuals-1
        for j = i+1:num_individuals
            %% Round-robin play
            rand_1 = rand;
            rand_2 = rand;
            if rand_1 <= population_strategies(i) && rand_2 <= population_strategies(j)
                fitnesses(i,2) = fitnesses(i,2)+payout{1,1}(1);
                fitnesses(j,2) = fitnesses(j,2)+payout{1,1}(2);
            elseif rand_1 <= population_strategies(i) && rand_2 > population_strategies(j)
                fitnesses(i,2) = fitnesses(i,2)+payout{1,2}(1);
                fitnesses(j,2) = fitnesses(j,2)+payout{1,2}(2);
            elseif rand_1 > population_strategies(i) && rand_2 <= population_strategies(j)
                fitnesses(i,2) = fitnesses(i,2)+payout{2,1}(1);
                fitnesses(j,2) = fitnesses(j,2)+payout{2,1}(2);
            elseif rand_1 > population_strategies(i) && rand_2 > population_strategies(j)
                fitnesses(i,2) = fitnesses(i,2)+payout{2,2}(1);
                fitnesses(j,2) = fitnesses(j,2)+payout{2,2}(2);
            end
        end
    end
    
    %% reproduction
    zeta = rand;
    r_index = find(cumsum(fitnesses(:,2))/sum(fitnesses(:,2)) > zeta, 1,'first');%Identify reproducing individual
    
    %% death
    d_index = randperm(num_individuals,1);
    
    %% next generation
    population_strategies(d_index) = population_strategies(r_index);
    
end
f = population_strategies(1);
end
