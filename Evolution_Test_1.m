clear

%% settings/parameters
num_competitors = 500;
num_traits = 3; % we want to generalize to more dimensions
num_frequencies = 3;
trig_amplitude = 1;
linear_amplitude = 0;
genetic_drift = 0.01;
num_epochs = 25;
num_experiments = 25;
games_per_competitor = 100;
num_conv_rate = 10^6;
num_stds = 40;
f_mode = 2;
tic;

%% display
% figure(1)
% clf;
% scatter3(competitor_traits(:,1),competitor_traits(:,2),competitor_traits(:,3),20,'fill')
% grid on
% axis square
% xlim([-1,1])
% ylim([-1,1])
% zlim([-1,1])
% view([1,1,1])
% drawnow

%% generate edge to endpoint mapping
edge_to_endpoints = NaN(num_competitors*(num_competitors - 1)/2,2);
k = 0;
for i = 1:num_competitors-1
    for j = i+1:num_competitors
        k = k+1;
        edge_to_endpoints(k,:) = [i,j];
    end
end


%% define parameters for performance function
if f_mode == 1
    alpha = trig_amplitude*rand([num_traits,num_frequencies])/num_frequencies; % second dimension is the upper frequency of sines used to construct the random performance function
    % the higher that number the less smooth the performance
    phase = 2*pi*rand([num_traits,num_frequencies]);
elseif f_mode == 2
    alpha = trig_amplitude*randn([num_traits,num_traits,num_frequencies])/num_frequencies; % second dimension is the upper frequency of sines used to construct the random performance function
    phase = 2*pi*rand([num_traits,num_traits,num_frequencies]);
end

%% define performance function
if f_mode == 1
    f = @(x,y) example_performance_4(x,y,alpha,linear_amplitude,phase);
elseif f_mode == 2
    f = @(x,y) example_performance_5(x,y,alpha,linear_amplitude,phase);
end


%% generate data array that tracks data over multiple experiments
experiment_array = zeros(num_experiments,7);
step_by_step = NaN(num_epochs,6,num_experiments);
correlation_coefficient = NaN(num_experiments,num_stds);

%% loop over experiments
for experiment = 1:num_experiments
    
    %% randomly generate traits
    [x,y,z] = rand_pick_sphere(num_competitors,0,1); %Call function
    first_step = [x,y,z];
    competitor_traits = [x,y,z];
    
    %% generate data array that tracks transitivity/intransitivity over time (epochs)
    epoch_array = zeros(num_epochs,6);
    cluster_centroid = zeros(1,num_traits);
    evolution_over = 0;%Dummy variable to check if we can stop the evolution process
    final_epoch = num_epochs;%final_epoch will output the right row of the epoch_array to experiment_array, this starts at the maximal epoch and changes if evolution_over changes
    epoch = 1;
    
    %% loop over epochs
    while epoch < final_epoch && evolution_over == 0
        
        %% Calculate performance using the performance function
        X = competitor_traits(edge_to_endpoints(:,1),:);
        Y = competitor_traits(edge_to_endpoints(:,2),:);
        competition = sparse(edge_to_endpoints(:,1), edge_to_endpoints(:,2),...
            f(X,Y),num_competitors,num_competitors) - ...
            sparse(edge_to_endpoints(:,2), edge_to_endpoints(:,1),...
            f(X,Y),num_competitors,num_competitors);
        competition = full(competition);
        
        %% Sample event outcomes
        n_events = games_per_competitor*num_competitors;
        
        competitors = NaN([n_events,2]);
        
        Z = rand([n_events,1]); % this is all the random numbers we need
        p_win = NaN([n_events,1]);
        
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
                p_win(k) = (1 + exp(-competition(i,j)))^(-1); % p_win = logistic(performance)
                
            end
            
        end
        
        Outcomes = (Z <= p_win); % = 1 if i wins, 0 if j wins
        
        %% Compute win frequencies (only left competitor)
        win_freq = sum(reshape(Outcomes, [games_per_competitor,num_competitors])',2)/games_per_competitor;
        winfreq_with_indices = NaN([num_competitors, 2]);
        indices = [1:num_competitors]';
        winfreq_with_indices = [indices,win_freq];
        
        %% performing the HHD for a complete graph ***only works for complete graphs
        ratings = (1/num_competitors)*sum(competition,2);
        
        
        %% another way to do this without G
        F_t = ratings - ratings';
        F_c = competition - F_t;
        
        %% sizes
        Transitivity = norm(F_t,'fro')/sqrt(2);
        Intransitivity = norm(F_c,'fro')/sqrt(2);
        
        
        %% covariance calculations
        cov_at_step = trace(cov(competitor_traits));
        
        %% choose highest-fitness competitors
        winfreq_with_indices = sortrows(winfreq_with_indices,2, 'descend');
        highest_fitness = winfreq_with_indices(1:num_competitors/10,:);
        
        %% analyze for clusters (I moved this before the reproduction process, I'm not sure that's right)
        cluster_epsilon = genetic_drift;
        cluster_min_points = 5;
        [classes] = dbscan(competitor_traits,cluster_epsilon,cluster_min_points);
        n_classes = max(classes);
        
        %% cluster centroid (I moved the analyze for clusters function earlier to put this)
        if n_classes == 1
            new_cluster_centroid = mean(competitor_traits);
            if (norm(cluster_centroid - new_cluster_centroid)) < cluster_epsilon/5%Cluster centroid has stopped moving
                evolution_over = 1;
                final_epoch = epoch;
            end
            cluster_centroid = new_cluster_centroid;
        end
        
        %% reproduction process
        new_competitor_traits = NaN(num_competitors,num_traits);
        for i=0:(num_competitors/10-1)
            for j=1:10
                %while true
                drift_vector = genetic_drift*(rand(1,3)-1/2);
                child_vector = competitor_traits(highest_fitness(i+1,1),:) + drift_vector;
                if norm(child_vector) > 1
                    child_vector = 1/norm(child_vector)*child_vector;
                end
                new_competitor_traits(j+(10*i),:) = child_vector;
                %break
                %end
            end
        end
        competitor_traits = new_competitor_traits;
        
        %% display
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
    
    %% Calculate convergence rate given deviations of competitors from final cluster centroid
    std_convergence_list = 10.^(linspace(-1,-2.5,num_stds));
    for repopulate_step = 1:length(std_convergence_list)
        
        %% compute rho 
        m = 100; % size of network to sample per epoch
        epoch_bounds = [10,10^3]; % max and min number of epochs to run
        tol = 10^(-4); % desired tolerance on estimated rho
        [rhos.mean(experiment,repopulate_step),rhos.std(experiment,repopulate_step)] = estimate_rho_Gauss(f,cluster_centroid,std_convergence_list(repopulate_step)^2,tol,epoch_bounds,m);
        correlation_coefficient(experiment,repopulate_step) = rhos.mean(experiment,repopulate_step);
        
%         %% sample competitors
%         modifiers = std_convergence_list(repopulate_step)*randn(3*num_conv_rate,num_traits);
%         X = repmat(cluster_centroid,num_conv_rate,1) + modifiers(3*(0:num_conv_rate-1)+1,:);
%         Y = repmat(cluster_centroid,num_conv_rate,1) + modifiers(3*(0:num_conv_rate-1)+2,:);
%         Z = repmat(cluster_centroid,num_conv_rate,1) + modifiers(3*(0:num_conv_rate-1)+3,:);
%         repopulate_competition = NaN(num_conv_rate,2);
%         
%         %% Compute performance
%         repopulate_competition(:,1) = f(X,Y);
%         repopulate_competition(:,2) = f(X,Z);
%         
%         %% compute correlation
%         correlation_matrix = corr(repopulate_competition);
%         correlation_coefficient(experiment,repopulate_step) = correlation_matrix(2,1);
    end
    
    %% display
    figure(5)
    clf
    hold on
    plot(std_convergence_list,abs(0.5 - correlation_coefficient),'Linewidth',0.5)
    plot(std_convergence_list,abs(0.5 - nanmean(correlation_coefficient)),'k-','Linewidth',2)
    xlabel('Radius of Selected Competitors', 'FontSize', 18)
    ylabel('Correlation Coefficient', 'FontSize', 18)
    title('Convergence Rate','FontSize',18)
    grid on
    set(gca,'yscale','log')
    set(gca,'xscale','log')
    axis square
    drawnow
    
    %% print
    fprintf('\n Trial %d of %d complete \n',experiment,num_experiments)
end

%% Add parameters from this to experiments array and structs
experiment_array(experiment,1) = epoch_array(1,3);
experiment_array(experiment,2) = epoch_array(1,4);
experiment_array(experiment,3) = epoch_array(2,5);
experiment_array(experiment,4) = epoch_array(final_epoch,3);
experiment_array(experiment,5) = epoch_array(final_epoch,4);
experiment_array(experiment,6) = epoch_array(final_epoch,5);
experiment_array(experiment,7) = final_epoch;

results.intransitivity.initial = experiment_array(:,1);
results.covariance.initial = experiment_array(:,2);
results.clusters.initial = experiment_array(:,3);
results.intransitivity.final = experiment_array(:,4);
results.covariance.final = experiment_array(:,5);
results.clusters.final = experiment_array(:,6);
results.num.steps = experiment_array(:,7);%(Struct)

results.convergence_test.stds = std_convergence_list;
results.convergence_test.rhos = correlation_coefficient;
results.convergence_test.rho_mean = nanmean(correlation_coefficient);

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

%% visualize data
column_one = squeeze(step_by_step(:,1,:))';
column_two = squeeze(step_by_step(:,2,:))';
column_three = squeeze(step_by_step(:,3,:))';
column_four = squeeze(step_by_step(:,4,:))';
column_five = squeeze(step_by_step(:,5,:))';

figure(2)%Increased font size
clf
boxplot(column_three);%Grid on, set(gca,'yscale','log'),increase font size of axes
xlabel('Evolutionary Steps', 'FontSize', 18)
ylabel('Proportion Intransitivity', 'FontSize', 18)
title('Intransitivity Step by Step', 'FontSize', 18)
grid on


figure(3)
clf
boxplot(column_four);
xlabel('Evolutionary Steps', 'FontSize', 18)
ylabel('Covariance', 'FontSize', 18)
title('Covariance Step by Step', 'FontSize', 18)
grid on


figure(4)
clf
boxplot(column_five);
xlabel('Evolutionary Steps', 'FontSize', 18)
ylabel('Number of Clusters', 'FontSize', 18)
title('Clusters Step by Step', 'FontSize', 18)
grid on



%% Find overall information about our experiment
experiment_array = sortrows(experiment_array,4,'desc');
experiment_metadata = median(experiment_array);
median_steps = experiment_metadata(1,7);
median_transitivity = experiment_metadata(1,4);
median_variance = experiment_metadata(1,5);

runtime = toc;

%% save structs
save('evolution_test_1_results.mat', 'results');





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