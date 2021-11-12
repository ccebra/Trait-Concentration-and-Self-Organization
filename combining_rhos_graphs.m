%% Clear figures
for j = 1:5
    figure(j);
    clf;
end
clear;

%% Load data from staghunt trials
load('evolution_test_moran_results_stag_lowGED.mat')
final_covariance_1 = squeeze(results.stepbysteparray(results.parameters.num_epochs,4,:));
empirical_rho_1 = (0.5-results.analysis.rho_empirical)';
grad_1 = abs(results.analysis.grad)';
drift_1 = results.parameters.genetic_drift*ones(results.parameters.num_experiments,1);
load('evolution_test_moran_results_stag_highGD.mat')
final_covariance_2 = squeeze(results.stepbysteparray(results.parameters.num_epochs,4,:));
empirical_rho_2 = (0.5-results.analysis.rho_empirical)';
grad_2 = abs(results.analysis.grad)';
drift_2 = results.parameters.genetic_drift*ones(results.parameters.num_experiments,1);
load('evolution_test_moran_results_stag_1000.mat')
final_covariance_3 = squeeze(results.stepbysteparray(results.parameters.num_epochs,4,:));
empirical_rho_3 = (0.5-results.analysis.rho_empirical)';
grad_3 = abs(results.analysis.grad)';
drift_3 = results.parameters.genetic_drift*ones(results.parameters.num_experiments,1);
load('evolution_test_moran_results_stag_vhighGD.mat')
final_covariance_4 = squeeze(results.stepbysteparray(results.parameters.num_epochs,4,:));
empirical_rho_4 = (0.5-results.analysis.rho_empirical)';
grad_4 = abs(results.analysis.grad)';
drift_4 = results.parameters.genetic_drift*ones(results.parameters.num_experiments,1);
load('evolution_test_moran_results_stag_vvhighGD.mat')
final_covariance_5 = squeeze(results.stepbysteparray(results.parameters.num_epochs,4,:));
empirical_rho_5 = (0.5-results.analysis.rho_empirical)';
grad_5 = abs(results.analysis.grad)';
drift_5 = results.parameters.genetic_drift*ones(results.parameters.num_experiments,1);

%% Combine vectors
final_cov_analysis = cat(1,final_covariance_1,final_covariance_2,final_covariance_3,final_covariance_4,final_covariance_5);
empirical_rho_analysis = cat(1,empirical_rho_1,empirical_rho_2,empirical_rho_3,empirical_rho_4,empirical_rho_5);
grad_analysis = cat(1,grad_1,grad_2,grad_3,grad_4,grad_5);
drift_analysis = cat(1,drift_1,drift_2,drift_3,drift_4,drift_5);
data_frame = cat(2,final_cov_analysis,empirical_rho_analysis,grad_analysis,drift_analysis);

%% Plot results
figure(1)
clf
hold on
scatter(data_frame(:,1),data_frame(:,2),10,data_frame(:,4),'filled');
colorbar;
grid on
set(gca,'xscale','log','yscale','log','ColorScale','log')
title('Concentration vs. $\rho$ Predicted (Drift coloring)','FontSize',16,'interpreter','latex')
xlabel('$\sqrt{\frac{1}{T}E[||X - \bar{x}||^2]}$','FontSize',16,'interpreter','latex')
ylabel('Empirical $0.5 - \rho$','FontSize',16,'interpreter','latex')
axis square
drawnow

figure(2)
clf
hold on
scatter(data_frame(:,1),data_frame(:,2),10,data_frame(:,3),'filled');
colorbar;
grid on
set(gca,'xscale','log','yscale','log')
title('Concentration vs. $\rho$ Predicted (Gradient coloring)','FontSize',16,'interpreter','latex')
xlabel('$\sqrt{\frac{1}{T}E[||X - \bar{x}||^2]}$','FontSize',16,'interpreter','latex')
ylabel('Empirical $0.5 - \rho$','FontSize',16,'interpreter','latex')
axis square
drawnow

%% Load data from chicken trials
load('evolution_test_moran_results_chicken_5E-5.mat')
final_covariance_1 = squeeze(results.stepbysteparray(results.parameters.num_epochs,4,:));
empirical_rho_1 = (0.5-results.analysis.rho_empirical)';
grad_1 = abs(results.analysis.grad)';
drift_1 = results.parameters.genetic_drift*ones(results.parameters.num_experiments,1);
mus_1 = squeeze(results.analysis.mus(50,1,:));
clusters_1 = squeeze(results.stepbysteparray(50,5,:));
load('evolution_test_moran_results_chicken_5E-4.mat')
final_covariance_2 = squeeze(results.stepbysteparray(results.parameters.num_epochs,4,:));
empirical_rho_2 = (0.5-results.analysis.rho_empirical)';
grad_2 = abs(results.analysis.grad)';
drift_2 = results.parameters.genetic_drift*ones(results.parameters.num_experiments,1);
mus_2 = squeeze(results.analysis.mus(50,1,:));
clusters_2 = squeeze(results.stepbysteparray(50,5,:));
load('evolution_test_moran_results_chicken_1E-4.mat')
final_covariance_3 = squeeze(results.stepbysteparray(results.parameters.num_epochs,4,:));
empirical_rho_3 = (0.5-results.analysis.rho_empirical)';
grad_3 = abs(results.analysis.grad)';
drift_3 = results.parameters.genetic_drift*ones(results.parameters.num_experiments,1);
mus_3 = squeeze(results.analysis.mus(50,1,:));
clusters_3 = squeeze(results.stepbysteparray(50,5,:));
load('evolution_test_moran_results_chicken_1E-3.mat')
final_covariance_4 = squeeze(results.stepbysteparray(results.parameters.num_epochs,4,:));
empirical_rho_4 = (0.5-results.analysis.rho_empirical)';
grad_4 = abs(results.analysis.grad)';
drift_4 = results.parameters.genetic_drift*ones(results.parameters.num_experiments,1);
mus_4 = squeeze(results.analysis.mus(50,1,:));
clusters_4 = squeeze(results.stepbysteparray(50,5,:));
load('evolution_test_moran_results_chicken_1E-2.mat')
final_covariance_5 = squeeze(results.stepbysteparray(results.parameters.num_epochs,4,:));
empirical_rho_5 = (0.5-results.analysis.rho_empirical)';
grad_5 = abs(results.analysis.grad)';
drift_5 = results.parameters.genetic_drift*ones(results.parameters.num_experiments,1);
mus_5 = squeeze(results.analysis.mus(50,1,:));
clusters_5 = squeeze(results.stepbysteparray(50,5,:));

%% Combine vectors
final_cov_analysis = cat(1,final_covariance_1,final_covariance_2,final_covariance_3,final_covariance_4,final_covariance_5);
empirical_rho_analysis = cat(1,empirical_rho_1,empirical_rho_2,empirical_rho_3,empirical_rho_4,empirical_rho_5);
grad_analysis = cat(1,grad_1,grad_2,grad_3,grad_4,grad_5);
drift_analysis = cat(1,drift_1,drift_2,drift_3,drift_4,drift_5);
mus_analysis = cat(1,mus_1,mus_2,mus_3,mus_4,mus_5);
clusters_analysis = cat(1,clusters_1,clusters_2,clusters_3,clusters_4,clusters_5);
chicken_data_frame = cat(2,final_cov_analysis,empirical_rho_analysis,grad_analysis,drift_analysis,mus_analysis,clusters_analysis);


%% Clean results for one final cluster only
idx = chicken_data_frame(:,6)==1;
cleaned_data = chicken_data_frame(idx,:);
%% Plot results
figure(3)
clf
hold on
scatter(chicken_data_frame(:,1),chicken_data_frame(:,2),10,chicken_data_frame(:,4),'filled');
colorbar;
grid on
set(gca,'xscale','log','yscale','log','ColorScale','log')
title('Concentration vs. $\rho$ Predicted (Drift coloring, Chicken)','FontSize',16,'interpreter','latex')
xlabel('$\sqrt{\frac{1}{T}E[||X - \bar{x}||^2]}$','FontSize',16,'interpreter','latex')
ylabel('Empirical $0.5 - \rho$','FontSize',16,'interpreter','latex')
axis square
drawnow

figure(4)
clf
hold on
scatter(chicken_data_frame(:,1),chicken_data_frame(:,2),10,chicken_data_frame(:,3),'filled');
colorbar;
grid on
set(gca,'xscale','log','yscale','log')
title('Concentration vs. $\rho$ Predicted (Gradient coloring, Chicken)','FontSize',16,'interpreter','latex')
xlabel('$\sqrt{\frac{1}{T}E[||X - \bar{x}||^2]}$','FontSize',16,'interpreter','latex')
ylabel('Empirical $0.5 - \rho$','FontSize',16,'interpreter','latex')
axis square
drawnow

figure(5)
clf
hold on
scatter(cleaned_data(:,1),cleaned_data(:,2),10,cleaned_data(:,5),'filled');
colorbar;
grid on
set(gca,'xscale','log','yscale','log')
title('Concentration vs. $\rho$ Predicted ($\mu$ coloring, Chicken)','FontSize',16,'interpreter','latex')
xlabel('$\sqrt{\frac{1}{T}E[||X - \bar{x}||^2]}$','FontSize',16,'interpreter','latex')
ylabel('Empirical $0.5 - \rho$','FontSize',16,'interpreter','latex')
axis square
drawnow