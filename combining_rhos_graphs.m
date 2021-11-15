%% Clear figures
for j = 1:6
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
load('evolution_test_moran_results_stag_5E-3.mat')
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
data_frame_1 = cat(2,final_covariance_1,empirical_rho_1,grad_1,drift_1);
data_frame_2 = cat(2,final_covariance_2,empirical_rho_2,grad_2,drift_2);
data_frame_3 = cat(2,final_covariance_3,empirical_rho_3,grad_3,drift_3);
data_frame_4 = cat(2,final_covariance_4,empirical_rho_4,grad_4,drift_4);
data_frame_5 = cat(2,final_covariance_5,empirical_rho_5,grad_5,drift_5);

%% Isolate points with low gradients
idx = data_frame_1(:,3) < 0.02;
lg_data_frame_1 = data_frame_1(idx,:);
idx = data_frame_2(:,3) < 0.02;
lg_data_frame_2 = data_frame_2(idx,:);
idx = data_frame_3(:,3) < 0.02;
lg_data_frame_3 = data_frame_3(idx,:);
idx = data_frame_4(:,3) < 0.02;
lg_data_frame_4 = data_frame_4(idx,:);
idx = data_frame_5(:,3) < 0.02;
lg_data_frame_5 = data_frame_5(idx,:);

full_matrix = cat(1,data_frame_1,data_frame_2,data_frame_3,data_frame_4,data_frame_5);
full_matrix = [log(full_matrix(:,1)),log(full_matrix(:,2))];
hull_matrix = cat(1,lg_data_frame_1,lg_data_frame_2,lg_data_frame_3,lg_data_frame_4,lg_data_frame_5);
hull_matrix = [log(hull_matrix(:,1)),log(hull_matrix(:,2))];
[k,av] = convhull(hull_matrix);
[k2,av2] = convhull(full_matrix);


%% 
figure(8)
clf
plot(full_matrix(:,1),full_matrix(:,2),'*');
hold on
plot(full_matrix(k2,1),full_matrix(k2,2));

k4 = [4829,4165,555,336]';
figure(7)
clf
plot(full_matrix(:,1),full_matrix(:,2),'*');
hold on
plot(full_matrix(k4,1),full_matrix(k4,2));
drawnow


%% Plot results
figure(1)
clf
hold on
scatter(lg_data_frame_1(:,1),lg_data_frame_1(:,2),70,lg_data_frame_1(:,3),'d');
hold on
scatter(lg_data_frame_2(:,1),lg_data_frame_2(:,2),70,lg_data_frame_2(:,3),'s');
hold on
scatter(lg_data_frame_3(:,1),lg_data_frame_3(:,2),70,lg_data_frame_3(:,3),'p');
hold on
scatter(lg_data_frame_4(:,1),lg_data_frame_4(:,2),70,lg_data_frame_4(:,3),'h');
hold on
scatter(lg_data_frame_5(:,1),lg_data_frame_5(:,2),70,lg_data_frame_5(:,3),'o');
colorbar;
grid on
set(gca,'xscale','log','yscale','log')
title('Concentration vs. $\rho$ Predicted (Low gradients only)','FontSize',16,'interpreter','latex')
xlabel('$\sqrt{\frac{1}{T}E[||X - \bar{x}||^2]}$','FontSize',16,'interpreter','latex')
ylabel('Empirical $0.5 - \rho$','FontSize',16,'interpreter','latex')
axis square
drawnow

figure(2)
clf
hold on
scatter(data_frame_1(:,1),data_frame_1(:,2),30,data_frame_1(:,3),'d');
hold on
scatter(data_frame_2(:,1),data_frame_2(:,2),30,data_frame_2(:,3),'s');
hold on
scatter(data_frame_3(:,1),data_frame_3(:,2),30,data_frame_3(:,3),'p');
hold on
scatter(data_frame_4(:,1),data_frame_4(:,2),30,data_frame_4(:,3),'h');
hold on
scatter(data_frame_5(:,1),data_frame_5(:,2),30,data_frame_5(:,3),'o');
colorbar;
grid on
set(gca,'xscale','log','yscale','log')
title('Concentration vs. $\rho$ Predicted (Gradient coloring)','FontSize',16,'interpreter','latex')
xlabel('$\sqrt{\frac{1}{T}E[||X - \bar{x}||^2]}$','FontSize',16,'interpreter','latex')
ylabel('Empirical $0.5 - \rho$','FontSize',16,'interpreter','latex')
axis square
drawnow

figure(3)
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
load('evolution_test_moran_results_chicken_5E-3.mat')
final_covariance_6 = squeeze(results.stepbysteparray(results.parameters.num_epochs,4,:));
empirical_rho_6 = (0.5-results.analysis.rho_empirical)';
grad_6 = abs(results.analysis.grad)';
drift_6 = results.parameters.genetic_drift*ones(results.parameters.num_experiments,1);
mus_6 = squeeze(results.analysis.mus(50,1,:));
clusters_6 = squeeze(results.stepbysteparray(50,5,:));

%% Combine vectors
final_cov_analysis = cat(1,final_covariance_1,final_covariance_2,final_covariance_3,final_covariance_4,final_covariance_5,final_covariance_6);
empirical_rho_analysis = cat(1,empirical_rho_1,empirical_rho_2,empirical_rho_3,empirical_rho_4,empirical_rho_5,empirical_rho_6);
grad_analysis = cat(1,grad_1,grad_2,grad_3,grad_4,grad_5,grad_6);
drift_analysis = cat(1,drift_1,drift_2,drift_3,drift_4,drift_5,drift_6);
mus_analysis = cat(1,mus_1,mus_2,mus_3,mus_4,mus_5,mus_6);
clusters_analysis = cat(1,clusters_1,clusters_2,clusters_3,clusters_4,clusters_5,clusters_6);
chicken_data_frame = cat(2,final_cov_analysis,empirical_rho_analysis,grad_analysis,drift_analysis,mus_analysis,clusters_analysis);


%% Clean results for one final cluster only
idx = chicken_data_frame(:,6)==1;
cleaned_data = chicken_data_frame(idx,:);
%% Plot results
figure(4)
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

figure(5)
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

figure(6)
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