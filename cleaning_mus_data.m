%% Clear figures
for j = 1:6
    figure(j)
    clf
end

%% Load data from chicken test runs
load('evolution_test_moran_results_chicken_1E-2.mat');
mus1 = squeeze(results.analysis.mus(50,1,:));
hess1 = squeeze(results.analysis.Hessian.xx);
grad1 = results.analysis.grad';
clusters1 = squeeze(results.stepbysteparray(50,5,:));
load('evolution_test_moran_results_chicken_1E-3.mat');
mus2 = squeeze(results.analysis.mus(50,1,:));
hess2 = squeeze(results.analysis.Hessian.xx);
grad2 = results.analysis.grad';
clusters2 = squeeze(results.stepbysteparray(50,5,:));
load('evolution_test_moran_results_chicken_1E-4.mat');
mus3 = squeeze(results.analysis.mus(50,1,:));
hess3 = squeeze(results.analysis.Hessian.xx);
grad3 = results.analysis.grad';
clusters3 = squeeze(results.stepbysteparray(50,5,:));
load('evolution_test_moran_results_chicken_5E-4.mat');
mus4 = squeeze(results.analysis.mus(50,1,:));
hess4 = squeeze(results.analysis.Hessian.xx);
grad4 = results.analysis.grad';
clusters4 = squeeze(results.stepbysteparray(50,5,:));
load('evolution_test_moran_results_chicken_5E-5.mat');
mus5 = squeeze(results.analysis.mus(50,1,:));
hess5 = squeeze(results.analysis.Hessian.xx);
grad5 = results.analysis.grad';
clusters5 = squeeze(results.stepbysteparray(50,5,:));

%% Combine data
mus_analysis = cat(1,mus1,mus2,mus3,mus4,mus5);
hess_analysis = cat(1,hess1,hess2,hess3,hess4,hess5);
grad_analysis = cat(1,grad1,grad2,grad3,grad4,grad5);
clusters_analysis = cat(1,clusters1,clusters2,clusters3,clusters4,clusters5);
uncleaned_data = cat(2,mus_analysis,hess_analysis,grad_analysis,clusters_analysis);

%% Plot uncleaned data
figure(1)
clf
hold on
scatter(uncleaned_data(:,1),uncleaned_data(:,3),8,uncleaned_data(:,4),'filled')
colorbar;
grid on
title('Gradient value against final $\mu$ of the data cluster','FontSize',16,'Interpreter','latex')
xlabel('Final $\mu$','Interpreter','latex')
ylabel('Gradient value (not norm)')
axis square
drawnow

figure(2)
clf
hold on
scatter(uncleaned_data(:,1),uncleaned_data(:,2),8,uncleaned_data(:,4),'filled')
colorbar;
grid on
title('Hessian-xx value against final $\mu$ of the data cluster','FontSize',16,'Interpreter','latex')
xlabel('Final $\mu$','Interpreter','latex')
ylabel('Hessian value (not norm)')
axis square
drawnow

figure(3)
clf
hold on
scatter(uncleaned_data(:,1),abs(uncleaned_data(:,2)),8,uncleaned_data(:,4),'filled')
colorbar;
grid on
title('Hessian-xx norm against final $\mu$ of the data cluster','FontSize',16,'Interpreter','latex')
xlabel('Final $\mu$','Interpreter','latex')
ylabel('Hessian value (norm)')
axis square
drawnow

%% Clean data
idx = uncleaned_data(:,4)==1;
cleaned_data = uncleaned_data(idx,:);

%% Plot cleaned data
figure(4)
clf
hold on
scatter(cleaned_data(:,1),cleaned_data(:,3),8,[0.6350 0.0780 0.1840])
grid on
title('Gradient value against final $\mu$ of the data cluster','FontSize',16,'Interpreter','latex')
xlabel('Final $\mu$','Interpreter','latex')
ylabel('Gradient value (not norm)')
axis square
drawnow

figure(5)
clf
hold on
scatter(cleaned_data(:,1),cleaned_data(:,2),8,[0.6350 0.0780 0.1840])
grid on
title('Hessian-xx value against final $\mu$ of the data cluster','FontSize',16,'Interpreter','latex')
xlabel('Final $\mu$','Interpreter','latex')
ylabel('Hessian value (not norm)')
axis square
drawnow

figure(6)
clf
hold on
scatter(cleaned_data(:,1),abs(cleaned_data(:,2)),8,[0.6350 0.0780 0.1840])
grid on
title('Hessian-xx norm against final $\mu$ of the data cluster','FontSize',16,'Interpreter','latex')
xlabel('Final $\mu$','Interpreter','latex')
ylabel('Hessian value (norm)')
axis square
drawnow