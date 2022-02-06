function PlotResults(filename)

%% Load input data
load(filename)

%% Set up data and split into interior and boundary portions
column_one = [squeeze(results.stepbysteparray(:,1,:))',results.maxcoordinate];
column_two = [squeeze(results.stepbysteparray(:,2,:))',results.maxcoordinate];
column_three = [squeeze(results.stepbysteparray(:,3,:))',results.maxcoordinate];
column_four = [squeeze(results.stepbysteparray(:,4,:))',results.maxcoordinate];
column_five = [squeeze(results.stepbysteparray(:,5,:))',results.maxcoordinate];

column_one_interior = column_one(column_one(:,results.parameters.num_epochs+1)<1-1.96*results.parameters.genetic_drift, :);
column_one_boundary = column_one(column_one(:,results.parameters.num_epochs+1)>=1-1.96*results.parameters.genetic_drift, :);
column_two_interior = column_two(column_two(:,results.parameters.num_epochs+1)<1-1.96*results.parameters.genetic_drift, :);
column_two_boundary = column_two(column_two(:,results.parameters.num_epochs+1)>=1-1.96*results.parameters.genetic_drift, :);
column_three_interior = column_three(column_three(:,results.parameters.num_epochs+1)<1-1.96*results.parameters.genetic_drift, :);
column_three_boundary = column_three(column_three(:,results.parameters.num_epochs+1)>=1-1.96*results.parameters.genetic_drift, :);
column_four_interior = column_four(column_four(:,results.parameters.num_epochs+1)<1-1.96*results.parameters.genetic_drift, :);
column_four_boundary = column_four(column_four(:,results.parameters.num_epochs+1)>=1-1.96*results.parameters.genetic_drift, :);
column_five_interior = column_five(column_five(:,results.parameters.num_epochs+1)<1-1.96*results.parameters.genetic_drift, :);
column_five_boundary = column_five(column_five(:,results.parameters.num_epochs+1)>=1-1.96*results.parameters.genetic_drift, :);

column_three_interior(:,end) = [];
column_three_boundary(:,end) = [];
column_four_interior(:,end) = [];
column_four_boundary(:,end) = [];
column_five_interior(:,end) = [];
column_five_boundary(:,end) = [];

%% Manipulate data for norms of gradient and Hessian
gradient_norm = [results.norms.gradient,results.maxcoordinate];
hessian_xx_norm = [results.norms.xxhessian,results.maxcoordinate];
hessian_xy_norm = [results.norms.xyhessian,results.maxcoordinate];
gradient_interior = gradient_norm(gradient_norm(:,2)<1-1.96*results.parameters.genetic_drift, :);
gradient_boundary = gradient_norm(gradient_norm(:,2)>=1-1.96*results.parameters.genetic_drift, :);
hessian_xx_interior = hessian_xx_norm(hessian_xx_norm(:,2)<1-1.96*results.parameters.genetic_drift, :);
hessian_xx_boundary = hessian_xx_norm(hessian_xx_norm(:,2)>=1-1.96*results.parameters.genetic_drift, :);
hessian_xy_interior = hessian_xy_norm(hessian_xy_norm(:,2)<1-1.96*results.parameters.genetic_drift, :);
hessian_xy_boundary = hessian_xy_norm(hessian_xy_norm(:,2)>=1-1.96*results.parameters.genetic_drift, :);


%% Plot intransitivity on interior and boundary
xcoords = (1:50);
offset1xcoords = xcoords + 0.25;

figure(22)
clf
hold on
boxplot(column_three(:,[1:50]),'Colors','b','PlotStyle','compact','OutlierSize',1,'positions',xcoords);
yline(0);
%boxplot(0.1,column_four(:,[1:50]),'Colors','g','PlotStyle','compact');
%Grid on, set(gca,'yscale','log'),increase font size of axes
xlabel('Evolutionary Steps', 'FontSize', 36, 'interpreter', 'latex')
ylabel('Proportion Intransitivity', 'FontSize', 36, 'interpreter', 'latex')
ylim([-0.1,1.1])
set(gca, 'FontSize',28)
title('Intransitivity Step by Step (zero linear amplitude)', 'FontSize', 28, 'interpreter', 'latex')
grid on

figure(2)
clf
boxplot(column_three_interior);%Grid on, set(gca,'yscale','log'),increase font size of axes
xlabel('Evolutionary Steps', 'FontSize', 18, 'interpreter', 'latex')
ylabel('Proportion Intransitivity', 'FontSize', 18, 'interpreter', 'latex')
ylim([-0.1,1.1])
title('Intransitivity Step by Step (Interior)', 'FontSize', 18, 'interpreter', 'latex')
grid on

figure(3)
clf
boxplot(column_three_boundary);%Grid on, set(gca,'yscale','log'),increase font size of axes
xlabel('Evolutionary Steps', 'FontSize', 18, 'interpreter', 'latex')
ylabel('Proportion Intransitivity', 'FontSize', 18, 'interpreter', 'latex')
ylim([-0.1,1.1])
title('Intransitivity Step by Step (Boundary)', 'FontSize', 18, 'interpreter', 'latex')
grid on

%% Plot covariance on interior and boundary
figure(23)
clf
boxplot(column_four(:,[1:50]));
xlabel('Evolutionary Steps', 'FontSize', 18, 'interpreter', 'latex')
ylabel('Covariance', 'FontSize', 18, 'interpreter', 'latex')
ylim([-0.1,1.1])
title('Covariance Step by Step', 'FontSize', 18, 'interpreter', 'latex')
grid on

figure(4)
clf
boxplot(column_four_interior);
xlabel('Evolutionary Steps', 'FontSize', 18, 'interpreter', 'latex')
ylabel('Covariance', 'FontSize', 18, 'interpreter', 'latex')
ylim([-0.1,1.1])
title('Covariance Step by Step (Interior)', 'FontSize', 18, 'interpreter', 'latex')
grid on

figure(5)
clf
boxplot(column_four_boundary);
xlabel('Evolutionary Steps', 'FontSize', 18, 'interpreter', 'latex')
ylabel('Covariance', 'FontSize', 18, 'interpreter', 'latex')
ylim([-0.1,1.1])
title('Covariance Step by Step (Boundary)', 'FontSize', 18, 'interpreter', 'latex')
grid on

%% Plot number of clusters on interior and boundary
figure(24)
clf
boxplot(column_five(:,[1:50]));
xlabel('Evolutionary Steps', 'FontSize', 18, 'interpreter', 'latex')
ylabel('Number of Clusters', 'FontSize', 18, 'interpreter', 'latex')
title('Clusters Step by Step', 'FontSize', 18, 'interpreter', 'latex')
axis tight
grid on

figure(6)
clf
boxplot(column_five_interior);
xlabel('Evolutionary Steps', 'FontSize', 18, 'interpreter', 'latex')
ylabel('Number of Clusters', 'FontSize', 18, 'interpreter', 'latex')
title('Clusters Step by Step (Interior)', 'FontSize', 18, 'interpreter', 'latex')
axis tight
grid on

figure(7)
clf
boxplot(column_five_boundary);
xlabel('Evolutionary Steps', 'FontSize', 18, 'interpreter', 'latex')
ylabel('Number of Clusters', 'FontSize', 18, 'interpreter', 'latex')
title('Clusters Step by Step (Boundary)', 'FontSize', 18, 'interpreter', 'latex')
axis tight
grid on

%% display convergence rate graph (boundary)
convergence_test_stds = [results.convergence_test.boundary.stds,results.convergence_test.interior.stds];
convergence_test_rhos = [results.convergence_test.boundary.rhos,results.convergence_test.interior.rhos];
convergence_test_rho_means = [results.convergence_test.boundary.rho_mean,results.convergence_test.interior.rho_mean];
figure(24)
clf
hold on
plot(convergence_test_stds,abs(0.5-convergence_test_rhos),'Linewidth',0.5)
plot(convergence_test_stds,abs(0.5-convergence_test_rho_means),'k-','Linewidth',2)
set(gca, 'FontSize',28)
xlabel('Radius of Selected Competitors', 'FontSize', 36, 'interpreter', 'latex')
ylabel('Correlation Coefficient', 'FontSize', 36, 'interpreter', 'latex')
title('Rate of Convergence Test','FontSize',36, 'interpreter', 'latex')
l = legend('data','fitline');
set(l,'FontSize',32,'interpreter','latex','Location','best')
grid on
set(gca,'yscale','log')
set(gca,'xscale','log')
axis square
drawnow

figure(8)
clf
hold on
plot(results.convergence_test.boundary.stds,abs(0.5 - results.convergence_test.boundary.rhos),'Linewidth',0.5)
plot(results.convergence_test.boundary.stds,abs(0.5 - results.convergence_test.boundary.rho_mean),'k-','Linewidth',2)
set(gca, 'FontSize',14)
xlabel('Radius of Selected Competitors', 'FontSize', 18, 'interpreter', 'latex')
ylabel('Correlation Coefficient', 'FontSize', 18, 'interpreter', 'latex')
title('Convergence Rate for Boundary','FontSize',18, 'interpreter', 'latex')
l = legend('data','fitline');
set(l,'FontSize',16,'interpreter','latex','Location','best')
grid on
set(gca,'yscale','log')
set(gca,'xscale','log')
axis square
drawnow

%% display convergence rate graph (interior)
figure(9)
clf
hold on
plot(results.convergence_test.interior.stds,abs(0.5 - results.convergence_test.interior.rhos),'Linewidth',0.5)
plot(results.convergence_test.interior.stds,abs(0.5 - results.convergence_test.interior.rho_mean),'k-','Linewidth',2)
set(gca, 'FontSize',14)
xlabel('Radius of Selected Competitors', 'FontSize', 18, 'interpreter', 'latex')
ylabel('Correlation Coefficient', 'FontSize', 18, 'interpreter', 'latex')
title('Convergence Rate for Interior','FontSize',18, 'interpreter', 'latex')
l = legend('data','fitline');
set(l,'FontSize',16,'interpreter','latex','Location','best')
grid on
set(gca,'yscale','log')
set(gca,'xscale','log')
axis square
drawnow

%% plot cluster centroid proximity to boundary
figure(10)
hist(results.maxcoordinate)
xlabel('Max Coordinate', 'FontSize', 18, 'interpreter', 'latex')
ylabel('', 'FontSize', 18, 'interpreter', 'latex')
title('Number of Trials at Each Max Coordinate', 'FontSize', 18, 'interpreter', 'latex')
axis tight
grid on

%% plot histograms for gradient and Hessian norms
figure(16)
clf
hist(gradient_interior(:,1));
xlabel('Norm', 'FontSize', 18, 'interpreter', 'latex')
ylabel('', 'FontSize', 18, 'interpreter', 'latex')
title('Norm of the Gradient (Interior)', 'FontSize', 18, 'interpreter', 'latex')
axis tight
grid on

figure(17)
clf
hist(gradient_boundary(:,1));
xlabel('Norm', 'FontSize', 18, 'interpreter', 'latex')
ylabel('', 'FontSize', 18, 'interpreter', 'latex')
title('Norm of the Gradient (Boundary)', 'FontSize', 18, 'interpreter', 'latex')
axis tight
grid on

figure(18)
clf
hist(hessian_xx_interior(:,1));
xlabel('Norm', 'FontSize', 18, 'interpreter', 'latex')
ylabel('', 'FontSize', 18, 'interpreter', 'latex')
title('Norm of the xx-Hessian (Interior)', 'FontSize', 18, 'interpreter', 'latex')
axis tight
grid on

figure(19)
clf
hist(hessian_xx_boundary(:,1));
xlabel('Norm', 'FontSize', 18, 'interpreter', 'latex')
ylabel('', 'FontSize', 18, 'interpreter', 'latex')
title('Norm of the xx-Hessian (Boundary)', 'FontSize', 18, 'interpreter', 'latex')
axis tight
grid on

figure(20)
clf
hist(hessian_xy_interior(:,1));
xlabel('Norm', 'FontSize', 18, 'interpreter', 'latex')
ylabel('', 'FontSize', 18, 'interpreter', 'latex')
title('Norm of the xy-Hessian (Interior)', 'FontSize', 18, 'interpreter', 'latex')
axis tight
grid on

figure(21)
clf
hist(hessian_xy_boundary(:,1));
xlabel('Norm', 'FontSize', 18, 'interpreter', 'latex')
ylabel('', 'FontSize', 18, 'interpreter', 'latex')
title('Norm of the xy-Hessian (Boundary)', 'FontSize', 18, 'interpreter', 'latex')
axis tight
grid on

%% display rhos (experimental versus predicted)
figure(11)
clf
colormap turbo
hold on
scatter3(0.5 - results.analysis.rho,0.5 - results.analysis.rho_empirical,sqrt(results.covariance.final/results.parameters.num_traits),...
    15,sqrt(results.covariance.final),'fill')
colorbar
plot3([0.9*min(0.5-results.analysis.rho_empirical),1],[0.9*min(0.5-results.analysis.rho_empirical),1],[10^-2,10^(-2)],'k-','Linewidth',1)
grid on
set(gca,'xscale','log','yscale','log','zscale','log','ColorScale','log')
title('$\rho$ Predicted vs. Actual','FontSize',32,'interpreter','latex')
xlabel('Predicted $0.5 - \rho$','FontSize',32,'interpreter','latex')
ylabel('Empirical $0.5 - \rho$','FontSize',32,'interpreter','latex')
zlabel('$\sqrt{\frac{1}{T}E[||X - \bar{x}||^2]}$','FontSize',32,'interpreter','latex')
axis square
drawnow

%% display to test
figure(12)
clf
hold on
plot(results.convergence_test.boundary.stds,0.5 - mean(results.convergence_test.boundary.rho_predictions,'omitnan'),'k-','Linewidth',1.5)
plot(results.convergence_test.boundary.stds,0.5 - results.convergence_test.boundary.rho_predictions,'Linewidth',1)
grid on
set(gca,'yscale','log','xscale','log')
xlabel('Standard Deviation','FontSize',16,'interpreter','latex')
ylabel('$0.5 - \rho$','FontSize',16,'interpreter','latex')
title('Predicted $\rho$ on Boundary','FontSize',16,'interpreter','latex')
l = legend('Average over Trials','Per Trial');
set(l,'FontSize',14,'interpreter','latex','location','best');
drawnow

figure(14)
clf
hold on
plot(results.convergence_test.boundary.stds,mean(abs(results.convergence_test.boundary.rho_mean - results.convergence_test.boundary.rho_predictions)./results.convergence_test.boundary.rho_predictions,'omitnan'),...
    'k-','Linewidth',2)
plot(results.convergence_test.boundary.stds,abs(results.convergence_test.boundary.rho_mean - results.convergence_test.boundary.rho_predictions),...
    'Linewidth',1)
grid on
set(gca,'yscale','log','xscale','log')
xlabel('Standard Deviation','FontSize',16,'interpreter','latex')
ylabel('Relative Error','FontSize',16,'interpreter','latex')
title('Error in Estimated $\rho$ on Boundary','FontSize',16,'interpreter','latex')
l = legend('Average over Trials','Per Trial');
set(l,'FontSize',14,'interpreter','latex','location','best');
drawnow

%% display to test
figure(13)
clf
hold on
plot(results.convergence_test.interior.stds,0.5 - mean(results.convergence_test.interior.rho_predictions,'omitnan'),'k-','Linewidth',1.5)
plot(results.convergence_test.interior.stds,0.5 - results.convergence_test.interior.rho_predictions,'Linewidth',1)
grid on
set(gca,'yscale','log','xscale','log')
xlabel('Standard Deviation','FontSize',16,'interpreter','latex')
ylabel('$0.5 - \rho$','FontSize',16,'interpreter','latex')
title('Predicted $\rho$ on Interior','FontSize',16,'interpreter','latex')
l = legend('Average over Trials','Per Trial');
set(l,'FontSize',14,'interpreter','latex','location','best');
drawnow

figure(15)
clf
hold on
plot(results.convergence_test.interior.stds,mean(abs(results.convergence_test.interior.rho_mean - results.convergence_test.interior.rho_predictions)./results.convergence_test.interior.rho_predictions,'omitnan'),...
    'k-','Linewidth',2)
plot(results.convergence_test.interior.stds,abs(results.convergence_test.interior.rho_mean - results.convergence_test.interior.rho_predictions),...
    'Linewidth',1)
grid on
set(gca,'yscale','log','xscale','log')
xlabel('Standard Deviation','FontSize',16,'interpreter','latex')
ylabel('Relative Error','FontSize',16,'interpreter','latex')
title('Error in Estimated $\rho$ on Interior','FontSize',16,'interpreter','latex')
l = legend('Average over Trials','Per Trial');
set(l,'FontSize',14,'interpreter','latex','location','best');
drawnow