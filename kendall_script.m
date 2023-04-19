%% Load data

load('evolution_test_kendall_results_lowGD.mat');
column_one = squeeze(results.stepbysteparray(:,7,:))';
column_one_means = mean(column_one,1);

load('evolution_test_kendall_results_medGD.mat');
column_two = squeeze(results.stepbysteparray(:,7,:))';
column_two_means = mean(column_two,1);
axis2 = 1:50;
axis2 = axis2 + 0.25;

%% Plot Figure
figure(1)
clf
hold on
boxplot(column_one,'BoxStyle','filled','OutlierSize',1,'PlotStyle','compact')
plot(column_one_means,'b-o','LineWidth',2)
boxplot(column_two,'BoxStyle','filled','Colors','r','OutlierSize',1,'PlotStyle','compact','positions',axis2)
plot(axis2,column_two_means,'r-o','LineWidth',2)
drawnow