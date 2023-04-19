%% Load input data
load('Test_2_Results_small_drift.mat');
smalldrift_intrans = results.intransitivity.final;
smalldrift_drift = results.parameters.genetic_drift*ones(1000,1)+0.00005*randn(1000,1);
smalldrift_covs = results.covariance.final;
smalldrift_column_three = [squeeze(results.stepbysteparray(:,3,:))',results.maxcoordinate];
load('Test_1_results.mat');
mediumdrift_intrans = results.intransitivity.final;
mediumdrift_drift = results.parameters.genetic_drift*ones(1000,1)+0.0002*randn(1000,1);
mediumdrift_covs = results.covariance.final;
mediumdrift_column_three = [squeeze(results.stepbysteparray(:,3,:))',results.maxcoordinate];
load('Test_3_Results_big_drift.mat');
highdrift_intrans = results.intransitivity.final;
highdrift_drift = results.parameters.genetic_drift*ones(1000,1)+0.0005*randn(1000,1);
highdrift_covs = results.covariance.final;
highdrift_column_three = [squeeze(results.stepbysteparray(:,3,:))',results.maxcoordinate];
load('Test_4_Results_very_big_drift.mat');
vhighdrift_intrans = results.intransitivity.final;
vhighdrift_drift = results.parameters.genetic_drift*ones(1000,1)+0.001*randn(1000,1);
vhighdrift_covs = results.covariance.final;
vhighdrift_column_three = [squeeze(results.stepbysteparray(:,3,:))',results.maxcoordinate];
load('Test_Results_zero_linear.mat')
zlinear_column_three = [squeeze(results.stepbysteparray(:,3,:))',results.maxcoordinate];
load('Test_Results_half.mat')
htrig_column_three = [squeeze(results.stepbysteparray(:,3,:))',results.maxcoordinate];
load('Test_Results_4.mat')
Four_mode_column_three = [squeeze(results.stepbysteparray(:,3,:))',results.maxcoordinate];
load('Test_Results_6.mat')
Six_mode_column_three = [squeeze(results.stepbysteparray(:,3,:))',results.maxcoordinate];
load('evolution_test_moran_results_chicken_5E-2.mat')
column_three_vvhighGD = squeeze(results.stepbysteparray(:,3,:))';
load('evolution_test_moran_results_chicken_1E-2.mat')
column_three_vhighGD = squeeze(results.stepbysteparray(:,3,:))';
load('evolution_test_moran_results_chicken_5E-3.mat')
column_three_highGD = squeeze(results.stepbysteparray(:,3,:))';
load('evolution_test_moran_results_chicken_1E-3.mat')
column_three_mhighGD = squeeze(results.stepbysteparray(:,3,:))';
load('evolution_test_moran_results_chicken_5E-4.mat')
column_three_medGD = squeeze(results.stepbysteparray(:,3,:))';
load('evolution_test_moran_results_chicken_1E-4.mat')
column_three_mlowGD = squeeze(results.stepbysteparray(:,3,:))';
load('evolution_test_moran_results_chicken_5E-5.mat')
column_three_lowGD = squeeze(results.stepbysteparray(:,3,:))';
smalldrift_arr = [smalldrift_intrans,smalldrift_drift,smalldrift_covs];
mediumdrift_arr = [mediumdrift_intrans,mediumdrift_drift,mediumdrift_covs];
highdrift_arr = [highdrift_intrans,highdrift_drift,highdrift_covs];
vhighdrift_arr = [vhighdrift_intrans,vhighdrift_drift,vhighdrift_covs];
drift_arr = [smalldrift_arr;mediumdrift_arr;highdrift_arr;vhighdrift_arr];

%% Plot rhos predicted vs actual
figure(1)
clf
colormap turbo
scatter(drift_arr(:,2),drift_arr(:,1),20,drift_arr(:,3),'fill');
colorbar
grid on
set(gca,'FontSize',24)
set(gca,'xscale','log','yscale','log','ColorScale','log')
xlim([0.0005 0.1])
title('Genetic drift versus final intransitivity','FontSize',32,'interpreter','latex')
xlabel('Genetic drift parameter','FontSize',32,'interpreter','latex')
ylabel('Final intransitivity','FontSize',32,'interpreter','latex')
axis square

%% Plot genetic drift adjusted intransitivity boxplot
xcoords = (1:50);
offset1xcoords = xcoords + 0.25;
offset2xcoords = xcoords + 0.5;
offset3xcoords = xcoords + 0.75;
smallmed = median(smalldrift_column_three(:,[1:50]));
mediummed = median(mediumdrift_column_three(:,[1:50]));
highmed = median(highdrift_column_three(:,[1:50]));
vhighmed = median(vhighdrift_column_three(:,[1:50]));
zlinearmed = median(zlinear_column_three(:,[1:50]));
htrigmed = median(htrig_column_three(:,[1:50]));
Fourmodemed = median(Four_mode_column_three(:,[1:50]));
Sixmodemed = median(Six_mode_column_three(:,[1:50]));
medvvhigh = median(column_three_vvhighGD(:,[1:50]));
medvhigh = median(column_three_vhighGD(:,[1:50]));
medhigh = median(column_three_highGD(:,[1:50]));
medmhigh = median(column_three_mhighGD(:,[1:50]));
medmed = median(column_three_medGD(:,[1:50]));
medmlow = median(column_three_mlowGD(:,[1:50]));
medlow = median(column_three_lowGD(:,[1:50]));

figure(2)
clf
hold on
boxplot(smalldrift_column_three(:,[1:50]),'Colors','b','PlotStyle','compact','OutlierSize',1,'positions',xcoords);
boxplot(mediumdrift_column_three(:,[1:50]),'Colors','r','PlotStyle','compact','OutlierSize',1,'positions',offset1xcoords);
boxplot(highdrift_column_three(:,[1:50]),'Colors','m','PlotStyle','compact','OutlierSize',1,'positions',offset2xcoords);
boxplot(vhighdrift_column_three(:,[1:50]),'Colors','k','PlotStyle','compact','OutlierSize',1,'positions',offset3xcoords);
yline(0);
plot(xcoords, smallmed,'b-','LineWidth',1.75)
plot(offset1xcoords, mediummed,'r-','LineWidth',1.75)
plot(offset2xcoords, highmed,'m-','LineWidth',1.75)
plot(offset3xcoords, vhighmed,'k-','LineWidth',1.75)
set(gca,'FontSize',20)
a=gca
%a.YRuler.TickLabelGapOffset = 36;  
xticks([0 5 10 15 20 25 30 35 40 45 50])
xticklabels({'0', '5', '10', '15', '20', '25', '30', '35', '40', '45', '50'})
xlabel('Evolutionary Steps', 'FontSize', 28, 'interpreter', 'latex')
xtickangle(45)
ylabel('Proportion Intransitivity', 'FontSize', 28, 'interpreter', 'latex')
ylim([-0.1,1.1])
title('Intransitivity Step by Step (varying genetic drift)', 'FontSize', 28, 'interpreter', 'latex')
grid on

figure(3)
clf
hold on
boxplot(mediumdrift_column_three(:,[1:50]),'Colors','r','PlotStyle','compact','OutlierSize',1,'positions',offset1xcoords);
boxplot(zlinear_column_three(:,[1:50]),'Colors','b','PlotStyle','compact','OutlierSize',1,'positions',offset2xcoords);
boxplot(htrig_column_three(:,[1:50]),'Colors','k','PlotStyle','compact','OutlierSize',1,'positions',offset3xcoords);
yline(0);
plot(offset1xcoords, mediummed,'r-','LineWidth',1.75)
plot(offset2xcoords, zlinearmed,'b-','LineWidth',1.75)
plot(offset3xcoords, htrigmed,'k-','LineWidth',1.75)
set(gca,'FontSize',20)
a=gca
%a.YRuler.TickLabelGapOffset = 36;  
xticks([0 5 10 15 20 25 30 35 40 45 50])
xticklabels({'0', '5', '10', '15', '20', '25', '30', '35', '40', '45', '50'})
xlabel('Evolutionary Steps', 'FontSize', 28, 'interpreter', 'latex')
xtickangle(45)
ylabel('Proportion Intransitivity', 'FontSize', 28, 'interpreter', 'latex')
ylim([-0.1,1.1])
title('Intransitivity Step by Step (varying trig/linear amplitude)', 'FontSize', 28, 'interpreter', 'latex')
grid on

figure(4)
clf
hold on
boxplot(mediumdrift_column_three(:,[1:50]),'Colors','r','PlotStyle','compact','OutlierSize',1,'positions',offset1xcoords);
boxplot(Four_mode_column_three(:,[1:50]),'Colors','b','PlotStyle','compact','OutlierSize',1,'positions',offset2xcoords);
boxplot(Six_mode_column_three(:,[1:50]),'Colors','k','PlotStyle','compact','OutlierSize',1,'positions',offset3xcoords);
yline(0);
plot(offset1xcoords, mediummed,'r-','LineWidth',1.75)
plot(offset2xcoords, Fourmodemed,'b-','LineWidth',1.75)
plot(offset3xcoords, Sixmodemed,'k-','LineWidth',1.75)
set(gca,'FontSize',20)
a=gca
%a.YRuler.TickLabelGapOffset = 36;  
xticks([0 5 10 15 20 25 30 35 40 45 50])
xticklabels({'0', '5', '10', '15', '20', '25', '30', '35', '40', '45', '50'})
xlabel('Evolutionary Steps', 'FontSize', 28, 'interpreter', 'latex')
xtickangle(45)
ylabel('Proportion Intransitivity', 'FontSize', 28, 'interpreter', 'latex')
ylim([-0.1,1.1])
title('Intransitivity Step by Step (varying number of modes)', 'FontSize', 28, 'interpreter', 'latex')
grid on

figure(5)
clf
hold on
boxplot(column_three_vvhighGD(:,[1:50]),'Colors','b','PlotStyle','compact','OutlierSize',1,'positions',xcoords);
boxplot(column_three_highGD(:,[1:50]),'Colors','r','PlotStyle','compact','OutlierSize',1,'positions',offset1xcoords);
boxplot(column_three_medGD(:,[1:50]),'Colors','m','PlotStyle','compact','OutlierSize',1,'positions',offset2xcoords);
boxplot(column_three_lowGD(:,[1:50]),'Colors','k','PlotStyle','compact','OutlierSize',1,'positions',offset3xcoords);
yline(0);
plot(xcoords, medvvhigh,'b-','LineWidth',1.75)
plot(offset1xcoords, medhigh,'r-','LineWidth',1.75)
plot(offset2xcoords, medmed,'m-','LineWidth',1.75)
plot(offset3xcoords, medlow,'k-','LineWidth',1.75)
set(gca,'FontSize',20)
a=gca;
%a.YRuler.TickLabelGapOffset = 36;  
xticks([0 5 10 15 20 25 30 35 40 45 50])
xticklabels({'0', '5', '10', '15', '20', '25', '30', '35', '40', '45', '50'})
xlabel('Evolutionary Steps', 'FontSize', 28, 'interpreter', 'latex')
xtickangle(45)
ylabel('Proportion Intransitivity', 'FontSize', 28, 'interpreter', 'latex')
ylim([-0.1,1.1])
title('Intransitivity Step by Step (chicken, varying genetic drift)', 'FontSize', 28, 'interpreter', 'latex')
grid on

figure(6)
hold on
yline(0);
plot(xcoords, medvvhigh,'b-','LineWidth',1.75)
plot(xcoords, medvhigh,'g-','LineWidth',1.75)
plot(xcoords, medhigh,'r-','LineWidth',1.75)
plot(xcoords, medmhigh,'c-','LineWidth',1.75)
plot(xcoords, medmed,'m-','LineWidth',1.75)
plot(xcoords, medmlow,'y-','LineWidth',1.75)
plot(xcoords, medlow,'k-','LineWidth',1.75)
xlabel('Evolutionary Steps', 'FontSize', 36, 'interpreter', 'latex')
ylabel('Proportion Intransitivity', 'FontSize', 36, 'interpreter', 'latex')
ylim([-0.1,1.1])
title('Intransitivity Step by Step (chicken, varying genetic drift)', 'FontSize', 28, 'interpreter', 'latex')
grid on
