payout = cell(2,2);
payout{1,1} = [3,3];
payout{1,2} = [0,2];
payout{2,1} = [2,0];
payout{2,2} = [1,1];

tic
for i=1:100
    strat = moran(payout,0.5,0.25,24);
    disp(strat);
end
toc;