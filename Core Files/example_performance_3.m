function f = example_performance_3(x,y,alpha,linear_amplitude,phase)

%% get dimensions
[~,n] = size(alpha);

%% loop over traits
advantages = (x - y)';
p = advantages*linear_amplitude;
for frequency = 1:n
    p = p + (alpha(:,frequency)/frequency^2).*(sin(2*pi*frequency*x' - phase(:,frequency)).*cos(2*pi*frequency*y' - phase(:,frequency))...
        - sin(2*pi*frequency*y' - phase(:,frequency)).*cos(2*pi*frequency*x' - phase(:,frequency)));
end

%% get performance
f = sum(p); %automatically fair since sin is an odd function, is a sum of sines in each advantage

end