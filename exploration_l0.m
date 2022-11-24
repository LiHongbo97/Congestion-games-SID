function [ output_args ] = exploration_l0( input_args )

% Created by Li Hongbo on 13th July, 2022
% Last modification on 13th July, 2022

%% System parameter definition
l0=[]; % latency of path 0
% L=[4.5 12 8]; % latency set of path 1,...,N
lm=[]; % path m has the least latency
% X=[0.5 0.3 0.8]; % initial belief state set
xm=0.1;
count=10;

%% Calculation of l*
k=0;
for i=1:21
    i
    l0(i)=i-1;
    lm(i)=11+l0(i);
    k=0;
    while k<100
        lm(i)=lm(i)-0.5;
        [C,pi]=C_opt_cal(lm(i),l0(i),xm,count);  
        pi;
        if pi==1
            k=100;
        end
        k=k+1;
    end
end
% lm(1)=lm(3)-2;
% lm(2)=lm(3)-1;
lm(1)=max(lm(1),0);
%% Plot figure
figure % marker size: 16; font size: 24; legend font size: 20;
plot(l0(:),l0(:),'-.r*','linewidth',1);hold on;
plot(l0(:),lm(:),'-bo','linewidth',1);hold on;
xlabel('Travel latency of path 0');
ylabel('Exploration thresholds');
legend('Myopic policy $\ell^{(m)}(t)$ in (20)', 'Optimal policy $\ell_1^*(t)$ in (21)', 'interpreter', 'latex');
