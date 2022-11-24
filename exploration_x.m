function [ output_args ] = exploration_x( input_args )

% Created by Li Hongbo on 13th July, 2022
% Last modification on 13th July, 2022

%% System parameter definition
l0=10; % latency of path 0
% L=[4.5 12 8]; % latency set of path 1,...,N
lm=[]; % path m has the least latency
% X=[0.5 0.3 0.8]; % initial belief state set
count=10;

%% Calculation of l*
k=0;
xm=[];
for i=1:21
    i
    xm(i)=(i-1)*0.05;
    lm(i)=20.5;
    k=0;
    while k<40
        lm(i)=lm(i)-0.5;
        [C,pi]=C_opt_cal(lm(i),l0,xm(i),count);  
        pi;
        if pi==1
            k=100;
        end
        k=k+1;
    end
end
lm
l0_copy=[];
for i=1:20
   l0_copy(i)=l0;
%    lm(22-i)=lm(21-i);
   lm(i)=lm(i);
end
l0_copy(21)=l0;

%% Plot figure
figure % marker size: 16; font size: 24; legend font size: 20;
plot(xm(:),l0_copy(:),'-.r*','linewidth',1);hold on;
plot(xm(:),lm(:),'-bo','linewidth',1);hold on;
xlabel('Hazard Belief x_1(t) of path 1');
ylabel('Exploration thresholds');
legend('Myopic policy $\ell^{(m)}(t)$ in (20)', 'Optimal policy $\ell_1^*(t)$ in (21)', 'interpreter', 'latex');
