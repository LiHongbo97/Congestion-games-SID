function [ C,pi ] = C_opt_cal(lm,l0,xm,count)

% Created by Li Hongbo on 13th July, 2022
% Last modification on 13th July, 2022

%% System parameter definition
rho=0.8; % discount factor
% alpha=0.3; %under-exploration parameter, xm=0.9
% alpha_h=1.5;alpha_l=0.2;
% q_ll=0.3;q_hh=0.8;

% alpha=0.9; % over-exploration parameter, xm=0.5
% alpha_h=1.2;alpha_l=0.1;
% q_ll=0.5;q_hh=0.5;

alpha=0.6; % proper parameters, l* versus xm
alpha_h=1.2;alpha_l=0.2;
q_ll=0.5;q_hh=0.5;

delta_l=2;
p_H=0.8; p_L=0.3;

%% Calculation of C*
P_B=(1-xm)*p_L+xm*p_H; % observation state probability
P_G=(1-xm)*(1-p_L)+xm*(1-p_H);

x_G=xm*(1-p_H)/(xm*(1-p_H)+(1-xm)*(1-p_L)); % posterier probability
x_B=xm*p_H/(xm*p_H+(1-xm)*p_L);
x_0=xm;
x_G=x_G*q_hh+(1-x_G)*(1-q_ll); % state transition
x_B=x_B*q_hh+(1-x_B)*(1-q_ll);
x_0=x_0*q_hh+(1-x_0)*(1-q_ll);

E_alpha=xm*alpha_h+(1-xm)*alpha_l;

if count>0
    [C1,pi1]=C_opt_cal(lm*E_alpha+delta_l,l0*alpha,x_G,count-1);
    [C2,pi2]=C_opt_cal(lm*E_alpha+delta_l,l0*alpha,x_B,count-1);
    Qi=lm+rho*P_G*C1+rho*P_B*C2;
    Q0=l0+rho*C_opt_cal(lm*E_alpha,l0*alpha+delta_l,x_0,count-1);
else
    Qi=lm;
    Q0=l0;
end

if Qi<=Q0
    C=Qi; pi=1;
else
    C=Q0; pi=0;
end
