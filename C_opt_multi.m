function [ C,pi ] = C_opt_multi(alpha_h,alpha_l,L,l0,X,count,N,q_ll)

% Created by Li Hongbo on 13th July, 2022
% Last modification on 13th July, 2022

%% System parameter definition
rho=0.99; % discount factor

alpha=0.99; % proper parameters, l* versus xm
% alpha_h=2;alpha_l=0;
% q_ll=0.7;

q_hh=0.9999;

delta_l=1;
p_H=0.8; p_L=0.2;
% N=3;

%% Calculation of C*
lm=min(L);
m=find(L==min(L));
m=min(m);

P_B=(1-X(m))*p_L+X(m)*p_H; % observation state probability
P_G=(1-X(m))*(1-p_L)+X(m)*(1-p_H);

X_G=X;X_B=X;X_exploit=X;
X_G(m)=X(m)*(1-p_H)/(X(m)*(1-p_H)+(1-X(m))*(1-p_L)); % posterier probability
X_B(m)=X(m)*p_H/(X(m)*p_H+(1-X(m))*p_L);

for j=1:N
    X_G(j)=X_G(j)*q_hh+(1-X_G(j))*(1-q_ll);
    X_B(j)=X_B(j)*q_hh+(1-X_B(j))*(1-q_ll);
    X_exploit(j)=X(j)*q_hh+(1-X(j))*(1-q_ll);
end

for j=1:N %update L for both choice
    alpha_opt(j)=X(j)*alpha_h+(1-X(j))*alpha_l;
    L_explore(j)=L(j)*alpha_opt(j);
end
L=L_explore;
L_explore(m)=L_explore(m)+delta_l;

if count>0
    [C1,pi1]=C_opt_multi(alpha_h,alpha_l,L_explore,l0*alpha,X_G,count-1,N,q_ll);
    [C2,pi2]=C_opt_multi(alpha_h,alpha_l,L_explore,l0*alpha,X_B,count-1,N,q_ll);
%     if N==2
%         P_G
%         C1
%         P_B
%         C2
%     end
    Qi=lm+rho*P_G*C1+rho*P_B*C2;
    Q0=l0+rho*C_opt_multi(alpha_h,alpha_l,L,l0*alpha+delta_l,X_exploit,count-1,N,q_ll);
else
    Qi=lm;
    Q0=l0;
end

if Qi<=Q0
    C=Qi; pi=1;
else
    C=Q0; pi=0;
end
