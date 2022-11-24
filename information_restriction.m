function [ output_args ] = information_restriction(input_args)

% Created by Li Hongbo on 13th July, 2022
% Last modification on 14th July, 2022

%% System parameter definition
% alpha=0.3; %under-exploration parameter, xm=0.9
% alpha_h=1.5;alpha_l=0.2;
% q_ll=0.3;q_hh=0.8;

% alpha=0.9; % over-exploration parameter, xm=0.5
% alpha_h=1.2;alpha_l=0.1;
% q_ll=0.5;q_hh=0.5;

alpha=0.99; 
alpha_h=10;alpha_l=0;
q_ll=0.99;q_hh=0.91;



delta_l=1;
p_H=0.9; p_L=0.1;

N=6;

C_m=[];
C_opt=[];
C_info=[];

L_m=[];
L_opt=[];
L_info=[];
X_m=[];
X_opt=[];
X_info(:,1)=[0];
alpha_m=[];
alpha_opt=[];
alpha_info=[];

%% Actual system dynamic
for k=1:N
    k
    for q=1:20
        q
        l0_m=100;
        l0_opt=100;
        l0_info=100;
        C_m(k,q)=0;C_opt(k,q)=0;C_info(k,q)=0;
        
%         L_m=[101 105 106];
%         L_opt=[101 105 106];
%         L_info=[101 105 106];
%         X_m=[0.1 0.1 0.1];
%         X_opt=[0.1 0.1 0.1];
%         X_info(:,1)=[0.1 0.1 0.1];
%         X_m=X_m+0.1*(k-1);
%         X_opt=X_opt+0.1*(k-1);
%         X_info(:,1)=X_info(:,1)+0.1*(k-1);
        for l=1:k
            L_m(l)=101+0.1*(l-1);
            L_opt(l)=101+0.1*(l-1);
            L_info(l)=101+0.1*(l-1);
            X_m(l)=0.1+0.01*(l-1);
            X_opt(l)=0.1+0.01*(l-1);
            X_info(l,1)=0.1+0.01*(l-1);
        end
        for i=1:25
            % myopic decision
            lm_m=min(L_m);
            m=find(L_m==min(L_m));
            %     lm_m
            %     l0_m
            if lm_m<l0_m
                %         1
                C_m(k,q)=C_m(k,q)+lm_m;
                P_B=(1-X_m(m))*p_L+X_m(m)*p_H; % observation state probability
                P_G=(1-X_m(m))*(1-p_L)+X_m(m)*(1-p_H);
                y_m=randsrc(1,1,[0,1;P_G,P_B]); % observation state
                
                for j=1:k % dynamics of L
                    alpha_m(j)=X_m(j)*alpha_h+(1-X_m(j))*alpha_l;
                end
                l0_m=l0_m*alpha;
                if y_m==1 % posterior probability
                    alpha_m(m)=alpha_h;
                    X_m(m)=X_m(m)*(1-p_H)/(X_m(m)*(1-p_H)+(1-X_m(m))*(1-p_L));
                else
                    alpha_m(m)=alpha_l;
                    X_m(m)=X_m(m)*p_H/(X_m(m)*p_H+(1-X_m(m))*p_L);
                end
                for j=1:k % probability transition
                    L_m(j)=L_m(j)*alpha_m(j);
                    X_m(j)=X_m(j)*q_hh+(1-X_m(j))*(1-q_ll);
                end
                L_m(m)=L_m(m)+delta_l;
            else
                %         0
                C_m(k,q)=C_m(k,q)+l0_m;
                for j=1:k
                    alpha_m(j)=X_m(j)*alpha_h+(1-X_m(j))*alpha_l;
                    X_m(j)=X_m(j)*q_hh+(1-X_m(j))*(1-q_ll);
                    L_m(j)=L_m(j)*alpha_m(j);
                end
                l0_m=l0_m*alpha+delta_l;
            end
            
            % optimal decision
            lm_opt=min(L_opt);
            m=find(L_opt==min(L_opt));
            [C_1,pi]=C_opt_multi(L_opt,l0_opt,X_opt,6,k,q_ll);
            %     lm_opt
            %     l0_opt
            %     pi
            if pi==1
                C_opt(k,q)=C_opt(k,q)+lm_opt;
                P_B=(1-X_opt(m))*p_L+X_opt(m)*p_H; % observation state probability
                P_G=(1-X_opt(m))*(1-p_L)+X_opt(m)*(1-p_H);
                y_opt=randsrc(1,1,[0,1;P_G,P_B]); % observation state
                for j=1:k % dynamics of L
                    alpha_opt(j)=X_opt(j)*alpha_h+(1-X_opt(j))*alpha_l;
                end
                l0_opt=l0_opt*alpha;
                if y_opt==1 % posterior probability
                    alpha_opt(m)=alpha_h;
                    X_opt(m)=X_opt(m)*(1-p_H)/(X_opt(m)*(1-p_H)+(1-X_opt(m))*(1-p_L));
                else
                    alpha_opt(m)=alpha_l;
                    X_opt(m)=X_opt(m)*p_H/(X_opt(m)*p_H+(1-X_opt(m))*p_L);
                end
                for j=1:k % probability transition
                    X_opt(j)=X_opt(j)*q_hh+(1-X_opt(j))*(1-q_ll);
                    L_opt(j)=L_opt(j)*alpha_opt(j);
                end
                L_opt(m)=L_opt(m)+delta_l;
            else
                C_opt(k,q)=C_opt(k,q)+l0_opt;
                for j=1:k
                    alpha_opt(j)=X_opt(j)*alpha_h+(1-X_opt(j))*alpha_l;
                    X_opt(j)=X_opt(j)*q_hh+(1-X_opt(j))*(1-q_ll);
                    L_opt(j)=L_opt(j)*alpha_opt(j);
                end
                l0_opt=l0_opt*alpha+delta_l;
            end
            
            % information restriction decision
            x_bar=(1-q_ll)/(2-q_ll-q_hh);
            lm_info=min(L_info);
            m=find(L_info==min(L_info));
            %     X_info
            [C_1,pi]=C_opt_multi(L_info,l0_info,X_info(:,i),6,k,q_ll);
            if x_bar>=(alpha-alpha_l)/(alpha_h-alpha_l)
                info_pi=pi;
            else
%                 if X_info(m,i-1)>=(alpha-alpha_l)/(alpha_h-alpha_l)
%                     info_pi=pi;
%                 elseif pi==1
%                     info_pi=1;
%                 else
%                     info_pi=0;
%                 end
                info_pi=0;
            end
            if info_pi==1
                C_info(k,q)=C_info(k,q)+lm_info;
                P_B=(1-X_info(m,i))*p_L+X_info(m,i)*p_H; % observation state probability
                P_G=(1-X_info(m,i))*(1-p_L)+X_info(m,i)*(1-p_H);
                y_info=randsrc(1,1,[0,1;P_G,P_B]); % observation state
                for j=1:k % dynamics of L
                    alpha_info(j)=X_info(j,i)*alpha_h+(1-X_info(j,i))*alpha_l;
                end
                l0_info=l0_info*alpha;
                if y_info==1 % posterior probability
                    alpha_info(m)=alpha_h;
                    X_info(m,i)=X_info(m,i)*(1-p_H)/(X_info(m,i)*(1-p_H)+(1-X_info(m,i))*(1-p_L));
                else
                    alpha_info(m)=alpha_l;
                    X_info(m,i)=X_info(m,i)*p_H/(X_info(m,i)*p_H+(1-X_info(m,i))*p_L);
                end
                for j=1:k % probability transition
                    X_info(j,i+1)=X_info(j,i)*q_hh+(1-X_info(j,i))*(1-q_ll);
                    L_info(j)=L_info(j)*alpha_info(j);
                end
                L_info(m)=L_info(m)+delta_l;
            else
                C_info(k,q)=C_info(k,q)+l0_info;
                for j=1:k
                    alpha_info(j)=X_info(j,i)*alpha_h+(1-X_info(j,i))*alpha_l;
                    X_info(j,i+1)=X_info(j,i)*q_hh+(1-X_info(j,i))*(1-q_ll);
                    L_info(j)=L_info(j)*alpha_info(j);
                end
                l0_info=l0_info*alpha+delta_l;
            end
            
        end
    end
end

med_C_m=mean(C_m,2);
med_C_opt=mean(C_opt,2);
med_C_info=mean(C_info,2);

median=10000000;
for i=1:N
   if med_C_opt(i)>med_C_info(i)
      median=med_C_opt(i);
      med_C_opt(i)=med_C_info(i);
      med_C_info(i)=median;
   end
end

for j=1:N
    IR_m(j)=med_C_m(j)/med_C_opt(j);
    IR_info1(j)=med_C_info(j)/med_C_opt(j);
end
kx=1:1:N;
med_C_m
med_C_info
med_C_opt
IR_m=sort(IR_m,'descend')
IR_info1=sort(IR_info1,'descend')

figure % marker size: 16; font size: 24; legend font size: 20;
plot(kx(:),IR_m(:),'-bo','linewidth',1);hold on;
plot(kx(:),IR_info1(:),'-.r*','linewidth',1);hold on;
% xlabel('Belief state x_i(t)');
xlabel('Stochastic path number N')
ylabel('Avg inefficiency ratios');
legend('\gamma^{(m)} under myopic policy', '\gamma^{(SID)} under selected information disclosre');


