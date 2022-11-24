function [ IR_m,IR_info ] = mechanism_comparison_N1(alpha_h)

% Created by Li Hongbo on 22th November, 2022
% Last modification on 22th November, 2022

%% System parameter definition

alpha=0.99; 
alpha_l=0;
q_hh=0.9999;


bar_alpha=0.5; % Worst IR parameters
% bar_alpha=0.1;
q_ll=(1-2*bar_alpha+q_hh*bar_alpha)/(1-bar_alpha);
q_ll=0.99999999;
% bar_alpha=(alpha-alpha_l)/(alpha_h-alpha_l)

delta_l=1;
p_H=0.8; p_L=0.2;

%% Actual system dynamic for N=1
for q=1:20
%     q
    % Initialization of parameters
    alpha_m=[];
    alpha_opt=[];
    alpha_info=[];
    l0_m1=100;
    l0_opt1=100;
    l0_info1=100;
    C_m1(q)=0;C_opt1(q)=0;C_info1(q)=0;

    L_m1=[105];
    L_opt1=[105];
    L_info1=[105];
    X_m1=[0.5];
    X_opt1=[0.5];
    X_info1(:,1)=[0.5];
    for i=1:25
        % myopic decision
        lm_m=min(L_m1);
        m=find(L_m1==min(L_m1));
        %     lm_m
        %     l0_m
        if lm_m<l0_m1
%                     1
            C_m1(q)=C_m1(q)+lm_m;
            P_B=(1-X_m1(m))*p_L+X_m1(m)*p_H; % observation state probability
            P_G=(1-X_m1(m))*(1-p_L)+X_m1(m)*(1-p_H);
            y_m=randsrc(1,1,[0,1;P_G,P_B]); % observation state

            for j=1:1 % dynamics of L
                alpha_m(j)=X_m1(j)*alpha_h+(1-X_m1(j))*alpha_l;
            end
            l0_m1=l0_m1*alpha;
            if y_m==1 % posterior probability
                alpha_m(m)=alpha_h;
                X_m1(m)=X_m1(m)*(1-p_H)/(X_m1(m)*(1-p_H)+(1-X_m1(m))*(1-p_L));
            else
                alpha_m(m)=alpha_l;
                X_m1(m)=X_m1(m)*p_H/(X_m1(m)*p_H+(1-X_m1(m))*p_L);
            end
            for j=1:1 % probability transition
                L_m1(j)=L_m1(j)*alpha_m(j);
                X_m1(j)=X_m1(j)*q_hh+(1-X_m1(j))*(1-q_ll);
            end
            L_m1(m)=L_m1(m)+delta_l;
        else
%                     i
            C_m1(q)=C_m1(q)+l0_m1;
            for j=1:1
                alpha_m(j)=X_m1(j)*alpha_h+(1-X_m1(j))*alpha_l;
                X_m1(j)=X_m1(j)*q_hh+(1-X_m1(j))*(1-q_ll);
                L_m1(j)=L_m1(j)*alpha_m(j);
            end
            l0_m1=l0_m1*alpha+delta_l;
        end

        % optimal decision
        lm_opt=min(L_opt1);
        m=find(L_opt1==min(L_opt1));
        [C_1,pi]=C_opt_multi(alpha_h,alpha_l,L_opt1,l0_opt1,X_opt1,6,1,q_ll);
        %     lm_opt
        %     l0_opt
%             pi
        if pi==1
%             i
%          if l0_opt1<L_opt1(m)
            C_opt1(q)=C_opt1(q)+lm_opt;
            P_B=(1-X_opt1(m))*p_L+X_opt1(m)*p_H; % observation state probability
            P_G=(1-X_opt1(m))*(1-p_L)+X_opt1(m)*(1-p_H);
            y_opt=randsrc(1,1,[0,1;P_G,P_B]); % observation state
            for j=1:1 % dynamics of alpha
                alpha_opt(j)=X_opt1(j)*alpha_h+(1-X_opt1(j))*alpha_l;
            end
            l0_opt1=l0_opt1*alpha;
            if y_opt==1 % posterior probability
                X_opt1(m)=X_opt1(m)*(1-p_H)/(X_opt1(m)*(1-p_H)+(1-X_opt1(m))*(1-p_L));
                alpha_opt(m)=alpha_h*X_opt1(m);
            else
                X_opt1(m)=X_opt1(m)*p_H/(X_opt1(m)*p_H+(1-X_opt1(m))*p_L);
                alpha_opt(m)=alpha_l*X_opt1(m);
            end
            for j=1:1 % probability transition
                X_opt1(j)=X_opt1(j)*q_hh+(1-X_opt1(j))*(1-q_ll);
                L_opt1(j)=L_opt1(j)*alpha_opt(j);
            end
            L_opt1(m)=L_opt1(m)+delta_l;
        else
            C_opt1(q)=C_opt1(q)+l0_opt1;
            for j=1:1
                alpha_opt(j)=X_opt1(j)*alpha_h+(1-X_opt1(j))*alpha_l;
                X_opt1(j)=X_opt1(j)*q_hh+(1-X_opt1(j))*(1-q_ll);
                L_opt1(j)=L_opt1(j)*alpha_opt(j);
            end
            l0_opt1=l0_opt1*alpha+delta_l;
        end
        L_opt1;
        X_opt1;

        % selective information disclosure decision
        x_bar=(1-q_ll)/(2-q_ll-q_hh);
        lm_info=min(L_info1);
        m=find(L_info1==min(L_info1));
%             X_info
%             X_info(:,i)
        [C_1,pi]=C_opt_multi(alpha_h,alpha_l,L_info1,l0_info1,X_info1(:,i),6,1,q_ll);
        if x_bar<=(alpha-alpha_l)/(alpha_h-alpha_l)
            info_pi=pi;
        else
%                 if X_info(m,i-1)>=(alpha-alpha_l)/(alpha_h-alpha_l)
%                     info_pi=pi;
%                 elseif pi==1
%                     info_pi=1;
%                 else
%                     info_pi=0;
%                 end
            if lm_info>l0_info1
                info_pi=0;
            else
                info_pi=1;
            end
        end
        if info_pi==1
            C_info1(q)=C_info1(q)+lm_info;
            P_B=(1-X_info1(m,i))*p_L+X_info1(m,i)*p_H; % observation state probability
            P_G=(1-X_info1(m,i))*(1-p_L)+X_info1(m,i)*(1-p_H);
            y_info=randsrc(1,1,[0,1;P_G,P_B]); % observation state
            for j=1:1 % dynamics of L
                alpha_info(j)=X_info1(j,i)*alpha_h+(1-X_info1(j,i))*alpha_l;
            end
            l0_info1=l0_info1*alpha;
            if y_info==1 % posterior probability
                X_info1(m,i)=X_info1(m,i)*(1-p_H)/(X_info1(m,i)*(1-p_H)+(1-X_info1(m,i))*(1-p_L));
                alpha_info(m)=alpha_h*X_info1(m,i);
            else
                X_info1(m,i)=X_info1(m,i)*p_H/(X_info1(m,i)*p_H+(1-X_info1(m,i))*p_L);
                alpha_info(m)=alpha_l*X_info1(m,i);
            end
            for j=1:1 % probability transition
                X_info1(j,i+1)=X_info1(j,i)*q_hh+(1-X_info1(j,i))*(1-q_ll);
                L_info1(j)=L_info1(j)*alpha_info(j);
            end
            L_info1(m)=L_info1(m)+delta_l;
        else
            C_info1(q)=C_info1(q)+l0_info1;
            for j=1:1
                alpha_info(j)=X_info1(j,i)*alpha_h+(1-X_info1(j,i))*alpha_l;
                X_info1(j,i+1)=X_info1(j,i)*q_hh+(1-X_info1(j,i))*(1-q_ll);
                L_info1(j)=L_info1(j)*alpha_info(j);
            end
            l0_info1=l0_info1*alpha+delta_l;
        end

    end
    C_m1;
    C_opt1;
    C_info1;
end
med_C_m1=mean(C_m1,2);
med_C_opt1=mean(C_opt1,2);
med_C_info1=mean(C_info1,2);

median1=[];

median1(1)=med_C_opt1;
median1(2)=med_C_info1;
median1=sort(median1,'descend');
med_C_opt1=median1(2);
med_C_info1=median1(1);


IR_m1=med_C_m1/med_C_opt1;
IR_info11=med_C_info1/med_C_opt1;

%% Actual system dynamic for N=2
for q=1:20
%     q
    % Initialization of parameters
    alpha_m=[];
    alpha_opt=[];
    alpha_info=[];
    l0_m2=100;
    l0_opt2=100;
    l0_info2=100;
    C_m2(q)=0;C_opt2(q)=0;C_info2(q)=0;

    L_m2=[];
    L_opt2=[];
    L_info2=[];
    X_m2=[];
    X_opt2=[];
    X_info2(:,1)=[0];
    
    for j=1:2
        L_m2(j)=103+j;
        L_opt2(j)=103+j;
        L_info2(j)=103+j;
        X_m2(j)=0.5;
        X_opt2(j)=0.5;
        X_info2(j,1)=0.5;
    end
    for i=1:25
        % myopic decision
        lm_m=min(L_m2);
        m=find(L_m2==min(L_m2));
        %     lm_m
        %     l0_m
        if lm_m<l0_m2
%                     1
            C_m2(q)=C_m2(q)+lm_m;
            P_B=(1-X_m2(m))*p_L+X_m2(m)*p_H; % observation state probability
            P_G=(1-X_m2(m))*(1-p_L)+X_m2(m)*(1-p_H);
            y_m=randsrc(1,1,[0,1;P_G,P_B]); % observation state

            for j=1:2 % dynamics of L
                alpha_m(j)=X_m2(j)*alpha_h+(1-X_m2(j))*alpha_l;
            end
            l0_m2=l0_m2*alpha;
            if y_m==1 % posterior probability
                alpha_m(m)=alpha_h;
                X_m2(m)=X_m2(m)*(1-p_H)/(X_m2(m)*(1-p_H)+(1-X_m2(m))*(1-p_L));
            else
                alpha_m(m)=alpha_l;
                X_m2(m)=X_m2(m)*p_H/(X_m2(m)*p_H+(1-X_m2(m))*p_L);
            end
            for j=1:2 % probability transition
                L_m2(j)=L_m2(j)*alpha_m(j);
                X_m2(j)=X_m2(j)*q_hh+(1-X_m2(j))*(1-q_ll);
            end
            L_m2(m)=L_m2(m)+delta_l;
        else
%                     0
            C_m2(q)=C_m2(q)+l0_m2;
            for j=1:2
                alpha_m(j)=X_m2(j)*alpha_h+(1-X_m2(j))*alpha_l;
                X_m2(j)=X_m2(j)*q_hh+(1-X_m2(j))*(1-q_ll);
                L_m2(j)=L_m2(j)*alpha_m(j);
            end
            l0_m2=l0_m2*alpha+delta_l;
        end

        % optimal decision
        lm_opt=min(L_opt2);
        m=find(L_opt2==min(L_opt2));
        [C_1,pi]=C_opt_multi(alpha_h,alpha_l,L_opt2,l0_opt2,X_opt2,6,2,q_ll);
        %     lm_opt
        %     l0_opt
%             pi
        if pi==1
%             i
%          if l0_opt2<L_opt2(m)
            C_opt2(q)=C_opt2(q)+lm_opt;
%             X_opt2(m);
            P_B=(1-X_opt2(m))*p_L+X_opt2(m)*p_H; % observation state probability
            P_G=(1-X_opt2(m))*(1-p_L)+X_opt2(m)*(1-p_H);
            y_opt=randsrc(1,1,[0,1;P_G,P_B]); % observation state
            for j=1:2 % dynamics of alpha
                alpha_opt(j)=X_opt2(j)*alpha_h+(1-X_opt2(j))*alpha_l;
            end
            l0_opt2=l0_opt2*alpha;
            if y_opt==1 % posterior probability
                X_opt2(m)=X_opt2(m)*(1-p_H)/(X_opt2(m)*(1-p_H)+(1-X_opt2(m))*(1-p_L));
                alpha_opt(m)=alpha_h*X_opt2(m);
            else
                X_opt2(m)=X_opt2(m)*p_H/(X_opt2(m)*p_H+(1-X_opt2(m))*p_L);
                alpha_opt(m)=alpha_l*X_opt2(m);
            end
            for j=1:2 % probability transition
                X_opt2(j)=X_opt2(j)*q_hh+(1-X_opt2(j))*(1-q_ll);
                L_opt2(j)=L_opt2(j)*alpha_opt(j);
            end
            L_opt2(m)=L_opt2(m)+delta_l;
        else
            C_opt2(q)=C_opt2(q)+l0_opt2;
            for j=1:2
                alpha_opt(j)=X_opt2(j)*alpha_h+(1-X_opt2(j))*alpha_l;
                X_opt2(j)=X_opt2(j)*q_hh+(1-X_opt2(j))*(1-q_ll);
                L_opt2(j)=L_opt2(j)*alpha_opt(j);
            end
            l0_opt2=l0_opt2*alpha+delta_l;
        end
        L_opt2;
        X_opt2;

        % selective information disclosure decision
        x_bar=(1-q_ll)/(2-q_ll-q_hh);
        lm_info=min(L_info2);
        m=find(L_info2==min(L_info2));
%             X_info
%             X_info(:,i)
        [C_1,pi]=C_opt_multi(alpha_h,alpha_l,L_info2,l0_info2,X_info2(:,i),6,2,q_ll);
        if x_bar<=(alpha-alpha_l)/(alpha_h-alpha_l)
            info_pi=pi;
        else
%                 if X_info(m,i-1)>=(alpha-alpha_l)/(alpha_h-alpha_l)
%                     info_pi=pi;
%                 elseif pi==1
%                     info_pi=1;
%                 else
%                     info_pi=0;
%                 end
            if lm_info>l0_info2
                info_pi=0;
            else
                info_pi=1;
            end
        end
        if info_pi==1
            C_info2(q)=C_info2(q)+lm_info;
            P_B=(1-X_info2(m,i))*p_L+X_info2(m,i)*p_H; % observation state probability
            P_G=(1-X_info2(m,i))*(1-p_L)+X_info2(m,i)*(1-p_H);
            y_info=randsrc(1,1,[0,1;P_G,P_B]); % observation state
            for j=1:2 % dynamics of L
                alpha_info(j)=X_info2(j,i)*alpha_h+(1-X_info2(j,i))*alpha_l;
            end
            l0_info2=l0_info2*alpha;
            if y_info==1 % posterior probability
                X_info2(m,i)=X_info2(m,i)*(1-p_H)/(X_info2(m,i)*(1-p_H)+(1-X_info2(m,i))*(1-p_L));
                alpha_info(m)=alpha_h*X_info2(m,i);
            else
                X_info2(m,i)=X_info2(m,i)*p_H/(X_info2(m,i)*p_H+(1-X_info2(m,i))*p_L);
                alpha_info(m)=alpha_l*X_info2(m,i);
            end
            for j=1:2 % probability transition
                X_info2(j,i+1)=X_info2(j,i)*q_hh+(1-X_info2(j,i))*(1-q_ll);
                L_info2(j)=L_info2(j)*alpha_info(j);
            end
            L_info2(m)=L_info2(m)+delta_l;
        else
            C_info2(q)=C_info2(q)+l0_info2;
            for j=1:2
                alpha_info(j)=X_info2(j,i)*alpha_h+(1-X_info2(j,i))*alpha_l;
                X_info2(j,i+1)=X_info2(j,i)*q_hh+(1-X_info2(j,i))*(1-q_ll);
                L_info2(j)=L_info2(j)*alpha_info(j);
            end
            l0_info2=l0_info2*alpha+delta_l;
        end

    end
%     C_m2
%     C_opt2
%     C_info2
end
med_C_m2=mean(C_m2,2);
med_C_opt2=mean(C_opt2,2);
med_C_info2=mean(C_info2,2);

median2=[];

median2(1)=med_C_opt2;
median2(2)=med_C_info2;
median2=sort(median2,'descend');
med_C_opt2=median2(2);
med_C_info2=median2(1);


IR_m2=med_C_m2/med_C_opt2;
IR_info12=med_C_info2/med_C_opt2;


%% Actual system dynamic for N=3
for q=1:20
%     q
    % Initialization of parameters
    alpha_m=[];
    alpha_opt=[];
    alpha_info=[];
    l0_m3=100;
    l0_opt3=100;
    l0_info3=100;
    C_m3(q)=0;C_opt3(q)=0;C_info3(q)=0;

    L_m3=[];
    L_opt3=[];
    L_info3=[];
    X_m3=[];
    X_opt3=[];
    X_info3(:,1)=[0];
    
    for j=1:3
        L_m3(j)=102+j;
        L_opt3(j)=102+j;
        L_info3(j)=102+j;
        X_m3(j)=0.5;
        X_opt3(j)=0.5;
        X_info3(j,1)=0.5;
    end
    for i=1:25
        % myopic decision
        lm_m=min(L_m3);
        m=find(L_m3==min(L_m3));
        %     lm_m
        %     l0_m
        if lm_m<l0_m3
%                     1
            C_m3(q)=C_m3(q)+lm_m;
            P_B=(1-X_m3(m))*p_L+X_m3(m)*p_H; % observation state probability
            P_G=(1-X_m3(m))*(1-p_L)+X_m3(m)*(1-p_H);
            y_m=randsrc(1,1,[0,1;P_G,P_B]); % observation state

            for j=1:3 % dynamics of L
                alpha_m(j)=X_m3(j)*alpha_h+(1-X_m3(j))*alpha_l;
            end
            l0_m3=l0_m3*alpha;
            if y_m==1 % posterior probability
                alpha_m(m)=alpha_h;
                X_m3(m)=X_m3(m)*(1-p_H)/(X_m3(m)*(1-p_H)+(1-X_m3(m))*(1-p_L));
            else
                alpha_m(m)=alpha_l;
                X_m3(m)=X_m3(m)*p_H/(X_m3(m)*p_H+(1-X_m3(m))*p_L);
            end
            for j=1:3 % probability transition
                L_m3(j)=L_m3(j)*alpha_m(j);
                X_m3(j)=X_m3(j)*q_hh+(1-X_m3(j))*(1-q_ll);
            end
            L_m3(m)=L_m3(m)+delta_l;
        else
%                     0
            C_m3(q)=C_m3(q)+l0_m3;
            for j=1:3
                alpha_m(j)=X_m3(j)*alpha_h+(1-X_m3(j))*alpha_l;
                X_m3(j)=X_m3(j)*q_hh+(1-X_m3(j))*(1-q_ll);
                L_m3(j)=L_m3(j)*alpha_m(j);
            end
            l0_m3=l0_m3*alpha+delta_l;
        end

        % optimal decision
        lm_opt=min(L_opt3);
        m=find(L_opt3==min(L_opt3));
        [C_1,pi]=C_opt_multi(alpha_h,alpha_l,L_opt3,l0_opt3,X_opt3,6,3,q_ll);
        %     lm_opt
        %     l0_opt
%             pi
        if pi==1
%             i
%          if l0_opt3<L_opt3(m)
            C_opt3(q)=C_opt3(q)+lm_opt;
%             X_opt3(m);
            P_B=(1-X_opt3(m))*p_L+X_opt3(m)*p_H; % observation state probability
            P_G=(1-X_opt3(m))*(1-p_L)+X_opt3(m)*(1-p_H);
            y_opt=randsrc(1,1,[0,1;P_G,P_B]); % observation state
            for j=1:3 % dynamics of alpha
                alpha_opt(j)=X_opt3(j)*alpha_h+(1-X_opt3(j))*alpha_l;
            end
            l0_opt3=l0_opt3*alpha;
            if y_opt==1 % posterior probability
                X_opt3(m)=X_opt3(m)*(1-p_H)/(X_opt3(m)*(1-p_H)+(1-X_opt3(m))*(1-p_L));
                alpha_opt(m)=alpha_h*X_opt3(m);
            else
                X_opt3(m)=X_opt3(m)*p_H/(X_opt3(m)*p_H+(1-X_opt3(m))*p_L);
                alpha_opt(m)=alpha_l*X_opt3(m);
            end
            for j=1:3 % probability transition
                X_opt3(j)=X_opt3(j)*q_hh+(1-X_opt3(j))*(1-q_ll);
                L_opt3(j)=L_opt3(j)*alpha_opt(j);
            end
            L_opt3(m)=L_opt3(m)+delta_l;
        else
            C_opt3(q)=C_opt3(q)+l0_opt3;
            for j=1:3
                alpha_opt(j)=X_opt3(j)*alpha_h+(1-X_opt3(j))*alpha_l;
                X_opt3(j)=X_opt3(j)*q_hh+(1-X_opt3(j))*(1-q_ll);
                L_opt3(j)=L_opt3(j)*alpha_opt(j);
            end
            l0_opt3=l0_opt3*alpha+delta_l;
        end
        L_opt3;
        X_opt3;

        % selective information disclosure decision
        x_bar=(1-q_ll)/(2-q_ll-q_hh);
        lm_info=min(L_info3);
        m=find(L_info3==min(L_info3));
%             X_info
%             X_info(:,i)
        [C_1,pi]=C_opt_multi(alpha_h,alpha_l,L_info3,l0_info3,X_info3(:,i),6,3,q_ll);
        if x_bar<=(alpha-alpha_l)/(alpha_h-alpha_l)
            info_pi=pi;
        else
%                 if X_info(m,i-1)>=(alpha-alpha_l)/(alpha_h-alpha_l)
%                     info_pi=pi;
%                 elseif pi==1
%                     info_pi=1;
%                 else
%                     info_pi=0;
%                 end
            if lm_info>l0_info3
                info_pi=0;
            else
                info_pi=1;
            end
        end
        if info_pi==1
            C_info3(q)=C_info3(q)+lm_info;
            P_B=(1-X_info3(m,i))*p_L+X_info3(m,i)*p_H; % observation state probability
            P_G=(1-X_info3(m,i))*(1-p_L)+X_info3(m,i)*(1-p_H);
            y_info=randsrc(1,1,[0,1;P_G,P_B]); % observation state
            for j=1:3 % dynamics of L
                alpha_info(j)=X_info3(j,i)*alpha_h+(1-X_info3(j,i))*alpha_l;
            end
            l0_info3=l0_info3*alpha;
            if y_info==1 % posterior probability
                X_info3(m,i)=X_info3(m,i)*(1-p_H)/(X_info3(m,i)*(1-p_H)+(1-X_info3(m,i))*(1-p_L));
                alpha_info(m)=alpha_h*X_info3(m,i);
            else
                X_info3(m,i)=X_info3(m,i)*p_H/(X_info3(m,i)*p_H+(1-X_info3(m,i))*p_L);
                alpha_info(m)=alpha_l*X_info3(m,i);
            end
            for j=1:3 % probability transition
                X_info3(j,i+1)=X_info3(j,i)*q_hh+(1-X_info3(j,i))*(1-q_ll);
                L_info3(j)=L_info3(j)*alpha_info(j);
            end
            L_info3(m)=L_info3(m)+delta_l;
        else
            C_info3(q)=C_info3(q)+l0_info3;
            for j=1:3
                alpha_info(j)=X_info3(j,i)*alpha_h+(1-X_info3(j,i))*alpha_l;
                X_info3(j,i+1)=X_info3(j,i)*q_hh+(1-X_info3(j,i))*(1-q_ll);
                L_info3(j)=L_info3(j)*alpha_info(j);
            end
            l0_info3=l0_info3*alpha+delta_l;
        end

    end
%     C_m3
%     C_opt3
%     C_info3
end
med_C_m3=mean(C_m3,2);
med_C_opt3=mean(C_opt3,2);
med_C_info3=mean(C_info3,2);

median3=[];

median3(1)=med_C_opt3;
median3(2)=med_C_info3;
median3=sort(median3,'descend');
med_C_opt3=median3(2);
med_C_info3=median3(1);


IR_m3=med_C_m3/med_C_opt3;
IR_info13=med_C_info3/med_C_opt3;

%% Actual system dynamic for N=4
for q=1:20
%     q
    % Initialization of parameters
    alpha_m=[];
    alpha_opt=[];
    alpha_info=[];
    l0_m4=100;
    l0_opt4=100;
    l0_info4=100;
    C_m4(q)=0;C_opt4(q)=0;C_info4(q)=0;

    L_m4=[];
    L_opt4=[];
    L_info4=[];
    X_m4=[];
    X_opt4=[];
    X_info4(:,1)=[0];
    
    for j=1:4
        L_m4(j)=101+j;
        L_opt4(j)=101+j;
        L_info4(j)=101+j;
        X_m4(j)=0.5;
        X_opt4(j)=0.5;
        X_info4(j,1)=0.5;
    end
    for i=1:25
        % myopic decision
        lm_m=min(L_m4);
        m=find(L_m4==min(L_m4));
        %     lm_m
        %     l0_m
        if lm_m<l0_m4
%                     1
            C_m4(q)=C_m4(q)+lm_m;
            P_B=(1-X_m4(m))*p_L+X_m4(m)*p_H; % observation state probability
            P_G=(1-X_m4(m))*(1-p_L)+X_m4(m)*(1-p_H);
            y_m=randsrc(1,1,[0,1;P_G,P_B]); % observation state

            for j=1:4 % dynamics of L
                alpha_m(j)=X_m4(j)*alpha_h+(1-X_m4(j))*alpha_l;
            end
            l0_m4=l0_m4*alpha;
            if y_m==1 % posterior probability
                alpha_m(m)=alpha_h;
                X_m4(m)=X_m4(m)*(1-p_H)/(X_m4(m)*(1-p_H)+(1-X_m4(m))*(1-p_L));
            else
                alpha_m(m)=alpha_l;
                X_m4(m)=X_m4(m)*p_H/(X_m4(m)*p_H+(1-X_m4(m))*p_L);
            end
            for j=1:4 % probability transition
                L_m4(j)=L_m4(j)*alpha_m(j);
                X_m4(j)=X_m4(j)*q_hh+(1-X_m4(j))*(1-q_ll);
            end
            L_m4(m)=L_m4(m)+delta_l;
        else
%                     0
            C_m4(q)=C_m4(q)+l0_m4;
            for j=1:4
                alpha_m(j)=X_m4(j)*alpha_h+(1-X_m4(j))*alpha_l;
                X_m4(j)=X_m4(j)*q_hh+(1-X_m4(j))*(1-q_ll);
                L_m4(j)=L_m4(j)*alpha_m(j);
            end
            l0_m4=l0_m4*alpha+delta_l;
        end

        % optimal decision
        lm_opt=min(L_opt4);
        m=find(L_opt4==min(L_opt4));
        [C_1,pi]=C_opt_multi(alpha_h,alpha_l,L_opt4,l0_opt4,X_opt4,6,4,q_ll);
        %     lm_opt
        %     l0_opt
%             pi
        if pi==1
%             i
%          if l0_opt4<L_opt4(m)
            C_opt4(q)=C_opt4(q)+lm_opt;
%             X_opt4(m);
            P_B=(1-X_opt4(m))*p_L+X_opt4(m)*p_H; % observation state probability
            P_G=(1-X_opt4(m))*(1-p_L)+X_opt4(m)*(1-p_H);
            y_opt=randsrc(1,1,[0,1;P_G,P_B]); % observation state
            for j=1:4 % dynamics of alpha
                alpha_opt(j)=X_opt4(j)*alpha_h+(1-X_opt4(j))*alpha_l;
            end
            l0_opt4=l0_opt4*alpha;
            if y_opt==1 % posterior probability
                X_opt4(m)=X_opt4(m)*(1-p_H)/(X_opt4(m)*(1-p_H)+(1-X_opt4(m))*(1-p_L));
                alpha_opt(m)=alpha_h*X_opt4(m);
            else
                X_opt4(m)=X_opt4(m)*p_H/(X_opt4(m)*p_H+(1-X_opt4(m))*p_L);
                alpha_opt(m)=alpha_l*X_opt4(m);
            end
            for j=1:4 % probability transition
                X_opt4(j)=X_opt4(j)*q_hh+(1-X_opt4(j))*(1-q_ll);
                L_opt4(j)=L_opt4(j)*alpha_opt(j);
            end
            L_opt4(m)=L_opt4(m)+delta_l;
        else
            C_opt4(q)=C_opt4(q)+l0_opt4;
            for j=1:4
                alpha_opt(j)=X_opt4(j)*alpha_h+(1-X_opt4(j))*alpha_l;
                X_opt4(j)=X_opt4(j)*q_hh+(1-X_opt4(j))*(1-q_ll);
                L_opt4(j)=L_opt4(j)*alpha_opt(j);
            end
            l0_opt4=l0_opt4*alpha+delta_l;
        end
        L_opt4;
        X_opt4;

        % selective information disclosure decision
        x_bar=(1-q_ll)/(2-q_ll-q_hh);
        lm_info=min(L_info4);
        m=find(L_info4==min(L_info4));
%             X_info
%             X_info(:,i)
        [C_1,pi]=C_opt_multi(alpha_h,alpha_l,L_info4,l0_info4,X_info4(:,i),6,4,q_ll);
        if x_bar<=(alpha-alpha_l)/(alpha_h-alpha_l)
            info_pi=pi;
        else
%                 if X_info(m,i-1)>=(alpha-alpha_l)/(alpha_h-alpha_l)
%                     info_pi=pi;
%                 elseif pi==1
%                     info_pi=1;
%                 else
%                     info_pi=0;
%                 end
            if lm_info>l0_info4
                info_pi=0;
            else
                info_pi=1;
            end
        end
        if info_pi==1
            C_info4(q)=C_info4(q)+lm_info;
            P_B=(1-X_info4(m,i))*p_L+X_info4(m,i)*p_H; % observation state probability
            P_G=(1-X_info4(m,i))*(1-p_L)+X_info4(m,i)*(1-p_H);
            y_info=randsrc(1,1,[0,1;P_G,P_B]); % observation state
            for j=1:4 % dynamics of L
                alpha_info(j)=X_info4(j,i)*alpha_h+(1-X_info4(j,i))*alpha_l;
            end
            l0_info4=l0_info4*alpha;
            if y_info==1 % posterior probability
                X_info4(m,i)=X_info4(m,i)*(1-p_H)/(X_info4(m,i)*(1-p_H)+(1-X_info4(m,i))*(1-p_L));
                alpha_info(m)=alpha_h*X_info4(m,i);
            else
                X_info4(m,i)=X_info4(m,i)*p_H/(X_info4(m,i)*p_H+(1-X_info4(m,i))*p_L);
                alpha_info(m)=alpha_l*X_info4(m,i);
            end
            for j=1:4 % probability transition
                X_info4(j,i+1)=X_info4(j,i)*q_hh+(1-X_info4(j,i))*(1-q_ll);
                L_info4(j)=L_info4(j)*alpha_info(j);
            end
            L_info4(m)=L_info4(m)+delta_l;
        else
            C_info4(q)=C_info4(q)+l0_info4;
            for j=1:4
                alpha_info(j)=X_info4(j,i)*alpha_h+(1-X_info4(j,i))*alpha_l;
                X_info4(j,i+1)=X_info4(j,i)*q_hh+(1-X_info4(j,i))*(1-q_ll);
                L_info4(j)=L_info4(j)*alpha_info(j);
            end
            l0_info4=l0_info4*alpha+delta_l;
        end

    end
%     C_m4
%     C_opt4
%     C_info4
end
med_C_m4=mean(C_m4,2);
med_C_opt4=mean(C_opt4,2);
med_C_info4=mean(C_info4,2);

median3=[];

median3(1)=med_C_opt4;
median3(2)=med_C_info4;
median3=sort(median3,'descend');
med_C_opt4=median3(2);
med_C_info4=median3(1);


IR_m4=med_C_m4/med_C_opt4;
IR_info14=med_C_info4/med_C_opt4;

%% Actual system dynamic for N=5
for q=1:20
%     q
    % Initialization of parameters
    alpha_m=[];
    alpha_opt=[];
    alpha_info=[];
    l0_m5=100;
    l0_opt5=100;
    l0_info5=100;
    C_m5(q)=0;C_opt5(q)=0;C_info5(q)=0;

    L_m5=[];
    L_opt5=[];
    L_info5=[];
    X_m5=[];
    X_opt5=[];
    X_info5(:,1)=[0];
    
    for j=1:5
        L_m5(j)=100+j;
        L_opt5(j)=100+j;
        L_info5(j)=100+j;
        X_m5(j)=0.5;
        X_opt5(j)=0.5;
        X_info5(j,1)=0.5;
    end
    for i=1:25
        % myopic decision
        lm_m=min(L_m5);
        m=find(L_m5==min(L_m5));
        %     lm_m
        %     l0_m
        if lm_m<l0_m5
%                     1
            C_m5(q)=C_m5(q)+lm_m;
            P_B=(1-X_m5(m))*p_L+X_m5(m)*p_H; % observation state probability
            P_G=(1-X_m5(m))*(1-p_L)+X_m5(m)*(1-p_H);
            y_m=randsrc(1,1,[0,1;P_G,P_B]); % observation state

            for j=1:5 % dynamics of L
                alpha_m(j)=X_m5(j)*alpha_h+(1-X_m5(j))*alpha_l;
            end
            l0_m5=l0_m5*alpha;
            if y_m==1 % posterior probability
                alpha_m(m)=alpha_h;
                X_m5(m)=X_m5(m)*(1-p_H)/(X_m5(m)*(1-p_H)+(1-X_m5(m))*(1-p_L));
            else
                alpha_m(m)=alpha_l;
                X_m5(m)=X_m5(m)*p_H/(X_m5(m)*p_H+(1-X_m5(m))*p_L);
            end
            for j=1:5 % probability transition
                L_m5(j)=L_m5(j)*alpha_m(j);
                X_m5(j)=X_m5(j)*q_hh+(1-X_m5(j))*(1-q_ll);
            end
            L_m5(m)=L_m5(m)+delta_l;
        else
%                     0
            C_m5(q)=C_m5(q)+l0_m5;
            for j=1:5
                alpha_m(j)=X_m5(j)*alpha_h+(1-X_m5(j))*alpha_l;
                X_m5(j)=X_m5(j)*q_hh+(1-X_m5(j))*(1-q_ll);
                L_m5(j)=L_m5(j)*alpha_m(j);
            end
            l0_m5=l0_m5*alpha+delta_l;
        end

        % optimal decision
        lm_opt=min(L_opt5);
        m=find(L_opt5==min(L_opt5));
        [C_1,pi]=C_opt_multi(alpha_h,alpha_l,L_opt5,l0_opt5,X_opt5,6,5,q_ll);
        %     lm_opt
        %     l0_opt
%             pi
        if pi==1
%             i
%          if l0_opt5<L_opt5(m)
            C_opt5(q)=C_opt5(q)+lm_opt;
%             X_opt5(m);
            P_B=(1-X_opt5(m))*p_L+X_opt5(m)*p_H; % observation state probability
            P_G=(1-X_opt5(m))*(1-p_L)+X_opt5(m)*(1-p_H);
            y_opt=randsrc(1,1,[0,1;P_G,P_B]); % observation state
            for j=1:5 % dynamics of alpha
                alpha_opt(j)=X_opt5(j)*alpha_h+(1-X_opt5(j))*alpha_l;
            end
            l0_opt5=l0_opt5*alpha;
            if y_opt==1 % posterior probability
                X_opt5(m)=X_opt5(m)*(1-p_H)/(X_opt5(m)*(1-p_H)+(1-X_opt5(m))*(1-p_L));
                alpha_opt(m)=alpha_h*X_opt5(m);
            else
                X_opt5(m)=X_opt5(m)*p_H/(X_opt5(m)*p_H+(1-X_opt5(m))*p_L);
                alpha_opt(m)=alpha_l*X_opt5(m);
            end
            for j=1:5 % probability transition
                X_opt5(j)=X_opt5(j)*q_hh+(1-X_opt5(j))*(1-q_ll);
                L_opt5(j)=L_opt5(j)*alpha_opt(j);
            end
            L_opt5(m)=L_opt5(m)+delta_l;
        else
            C_opt5(q)=C_opt5(q)+l0_opt5;
            for j=1:5
                alpha_opt(j)=X_opt5(j)*alpha_h+(1-X_opt5(j))*alpha_l;
                X_opt5(j)=X_opt5(j)*q_hh+(1-X_opt5(j))*(1-q_ll);
                L_opt5(j)=L_opt5(j)*alpha_opt(j);
            end
            l0_opt5=l0_opt5*alpha+delta_l;
        end
        L_opt5;
        X_opt5;

        % selective information disclosure decision
        x_bar=(1-q_ll)/(2-q_ll-q_hh);
        lm_info=min(L_info5);
        m=find(L_info5==min(L_info5));
%             X_info
%             X_info(:,i)
        [C_1,pi]=C_opt_multi(alpha_h,alpha_l,L_info5,l0_info5,X_info5(:,i),6,5,q_ll);
        if x_bar<=(alpha-alpha_l)/(alpha_h-alpha_l)
            info_pi=pi;
        else
%                 if X_info(m,i-1)>=(alpha-alpha_l)/(alpha_h-alpha_l)
%                     info_pi=pi;
%                 elseif pi==1
%                     info_pi=1;
%                 else
%                     info_pi=0;
%                 end
            if lm_info>l0_info5
                info_pi=0;
            else
                info_pi=1;
            end
        end
        if info_pi==1
            C_info5(q)=C_info5(q)+lm_info;
            P_B=(1-X_info5(m,i))*p_L+X_info5(m,i)*p_H; % observation state probability
            P_G=(1-X_info5(m,i))*(1-p_L)+X_info5(m,i)*(1-p_H);
            y_info=randsrc(1,1,[0,1;P_G,P_B]); % observation state
            for j=1:5 % dynamics of L
                alpha_info(j)=X_info5(j,i)*alpha_h+(1-X_info5(j,i))*alpha_l;
            end
            l0_info5=l0_info5*alpha;
            if y_info==1 % posterior probability
                X_info5(m,i)=X_info5(m,i)*(1-p_H)/(X_info5(m,i)*(1-p_H)+(1-X_info5(m,i))*(1-p_L));
                alpha_info(m)=alpha_h*X_info5(m,i);
            else
                X_info5(m,i)=X_info5(m,i)*p_H/(X_info5(m,i)*p_H+(1-X_info5(m,i))*p_L);
                alpha_info(m)=alpha_l*X_info5(m,i);
            end
            for j=1:5 % probability transition
                X_info5(j,i+1)=X_info5(j,i)*q_hh+(1-X_info5(j,i))*(1-q_ll);
                L_info5(j)=L_info5(j)*alpha_info(j);
            end
            L_info5(m)=L_info5(m)+delta_l;
        else
            C_info5(q)=C_info5(q)+l0_info5;
            for j=1:5
                alpha_info(j)=X_info5(j,i)*alpha_h+(1-X_info5(j,i))*alpha_l;
                X_info5(j,i+1)=X_info5(j,i)*q_hh+(1-X_info5(j,i))*(1-q_ll);
                L_info5(j)=L_info5(j)*alpha_info(j);
            end
            l0_info5=l0_info5*alpha+delta_l;
        end

    end
%     C_m5
%     C_opt5
%     C_info5
end
med_C_m5=mean(C_m5,2);
med_C_opt5=mean(C_opt5,2);
med_C_info5=mean(C_info5,2);

median5=[];

median5(1)=med_C_opt5;
median5(2)=med_C_info5;
median5=sort(median5,'descend');
med_C_opt5=median5(2);
med_C_info5=median5(1);


IR_m5=med_C_m5/med_C_opt5;
IR_info15=med_C_info5/med_C_opt5;

IR_m=[IR_m2 IR_m3 IR_m4 IR_m5];
IR_info=[IR_info12 IR_info13 IR_info14 IR_info15];
% 
% % Note that the following two sets are obtained by precalculating alpha_h=10 in line 16
% IR_m=[5.1203 4.9932 4.8002 4.6271];
% IR_info=[1.1662 1.1638 1.1555 1.1209]; 
% 
% figure % marker size: 16; font size: 24; legend font size: 20;
% plot(kx(:),IR_m(:),'-bo','linewidth',1);hold on;
% plot(kx(:),IR_m_1(:),'-go','linewidth',1);hold on;
% plot(kx(:),IR_info1(:),'-.r*','linewidth',1);hold on;
% plot(kx(:),IR_info1_1(:),'-.y*','linewidth',1);hold on;
% % xlabel('Belief state x_i(t)');
% xlabel('Stationary hazard belief $\bar{x}$','Interpreter','latex')
% ylabel('Avg inefficiency ratios','Interpreter','latex');
% legend('\gamma^{(m)} with \alpha_H=2', '\gamma^{(m)}  with \alpha_H=10','\gamma^{(SID)} with \alpha_H=2', '\gamma^{(SID)} with \alpha_H=10');
IR_m=sort(IR_m,'descend');
IR_info=sort(IR_info,'descend');

