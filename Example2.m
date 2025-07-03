%Example2

close all
clear
clc
format long

syms t

MatrixA=[cos(t),sin(t);-sin(t),cos(t)]+1i*[sin(t),cos(t);cos(t),-sin(t)];
MatrixC=[2*(cos(t))^2-2*cos(t)*sin(t)+6*sin(t),4*cos(t)+2*cos(t)*sin(t)-2*(cos(t))^2;-2*sin(2*t)-6*cos(t)+2,2*sin(2*t)-4*sin(t)-2]+1i*[2*(cos(t))^2+2*cos(t)*sin(t)+6*sin(t),4*cos(t)+2*cos(t)*sin(t)+2*(cos(t))^2;-2*sin(2*t)-6*cos(t)-2,-2*sin(2*t)-4*sin(t)-2];
MatrixF=[6+sin(t),cos(t);cos(t),4+sin(t)]+1i*[cos(t),sin(t);sin(t),cos(t)];

[cm,cn]=size(MatrixC);
MatrixI_left=eye(cm);
MatrixI_right=eye(cn);
MatrixI_kron_A=kron(MatrixI_right,MatrixA);
MatrixF_T_kron_I=kron(conj(MatrixF)',MatrixI_left);
leftoutput1=[real(MatrixF_T_kron_I)-real(MatrixI_kron_A),-imag(MatrixF_T_kron_I)-imag(MatrixI_kron_A);imag(MatrixF_T_kron_I)-imag(MatrixI_kron_A),real(MatrixF_T_kron_I)+real(MatrixI_kron_A)];
myFunctionHandlerleft1=matlabFunction(leftoutput1,'vars',t);

MatrixF_real=real(MatrixF);
MatrixF_imag=imag(MatrixF);
MatrixA_real=real(MatrixA);
MatrixA_imag=imag(MatrixA);
MatrixF_real_T_kron_I=kron(MatrixF_real',eye(cm));
MatrixF_imag_T_kron_I=kron(MatrixF_imag',eye(cm));
MatrixI_kron_A_real=kron(eye(cn),MatrixA_real);
MatrixI_kron_A_imag=kron(eye(cn),MatrixA_imag);
leftoutput2=[(MatrixF_real_T_kron_I-MatrixI_kron_A_real),-(MatrixF_imag_T_kron_I+MatrixI_kron_A_imag);MatrixF_imag_T_kron_I-MatrixI_kron_A_imag,MatrixF_real_T_kron_I+MatrixI_kron_A_real];%æÿ’ÛM
myFunctionHandlerleft2=matlabFunction(leftoutput2,'vars',t);


x11_true=sin(t)+1i*sin(t); 
x12_true=cos(t)+1i*cos(t); 
x21_true=-cos(t)+1i*-cos(t);
x22_true=-sin(t)+1i*-sin(t);

real_X=[real(x11_true);real(x21_true);real(x12_true);real(x22_true);imag(x11_true);imag(x21_true);imag(x12_true);imag(x22_true)];
real_X_func = matlabFunction(real_X, 'Vars', t);
num=1;

variable = inputdlg({'Enter gamma ¶√ (e.g., 10, 10+20i, or 10-20i): ','Enter step ¶≈ (e.g., 0.1, or 0.001): '});
gamma = str2double(variable{1});
step = str2double(variable{2});
t0=0;
tf=10;

tau=step;%Step.
tk=t0:tau:tf;%Generate iterative interval points.
tl=length(tk);%Number of iterations.

num_vars = 2*cm*cn; % Number of variables.
random_length=10;%Initial value length.

x=zeros(2*cm*cn,1,tl,num,6,2);

for k=1:num
x(:,:,1,k,:,:) = repmat(random_length*((rand(num_vars,1)-0.5*ones(num_vars,1))), 1, 1, 6, 2);
end

err=zeros(2,tl-1,num,5,2);

F=diff(MatrixF,t);
A=diff(MatrixA,t);
C=diff(MatrixC,t);

dotF=eval(F);
dotA=eval(A);
dotC=eval(C);
MatrixC_eval=eval(MatrixC);

AFMlinear = @(X) X;

tx_temp=sym('tx_temp', [2*cm*cn 1]);
tx_dou=tx_temp(1:cm*cn)+1i*tx_temp(cm*cn+1:end);
MatrixX=reshape(conj(tx_dou'),cm,cn);
disp(MatrixX);
ZnnRightHandSide=dotC+dotA*conj(MatrixX)-(MatrixX)*dotF-gamma*AFMlinear(MatrixX*MatrixF-MatrixA*conj(MatrixX)-MatrixC_eval);
ZnnRightHandSide_div=[real(ZnnRightHandSide),imag(ZnnRightHandSide)];
v=reshape(ZnnRightHandSide_div,2*cm*cn,1);
rightoutput1=v;
myFunctionHandleright1=matlabFunction(rightoutput1,'vars',{t,tx_temp});


TranLeftHandSide=diff(myFunctionHandlerleft2,t);

dotTranLeftHandSide=eval(TranLeftHandSide);
MatrixC_real=real(MatrixC);
MatrixC_imag=imag(MatrixC);
MatrixC_real_vec=reshape(MatrixC_real,cm*cn,1);
MatrixC_imag_vec=reshape(MatrixC_imag,cm*cn,1);
MatrixC_vec_group=[MatrixC_real_vec;MatrixC_imag_vec];

dotC_vec_group=diff(MatrixC_vec_group,t);
rightoutput2=dotC_vec_group-dotTranLeftHandSide*tx_temp-gamma*AFMlinear(myFunctionHandlerleft2*tx_temp-MatrixC_vec_group);
myFunctionHandleright2=matlabFunction(rightoutput2,'vars',{t,tx_temp});

model_1= 'Con-DZND1-';
model_2= 'Con-DZND2-';

gamma_is_real = isreal(gamma);

for i=1:num
    tic
    cpu=cputime;
    
    for j=1:tl-1
        t=tk(j);
        tx1_1=x(:,:,j,i,1,1);
        v1_1=myFunctionHandleright1(t,tx1_1);
        dx1_1=pinv(myFunctionHandlerleft1(t))*v1_1;%For Euler Con_DZND1-2i.
        
        if gamma_is_real,
        tx1_2=x(:,:,j,i,1,2);
        v1_2=myFunctionHandleright2(t,tx1_2);
        dx1_2=pinv(myFunctionHandlerleft2(t))*v1_2;%For Euler Con_DZND2-2i.
        end 
           
        x(:,:,j+1,i,1,1)=x(:,:,j,i,1,1)+tau*dx1_1;
        if gamma_is_real,
        x(:,:,j+1,i,1,2)=x(:,:,j,i,1,2)+tau*dx1_2;
        end
        
        if i==1,
            real_X_eval(:,:,j)=real_X_func(t);
        end
        
        %Calculation error.
        err(1,j,i,1,1)=norm(x(:,:,j,i,1,1)-real_X_eval(:,:,j),'fro');
        if gamma_is_real,
        err(1,j,i,1,2)=norm(x(:,:,j,i,1,2)-real_X_eval(:,:,j),'fro');
        end
        
        t
    end
    actpu=(cputime-cpu)/(tl-1);
    actpu1=toc/(tl-1);
end

for i = 1 : 1
    sf_1 = strcat(model_1, '%d', 'i');
    s_1 = sprintf(sf_1, 2*i);
    algrithm_names_1{i}=s_1;
    
    if gamma_is_real,
    sf_2 = strcat(model_2, '%d', 'i');
    s_2 = sprintf(sf_2, 2*i);
    algrithm_names_2{i}=s_2;
    end
    
end

methodnum = 1;

Ytt = zeros(num_vars, tl-1);
xt_1 = Ytt;
xt_2 = Ytt;


for p = 1:tl-1

    for i = 1:num_vars
        Ytt(i, p) = real_X_eval(i, 1, p);
    end
    
    for i = 1:num_vars
        xt_1(i, p) = x(i, 1, p, methodnum, 1, 1);
    end
    
    if gamma_is_real,

        for i = 1:num_vars
            xt_2(i, p) = x(i, 1, p, methodnum, 1, 2);
        end
    end
    
end

for s=1:num
    %Compare.
    f11=figure(11);
    f11.Name='Con_DZND1 single';
    f11.NumberTitle='off';
    semilogy(tk(1,1:end-1)*(1/tau),err(1,:,s,1,1),'r--','LineWidth',6);set(gca,'FontSize',16);hold on
    handle=legend(algrithm_names_1,'Position',[0.58,0.67 0.3,0.21],'FontSize',10);hold on
    set(handle,'Interpreter','latex', 'FontSize',12)
    xlabel('$k$','interpreter','latex', 'FontSize', 20);
    
    if isreal(gamma),
        f12=figure(12);
        f12.Name='Con_DZND2 single';
        f12.NumberTitle='off';
        semilogy(tk(1,1:end-1)*(1/tau),err(1,:,s,1,2),'g','LineWidth',2);set(gca,'FontSize',16);hold on
        handle=legend(algrithm_names_2,'Position',[0.58,0.67 0.3,0.21],'FontSize',10);hold on
        set(handle,'Interpreter','latex', 'FontSize',12)
        xlabel('$k$','interpreter','latex', 'FontSize', 20);
        
        f1=figure(1);
        f1.Name='Compare Con_DZND1 and Con_DZND2';
        f1.NumberTitle='off';
        semilogy(tk(1,1:end-1)*(1/tau),err(1,:,s,1,1),'r--','LineWidth',6);set(gca,'FontSize',16);hold on
        semilogy(tk(1,1:end-1)*(1/tau),err(1,:,s,1,2),'g','LineWidth',2);hold on
        handle=legend([algrithm_names_1,algrithm_names_2],'Position',[0.58,0.67 0.3,0.21],'FontSize',10);hold on
        set(handle,'Interpreter','latex', 'FontSize',12)
        xlabel('$k$','interpreter','latex', 'FontSize', 20);
    end
    
    %Setting graphical properties.
    figProperties = {
        {'X11_real_solve', Ytt(1,:), xt_1(1,:), '$y_{\mathrm{r},11}(\tau)$', '$x_{\mathrm{r},11}(\tau)$', 'r'}, ...
        {'X21_real_solve', Ytt(2,:), xt_1(2,:), '$y_{\mathrm{r},21}(\tau)$', '$x_{\mathrm{r},21}(\tau)$', 'r'}, ...
        {'X12_real_solve', Ytt(3,:), xt_1(3,:), '$y_{\mathrm{r},12}(\tau)$', '$x_{\mathrm{r},12}(\tau)$', 'r'}, ...
        {'X22_real_solve', Ytt(4,:), xt_1(4,:), '$y_{\mathrm{r},22}(\tau)$', '$x_{\mathrm{r},22}(\tau)$', 'r'}, ...
        {'X11_imaginary_solve', Ytt(5,:), xt_1(5,:), '$y_{\mathrm{i},11}(\tau)$', '$x_{\mathrm{i},11}(\tau)$', 'r'}, ...
        {'X21_imaginary_solve', Ytt(6,:), xt_1(6,:), '$y_{\mathrm{i},21}(\tau)$', '$x_{\mathrm{i},21}(\tau)$', 'r'}, ...
        {'X12_imaginary_solve', Ytt(7,:), xt_1(7,:), '$y_{\mathrm{i},12}(\tau)$', '$x_{\mathrm{i},12}(\tau)$', 'r'}, ...
        {'X22_imaginary_solve', Ytt(8,:), xt_1(8,:), '$y_{\mathrm{i},22}(\tau)$', '$x_{\mathrm{i},22}(\tau)$', 'r'}
        };
    
    %Generating graphics.
    for i = 1:length(figProperties)
        figureNumber= mod(i,cm)*1000+ceil(i/cm)*100+ceil(i/(cm*cn))*10+1;
        figure(figureNumber);
        f = gcf;
        f.Name = figProperties{i}{1};
        f.NumberTitle = 'off';
        
        plot(tk(1,1:end-1)*(1/tau), figProperties{i}{2}, '--', 'LineWidth', 6);set(gca, 'ZTick', []);view(-45,85);
        set(gca, 'FontSize', 16);
        hold on;
        plot(tk(1,1:end-1)*(1/tau), figProperties{i}{3}, figProperties{i}{6}, 'LineWidth', 2);
        
        handle = legend({figProperties{i}{4}, figProperties{i}{5}}, 'Position', [0.65, 0.7, 0.335, 0.285], 'FontSize', 40);
        set(handle, 'Interpreter', 'latex', 'FontSize', 36);
        xlabel('$k$', 'Interpreter', 'latex', 'FontSize', 20);
    end
    
    if gamma_is_real,
        %Setting graphical properties.
        figPropertiesReal = {
            {'X11_real_solve', Ytt(1,:), xt_2(1,:), '$y_{\mathrm{r},11}(\tau)$', '$x_{\mathrm{r},11}(\tau)$', 'g'}, ...
            {'X21_real_solve', Ytt(2,:), xt_2(2,:), '$y_{\mathrm{r},21}(\tau)$', '$x_{\mathrm{r},21}(\tau)$', 'g'}, ...
            {'X31_real_solve', Ytt(3,:), xt_2(3,:), '$y_{\mathrm{r},12}(\tau)$', '$x_{\mathrm{r},12}(\tau)$', 'g'}, ...
            {'X12_real_solve', Ytt(4,:), xt_2(4,:), '$y_{\mathrm{r},22}(\tau)$', '$x_{\mathrm{r},22}(\tau)$', 'g'}, ...
            {'X11_imaginary_solve', Ytt(5,:), xt_2(5,:), '$y_{\mathrm{i},11}(\tau)$', '$x_{\mathrm{i},11}(\tau)$', 'g'}, ...
            {'X21_imaginary_solve', Ytt(6,:), xt_2(6,:), '$y_{\mathrm{i},21}(\tau)$', '$x_{\mathrm{i},21}(\tau)$', 'g'}, ...
            {'X12_imaginary_solve', Ytt(7,:), xt_2(7,:), '$y_{\mathrm{i},12}(\tau)$', '$x_{\mathrm{i},12}(\tau)$', 'g'}, ...
            {'X22_imaginary_solve', Ytt(8,:), xt_2(8,:), '$y_{\mathrm{i},22}(\tau)$', '$x_{\mathrm{i},22}(\tau)$', 'g'}
            };
        
        %Generating graphics.
        for i = 1:length(figPropertiesReal)
            figureNumber= mod(i,cm)*1000+ceil(i/cm)*100+ceil(i/(cm*cn))*10+2;
            figure(figureNumber);
            f = gcf;
            f.Name = figPropertiesReal{i}{1};
            f.NumberTitle = 'off';
            
            plot(tk(1,1:end-1)*(1/tau), figPropertiesReal{i}{2}, '--', 'LineWidth', 6);set(gca, 'ZTick', []);view(-45,85);
            set(gca, 'FontSize', 16);
            hold on;
            plot(tk(1,1:end-1)*(1/tau), figPropertiesReal{i}{3}, figPropertiesReal{i}{6}, 'LineWidth', 2);
            
            handle = legend({figPropertiesReal{i}{4}, figPropertiesReal{i}{5}}, 'Position', [0.65, 0.7, 0.335, 0.285], 'FontSize', 40);
            set(handle, 'Interpreter', 'latex', 'FontSize', 36);
            xlabel('$k$', 'Interpreter', 'latex', 'FontSize', 20);
        end
    end
end
