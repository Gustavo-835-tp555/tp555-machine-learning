clear all;close all;clc

fontSize = 14;

M = 100000;

x = 10*randn(M, 1);

y = 2.0*x + 10*randn(M, 1);

a1 = -20:1/1:24;

[A1, A0] = meshgrid(a1, zeros(1, length(a1)));

J = zeros(length(A0),length(A1));
for iter0=1:1:length(a1)
    for iter1=1:1:length(a1)
        yhat = A0(iter0, iter1) + A1(iter0, iter1)*x;
        J(iter0, iter1) = (1/length(y)).*sum((y - yhat).^2);
    end
end

figure1 = figure('rend','painters','pos',[10 10 800 700]);
surf(A0, A1, J)
xlabel('a_0', 'FontSize', fontSize)
ylabel('a_1', 'FontSize', fontSize)
zlabel('J_e', 'FontSize', fontSize)
title('Superf�cie de Erro', 'FontSize', fontSize)

%% Closed-form solution.

X = [ones(M,1) x];

a_opt = pinv(X.'*X)*X.'*y;

yhat = a_opt(1) + a_opt(2)*x;

Joptimum = (1/M)*sum((y - yhat).^2);

%% Gradient-descent solution.
alpha = 0.001;

% Initialize 'a' at a random location within the parameter's space.
a(:,1) = [0;-20];

yhat = a(1,1) + a(2,1)*x;

Jgd(1) = (1/M)*sum((y - yhat).^2);

error = 1;
iter = 1;
while(error > 0.00001 && iter <= 10000)
    
    h = a(1,iter) + a(2,iter)*x;
    
    update = -(2./M).*(y - h).'*X;
    
    a(:,iter+1) = a(:,iter) - alpha.*update.';
    
    yhat = a(1,iter+1) + a(2,iter+1)*x;

    Jgd(iter+1) = (1/M).*sum((y - yhat).^2);
    
    error = abs(Jgd(iter)-Jgd(iter+1));
    
    iter = iter + 1;
    
end

figure2 = figure('rend','painters','pos',[10 10 800 700]);
plot(a1, J(1,:), 'LineWidth', 2)
xlabel('a_1', 'FontSize', fontSize)
ylabel('J_e', 'FontSize', fontSize)
hold on
%scatter3(a_opt(1), a_opt(2), Joptimum, 'r*', 'MarkerSize', 10)
%scatter3(a(1,:), a(2,:), Jgd, 'kx', 'MarkerSize', 10);
plot(a(2,:), Jgd, 'kx--', 'MarkerSize', 10, 'LineWidth',1);
plot(a_opt(2), Joptimum, 'r*', 'MarkerSize', 14, 'LineWidth',2);
hold off;
grid on;

figure3 = figure('rend','painters','pos',[10 10 800 700]);
semilogy(0:1:iter-1, Jgd, 'LineWidth',1)
xlabel('Itera��o', 'FontSize', fontSize)
ylabel('J_e', 'FontSize', fontSize)
title('Itera��es vs. Erro', 'FontSize', fontSize)
lgd = legend('\alpha = 0.001', 'Interpreter', 'latex');
lgd.FontSize = 14;
xlim([0 iter-1])
grid on