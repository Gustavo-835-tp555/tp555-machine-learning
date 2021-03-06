clear all;close all;clc;

fo = 1;
x = 0:1/50:1/2;
x = x.';

M = length(x);

y = cos(2*pi*fo*x + 0.2);

y_noisy = y + 0.5*randn(M, 1);

figure1 = figure('rend','painters','pos',[10 10 800 700]);
fontSize = 14;
plot(x, y, 'LineWidth', 1.5)
hold on
plot(x, y_noisy, 'o', 'LineWidth', 2)
xlabel('x', 'FontSize', fontSize)
ylabel('y', 'FontSize', fontSize)
lgd = legend('Dado original','Dado ruidoso');
lgd.FontSize = 12;
grid on
hold off


%% ------------------------------------------------------------------------
%% Closed-form solution.
%% ------------------------------------------------------------------------

X = [ones(M, 1) x x.^2 x.^3];

a_opt = pinv(X.'*X)*X.'*y;

yhat = a_opt(1) + a_opt(2)*x + a_opt(3)*x.^2 + a_opt(4).*x.^3;

Joptimum = (1/M)*sum((y - yhat).^2);


%% ------------------------------------------------------------------------
%% Batch Gradient Descent
%% ------------------------------------------------------------------------

%% Gradient-descent solution.
alpha = 0.9;

% Initialize 'a' at a random location within the parameter's space.
a = zeros(4, 10000);
a(:,1) = [2;-2;-30;50];

yhat = a(1,1) + a(2,1)*x + a(3,1)*x.^2 + a(4,1)*x.^3;

Jgd = zeros(1, 10000);
Jgd(1) = (1/M)*sum((y - yhat).^2);

error = 1;
iter = 1;
while(error > 0.0001 && iter <= 10000)
    
    h = a(1, iter) + a(2,iter)*x + a(3,iter)*x.^2 + a(4,iter)*x.^3;
    
    update = -(2./M).*(y - h).'*X;
    
    a(:,iter+1) = a(:,iter) - alpha.*update.';
    
    yhat = a(1,iter+1) + a(2,iter+1)*x + a(3,iter+1)*x.^2 + a(4,iter+1)*x.^3;
    
    Jgd(iter+1) = (1/M).*sum((y - yhat).^2);
    
    error = abs(Jgd(iter)-Jgd(iter+1));
    
    iter = iter + 1;
    
end

figure2 = figure('rend','painters','pos',[10 10 800 700]);
semilogy(1:1:iter, Jgd(1:iter), 'LineWidth', 2);
xlabel('Iteração');
ylabel('J_e');
grid on;


yhat = a(1,iter) + a(2,iter)*x + a(3,iter)*x.^2 + a(4,iter)*x.^3;
figure3 = figure('rend','painters','pos',[10 10 800 700]);
plot(x, yhat, 'LineWidth', 2)
hold on
plot(x, y_noisy, 'o', 'LineWidth', 2)
hold off
grid on;



% yhat = a_opt(1) + a_opt(2)*x + a_opt(3)*x.^2 + a_opt(4)*x.^3;
% figure3 = figure('rend','painters','pos',[10 10 800 700]);
% plot(x, yhat)

