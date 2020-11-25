clc;
clear;
% 超参数
param_K = 0.01;
param_gamma_p = 0.9;
param_gamma_i = 0.00001;
param_gamma_d = 0.3;
iteration = 10000;
K = 94.75;
tao = 530;
T = 2900.6;
Ts = 10;
Kp = 1.2*T/K/tao;
Ki = Kp*Ts/(2*tao);
Kd = Kp*(0.5*tao)/Ts;
w1_0 = Kp/(Kp+Ki+Kd);
w2_0 = Ki/(Kp+Ki+Kd);
w3_0 = Kd/(Kp+Ki+Kd);
disp(w1_0);disp(w2_0);disp(w3_0)
w1(1) = w1_0;
w2(1) = w2_0;
w3(1) = w3_0;
r = 80;
u(1) = 1;
y(1) = 0;
for k = 1:iteration
    e(k) = r - y(k);
    if k == 1
        x1(k) = e(k);
        x2(k) = e(k);
        x3(k) = e(k);
    elseif k == 2
            x1(k) = e(k) - e(k-1);
            x2(k) = e(k);
            x3(k) = e(k) - 2*e(k-1);
    elseif k > 2
        x1(k) = e(k) - e(k-1);
        x2(k) = e(k);
        x3(k) = e(k) - 2*e(k-1) + e(k-2);
    end
   
    deltau(k) = param_K * (w1(k)*x1(k) + w2(k)*x2(k) + w3(k)*x3(k))/(abs(w1(k)) + abs(w2(k)) + abs(w3(k)));
   
    if k == 1
        u(k) = deltau(k) + 1;
    elseif k > 1
        u(k) = deltau(k) + u(k-1);
    end
   
    if k < 53
        y(k+1) = 0.9966*y(k) + 0.3221;
    elseif k >= 53
        y(k+1) = 0.9966*y(k) + 0.3221*u(k-52);
    end
   
    w1(k+1) = w1(k)+param_gamma_p*e(k)*u(k)*x1(k);
    w2(k+1) = w2(k)+param_gamma_i*e(k)*u(k)*x2(k);
    w3(k+1) = w3(k)+param_gamma_d*e(k)*u(k)*x3(k);
end
y = y(1:iteration);
u = u(1:iteration);
w1 = w1(1:iteration);
w2 = w2(1:iteration);
w3 = w3(1:iteration);
n = (1:10:10*iteration);
figure(1)
plot(n,y);
figure(2)
plot(n,u);
figure(3)
plot(n,w1);
figure(4)
plot(n,w2);
figure(5)
plot(n,w3);