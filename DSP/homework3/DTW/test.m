% clear all;
% clc;
% close all;
% 
% q = 'D:\2020-fall\DSP\homework3\DTW\SpeechData\p1\6.wav';
% [speechIn,FS] = audioread(q);
% speechIn2 = my_vad(speechIn);
% 
% figure(1)
% plot(speechIn)
% 
% figure(2)
% plot(speechIn2);

% profile on;
% matchTemplates
% profile viewer

alpha = [0.04 0.06 0.08 0.1 0.12 0.14 0.16 0.18 0.2];
error_rate = [0.09 0.08 0.06 0.037 0.027 0.023 0.03 0.027 0.02];
time = [0.254 0.3 0.349 0.398 0.434 0.477 0.516 0.571 0.611];

figure(1)
plot(alpha,error_rate);
xlabel('alpha');
ylabel('error rate');
hold on 
scatter(1,0.013)

figure(2)
plot(alpha,time)
xlabel('alpha');
ylabel('time');
hold on 
scatter(1,1.42)