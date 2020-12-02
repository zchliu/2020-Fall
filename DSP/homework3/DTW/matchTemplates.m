clear all;
close all;

ncoeff = 12;          %MFCC参数阶数
N = 10;               %10个数字
fs=16000;             %采样频率
duration2 = 2;        %录音时长
k = 3;                %训练样本的人数


speech = audiorecorder(fs,16,1);
disp('Press key to record 2 second');
pause
disp('Recording.');
recordblocking(speech,duration2)             % duration*fs 为采样点数
disp('Finished recording.');
speechIn=getaudiodata(speech);

[num] = process(speechIn);
disp(num2str(num));

% t1 = clock;
% q = ['D:\2020-fall\DSP\homework3\DTW\SpeechData\p5\6.wav'];
% [speechIn,FS] = audioread(q);
% [num] = process(speechIn);
% disp(num);
% t2 = clock;
% t = etime(t2,t1);

%%% 测试模块
% Number_str = struct('zero',0,'One',1,'Two',2,'Three',3,'Four',4,'Five',5,'Six',6,'Seven',7,'Eight',8,'nine',9);
% dir = 'D:\2020-fall\DSP\homework3\DTW\SpeechData\';
% 
% t1 = clock;
% P = zeros(10,10);
% for i = 1:10
%     for j = 0:9
%         doc_path = [dir,'p',num2str(i),'\',num2str(j),'.wav'];
%         [speechIn,FS] = audioread(doc_path);
%         [num] = process(speechIn);
%         num = strrep(num,' ','');
%         answer = j;
%         Number_str.(num);
%         if Number_str.(num) == answer
%             P(i,j+1) = 1;
%             disp(['correct! ',num2str(j), '->',num2str(Number_str.(num))]);
%         else
%             disp(['wrong! ',num2str(j), '->',num2str(Number_str.(num))]);
%         end
%     end
% end
% t2 = clock;
% t = etime(t2,t1);
% disp(t)

