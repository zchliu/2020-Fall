function [myRecording] = record(time, name, i)
% 录音录times秒钟
recObj = audiorecorder(16000, 16, 1);
fprintf('开始录音请说：%d\n.',i)
recordblocking(recObj, time);
% 回放录音数据
play(recObj);
% 获取录音数据
myRecording = getaudiodata(recObj);
% 绘制录音数据波形
plot(myRecording);
%存储语音信号
filename = ['.\语音库\', name, '.wav']; 
audiowrite(filename,myRecording,16000);
end