function [myRecording] = record(time, name, i)
% ¼��¼times����
recObj = audiorecorder(16000, 16, 1);
fprintf('��ʼ¼����˵��%d\n.',i)
recordblocking(recObj, time);
% �ط�¼������
play(recObj);
% ��ȡ¼������
myRecording = getaudiodata(recObj);
% ����¼�����ݲ���
plot(myRecording);
%�洢�����ź�
filename = ['.\������\', name, '.wav']; 
audiowrite(filename,myRecording,16000);
end