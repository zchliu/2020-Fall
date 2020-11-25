clear all;
close all;
ncoeff = 12;          %MFCC��������
N = 10;               %10������
fs=16000;             %����Ƶ��                
duration2 = 2;        %¼��ʱ��
k = 3;                %ѵ������������

speech = audiorecorder(fs,16,1); 
disp('Press key to record 2 second'); 
pause
disp('Recording.'); 
recordblocking(speech,duration2)             % duration*fs Ϊ�������� 
speechIn=getaudiodata(speech);
disp('Finished recording.');
speechIn = my_vad(speechIn);                    %�˵��� 
rMatrix1 = mfccf(ncoeff,speechIn,fs);            %����MFCCϵ����Ϊ����ʸ��
rMatrix = CMN(rMatrix1);                         %��һ������                    

Sco = DTWScores(rMatrix,N);                      %����DTWֵ
[SortedScores,EIndex] = sort(Sco,2);             %���е������򣬲����ض�Ӧ��ԭʼ����
Nbr = EIndex(:,1:2)                              %�õ�ÿ��ģ��ƥ���2�����ֵ��Ӧ�Ĵ���

[Modal,Freq] = mode(Nbr(:));                      %���س���Ƶ����ߵ���Modal�������Ƶ��Freq

Word = char('zero','One','Two','Three','Four','Five','Six','Seven','Eight','Nine'); 
if mean(abs(speechIn)) < 0.01
    fprintf('No microphone connected or you have not said anything.\n');
elseif (Freq <2)                                %Ƶ��̫�Ͳ�ȷ��
    fprintf('The word you have said could not be properly recognised.\n');
else
    fprintf('You have just said %s.\n',Word(Modal,:)); 
end

