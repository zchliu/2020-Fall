function [num] = process(speechIn)

ncoeff = 12;          %MFCC参数阶数
N = 10;               %10个数字
fs=16000;             %采样频率
duration2 = 2;        %录音时长
k = 3;                %训练样本的人数

speechIn = my_vad(speechIn);                    %端点检测
rMatrix1 = mfccf(ncoeff,speechIn,fs);            %采用MFCC系数作为特征矢量
rMatrix = CMN(rMatrix1);                         %归一化处理

Sco = DTWScores(rMatrix,N);                      %计算DTW值
[SortedScores,EIndex] = sort(Sco,1);             %按行递增排序，并返回对应的原始次序

Nbr1 = EIndex(1:1,:);
Nbr2 = EIndex(1:2,:); 
count = hist(Nbr1,unique(Nbr1));
count = sort(count,'descend');
if count(1) > count(2)
    Nbr = Nbr1;
else
    Nbr = Nbr2;
end
[Modal,Freq] = mode(Nbr(:));                      %返回出现频率最高的数Modal及其出现频率Freq

Word = char('zero','One','Two','Three','Four','Five','Six','Seven','Eight','nine');
if mean(abs(speechIn)) < 0.01
    num = 'what?';
elseif (Freq <2)                                %频率太低不确定
    num = 'what?';
else
    num = Word(Modal,:);
end
end

