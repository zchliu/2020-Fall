%设置模板
ncoeff=12;                      %mfcc系数的个数
fMatrix1 = cell(1,10);
fMatrix2 = cell(1,10);
fMatrix3 = cell(1,10);

for i = 1:10
    q = ['D:\语音信号处理实验教程（MATLAB源代码）\第10章 语音识别\10.1基于动态时间规整（DTW）的孤立字语音识别实验\SpeechData\p1\' num2str(i-1) '.wav'];
%     q = ['D:\数据集\DSP_datasets\DSP_data\语音库_yjq\yjq_0_1.wav'];
    [speechIn1,FS1] = audioread(q);
    figure(1);
    t = (1:length(speechIn1))/FS1;
    plot(t,speechIn1);
    xlabel('时间/s');
    ylabel('幅度');
    title('原始信号幅度谱')
    speechIn1 = my_vad(speechIn1);
    figure(2);
    t = (1:length(speechIn1))/FS1;
    plot(t,speechIn1);
    xlabel('时间/s');
    ylabel('幅度');
    title('检测后信号幅度谱')
    fMatrix1(1,i) = {mfccf(ncoeff,speechIn1,FS1)};
    
end

for j = 1:10
    q = ['F:\SpeechData\p2\' num2str(j-1) '.wav'];
    [speechIn2,FS2] = audioread(q);
    speechIn2 = my_vad(speechIn2);
    fMatrix2(1,j) = {mfccf(ncoeff,speechIn2,FS2)};
end

for k = 1:10
    q = ['F:\SpeechData\p3\' num2str(k-1) '.wav'];
    [speechIn3,FS3] = audioread(q);
    speechIn3 = my_vad(speechIn3);
    fMatrix3(1,k) = {mfccf(ncoeff,speechIn3,FS3)};
end


%将数据保存为mat文件
fields = {'zero','One','Two','Three','Four','Five','Six','Seven','Eight','nine'};
s1 = cell2struct(fMatrix1, fields, 2);         %fields项作为行
save Vectors1.mat -struct s1;
s2 = cell2struct(fMatrix2, fields, 2);
save Vectors2.mat -struct s2;
s3 = cell2struct(fMatrix3, fields, 2);
save Vectors3.mat -struct s3;