%设置模板
ncoeff=12;                      %mfcc系数的个数
fMatrix1 = cell(1,10);
fields = {'zero','One','Two','Three','Four','Five','Six','Seven','Eight','nine'};

for i = 1:10
    q = ['D:\2020-fall\DSP\homework3\DTW\SpeechData\p1\' num2str(i-1) '.wav'];
    [speechIn1,FS1] = audioread(q);
    speechIn1 = my_vad(speechIn1);
    fMatrix1(1,i) = {mfccf(ncoeff,speechIn1,FS1)};
end
s1 = cell2struct(fMatrix1, fields, 2);         %fields项作为行
save Vectors1.mat -struct s1;

for i = 1:10
    q = ['D:\2020-fall\DSP\homework3\DTW\SpeechData\p2\' num2str(i-1) '.wav'];
    [speechIn1,FS1] = audioread(q);
    speechIn1 = my_vad(speechIn1);
    fMatrix1(1,i) = {mfccf(ncoeff,speechIn1,FS1)};
end
s1 = cell2struct(fMatrix1, fields, 2);         %fields项作为行
save Vectors2.mat -struct s1;

for i = 1:10
    q = ['D:\2020-fall\DSP\homework3\DTW\SpeechData\p3\' num2str(i-1) '.wav'];
    [speechIn1,FS1] = audioread(q);
    speechIn1 = my_vad(speechIn1);
    fMatrix1(1,i) = {mfccf(ncoeff,speechIn1,FS1)};
end
s1 = cell2struct(fMatrix1, fields, 2);         %fields项作为行
save Vectors3.mat -struct s1;

for i = 1:10
    q = ['D:\2020-fall\DSP\homework3\DTW\SpeechData\p4\' num2str(i-1) '.wav'];
    [speechIn1,FS1] = audioread(q);
    speechIn1 = my_vad(speechIn1);
    fMatrix1(1,i) = {mfccf(ncoeff,speechIn1,FS1)};
end
s1 = cell2struct(fMatrix1, fields, 2);         %fields项作为行
save Vectors4.mat -struct s1;

for i = 1:10
    q = ['D:\2020-fall\DSP\homework3\DTW\SpeechData\p5\' num2str(i-1) '.wav'];
    [speechIn1,FS1] = audioread(q);
    speechIn1 = my_vad(speechIn1);
    fMatrix1(1,i) = {mfccf(ncoeff,speechIn1,FS1)};
end
s1 = cell2struct(fMatrix1, fields, 2);         %fields项作为行
save Vectors5.mat -struct s1;

for i = 1:10
    q = ['D:\2020-fall\DSP\homework3\DTW\SpeechData\p6\' num2str(i-1) '.wav'];
    [speechIn1,FS1] = audioread(q);
    speechIn1 = my_vad(speechIn1);
    fMatrix1(1,i) = {mfccf(ncoeff,speechIn1,FS1)};
end
s1 = cell2struct(fMatrix1, fields, 2);         %fields项作为行
save Vectors6.mat -struct s1;

for i = 1:10
    q = ['D:\2020-fall\DSP\homework3\DTW\SpeechData\p7\' num2str(i-1) '.wav'];
    [speechIn1,FS1] = audioread(q);
    speechIn1 = my_vad(speechIn1);
    fMatrix1(1,i) = {mfccf(ncoeff,speechIn1,FS1)};
end
s1 = cell2struct(fMatrix1, fields, 2);         %fields项作为行
save Vectors7.mat -struct s1;

for i = 1:10
    q = ['D:\2020-fall\DSP\homework3\DTW\SpeechData\p8\' num2str(i-1) '.wav'];
    [speechIn1,FS1] = audioread(q);
    speechIn1 = my_vad(speechIn1);
    fMatrix1(1,i) = {mfccf(ncoeff,speechIn1,FS1)};
end
s1 = cell2struct(fMatrix1, fields, 2);         %fields项作为行
save Vectors8.mat -struct s1;

for i = 1:10
    q = ['D:\2020-fall\DSP\homework3\DTW\SpeechData\p9\' num2str(i-1) '.wav'];
    [speechIn1,FS1] = audioread(q);
    speechIn1 = my_vad(speechIn1);
    fMatrix1(1,i) = {mfccf(ncoeff,speechIn1,FS1)};
end
s1 = cell2struct(fMatrix1, fields, 2);         %fields项作为行
save Vectors9.mat -struct s1;

for i = 1:10
    q = ['D:\2020-fall\DSP\homework3\DTW\SpeechData\p10\' num2str(i-1) '.wav'];
    [speechIn1,FS1] = audioread(q);
    speechIn1 = my_vad(speechIn1);
    fMatrix1(1,i) = {mfccf(ncoeff,speechIn1,FS1)};
end
s1 = cell2struct(fMatrix1, fields, 2);         %fields项作为行
save Vectors10.mat -struct s1;

for i = 1:10
    q = ['D:\2020-fall\DSP\homework3\DTW\SpeechData\p11\' num2str(i-1) '.wav'];
    [speechIn1,FS1] = audioread(q);
    speechIn1 = my_vad(speechIn1);
    fMatrix1(1,i) = {mfccf(ncoeff,speechIn1,FS1)};
end
s1 = cell2struct(fMatrix1, fields, 2);         %fields项作为行
save Vectors11.mat -struct s1;

for i = 1:10
    q = ['D:\2020-fall\DSP\homework3\DTW\SpeechData\p12\' num2str(i-1) '.wav'];
    [speechIn1,FS1] = audioread(q);
    speechIn1 = my_vad(speechIn1);
    fMatrix1(1,i) = {mfccf(ncoeff,speechIn1,FS1)};
end
s1 = cell2struct(fMatrix1, fields, 2);         %fields项作为行
save Vectors12.mat -struct s1;

for i = 1:10
    q = ['D:\2020-fall\DSP\homework3\DTW\SpeechData\p13\' num2str(i-1) '.wav'];
    [speechIn1,FS1] = audioread(q);
    speechIn1 = my_vad(speechIn1);
    fMatrix1(1,i) = {mfccf(ncoeff,speechIn1,FS1)};
end
s1 = cell2struct(fMatrix1, fields, 2);         %fields项作为行
save Vectors13.mat -struct s1;

for i = 1:10
    q = ['D:\2020-fall\DSP\homework3\DTW\SpeechData\p14\' num2str(i-1) '.wav'];
    [speechIn1,FS1] = audioread(q);
    speechIn1 = my_vad(speechIn1);
    fMatrix1(1,i) = {mfccf(ncoeff,speechIn1,FS1)};
end
s1 = cell2struct(fMatrix1, fields, 2);         %fields项作为行
save Vectors14.mat -struct s1;

for i = 1:10
    q = ['D:\2020-fall\DSP\homework3\DTW\SpeechData\p15\' num2str(i-1) '.wav'];
    [speechIn1,FS1] = audioread(q);
    speechIn1 = my_vad(speechIn1);
    fMatrix1(1,i) = {mfccf(ncoeff,speechIn1,FS1)};
end
s1 = cell2struct(fMatrix1, fields, 2);         %fields项作为行
save Vectors15.mat -struct s1;

for i = 1:10
    q = ['D:\2020-fall\DSP\homework3\DTW\SpeechData\p16\' num2str(i-1) '.wav'];
    [speechIn1,FS1] = audioread(q);
    speechIn1 = my_vad(speechIn1);
    fMatrix1(1,i) = {mfccf(ncoeff,speechIn1,FS1)};
end
s1 = cell2struct(fMatrix1, fields, 2);         %fields项作为行
save Vectors16.mat -struct s1;

for i = 1:10
    q = ['D:\2020-fall\DSP\homework3\DTW\SpeechData\p17\' num2str(i-1) '.wav'];
    [speechIn1,FS1] = audioread(q);
    speechIn1 = my_vad(speechIn1);
    fMatrix1(1,i) = {mfccf(ncoeff,speechIn1,FS1)};
end
s1 = cell2struct(fMatrix1, fields, 2);         %fields项作为行
save Vectors17.mat -struct s1;

for i = 1:10
    q = ['D:\2020-fall\DSP\homework3\DTW\SpeechData\p18\' num2str(i-1) '.wav'];
    [speechIn1,FS1] = audioread(q);
    speechIn1 = my_vad(speechIn1);
    fMatrix1(1,i) = {mfccf(ncoeff,speechIn1,FS1)};
end
s1 = cell2struct(fMatrix1, fields, 2);         %fields项作为行
save Vectors18.mat -struct s1;

for i = 1:10
    q = ['D:\2020-fall\DSP\homework3\DTW\SpeechData\p19\' num2str(i-1) '.wav'];
    [speechIn1,FS1] = audioread(q);
    speechIn1 = my_vad(speechIn1);
    fMatrix1(1,i) = {mfccf(ncoeff,speechIn1,FS1)};
end
s1 = cell2struct(fMatrix1, fields, 2);         %fields项作为行
save Vectors19.mat -struct s1;

for i = 1:10
    q = ['D:\2020-fall\DSP\homework3\DTW\SpeechData\p20\' num2str(i-1) '.wav'];
    [speechIn1,FS1] = audioread(q);
    speechIn1 = my_vad(speechIn1);
    fMatrix1(1,i) = {mfccf(ncoeff,speechIn1,FS1)};
end
s1 = cell2struct(fMatrix1, fields, 2);         %fields项作为行
save Vectors20.mat -struct s1;

for i = 1:10
    q = ['D:\2020-fall\DSP\homework3\DTW\SpeechData\p21\' num2str(i-1) '.wav'];
    [speechIn1,FS1] = audioread(q);
    speechIn1 = my_vad(speechIn1);
    fMatrix1(1,i) = {mfccf(ncoeff,speechIn1,FS1)};
end
s1 = cell2struct(fMatrix1, fields, 2);         %fields项作为行
save Vectors21.mat -struct s1;

for i = 1:10
    q = ['D:\2020-fall\DSP\homework3\DTW\SpeechData\p22\' num2str(i-1) '.wav'];
    [speechIn1,FS1] = audioread(q);
    speechIn1 = my_vad(speechIn1);
    fMatrix1(1,i) = {mfccf(ncoeff,speechIn1,FS1)};
end
s1 = cell2struct(fMatrix1, fields, 2);         %fields项作为行
save Vectors22.mat -struct s1;

for i = 1:10
    q = ['D:\2020-fall\DSP\homework3\DTW\SpeechData\p23\' num2str(i-1) '.wav'];
    [speechIn1,FS1] = audioread(q);
    speechIn1 = my_vad(speechIn1);
    fMatrix1(1,i) = {mfccf(ncoeff,speechIn1,FS1)};
end
s1 = cell2struct(fMatrix1, fields, 2);         %fields项作为行
save Vectors23.mat -struct s1;

for i = 1:10
    q = ['D:\2020-fall\DSP\homework3\DTW\SpeechData\p24\' num2str(i-1) '.wav'];
    [speechIn1,FS1] = audioread(q);
    speechIn1 = my_vad(speechIn1);
    fMatrix1(1,i) = {mfccf(ncoeff,speechIn1,FS1)};
end
s1 = cell2struct(fMatrix1, fields, 2);         %fields项作为行
save Vectors24.mat -struct s1;

for i = 1:10
    q = ['D:\2020-fall\DSP\homework3\DTW\SpeechData\p25\' num2str(i-1) '.wav'];
    [speechIn1,FS1] = audioread(q);
    speechIn1 = my_vad(speechIn1);
    fMatrix1(1,i) = {mfccf(ncoeff,speechIn1,FS1)};
end
s1 = cell2struct(fMatrix1, fields, 2);         %fields项作为行
save Vectors25.mat -struct s1;

for i = 1:10
    q = ['D:\2020-fall\DSP\homework3\DTW\SpeechData\p26\' num2str(i-1) '.wav'];
    [speechIn1,FS1] = audioread(q);
    speechIn1 = my_vad(speechIn1);
    fMatrix1(1,i) = {mfccf(ncoeff,speechIn1,FS1)};
end
s1 = cell2struct(fMatrix1, fields, 2);         %fields项作为行
save Vectors26.mat -struct s1;

for i = 1:10
    q = ['D:\2020-fall\DSP\homework3\DTW\SpeechData\p27\' num2str(i-1) '.wav'];
    [speechIn1,FS1] = audioread(q);
    speechIn1 = my_vad(speechIn1);
    fMatrix1(1,i) = {mfccf(ncoeff,speechIn1,FS1)};
end
s1 = cell2struct(fMatrix1, fields, 2);         %fields项作为行
save Vectors27.mat -struct s1;

for i = 1:10
    q = ['D:\2020-fall\DSP\homework3\DTW\SpeechData\p28\' num2str(i-1) '.wav'];
    [speechIn1,FS1] = audioread(q);
    speechIn1 = my_vad(speechIn1);
    fMatrix1(1,i) = {mfccf(ncoeff,speechIn1,FS1)};
end
s1 = cell2struct(fMatrix1, fields, 2);         %fields项作为行
save Vectors28.mat -struct s1;

for i = 1:10
    q = ['D:\2020-fall\DSP\homework3\DTW\SpeechData\p29\' num2str(i-1) '.wav'];
    [speechIn1,FS1] = audioread(q);
    speechIn1 = my_vad(speechIn1);
    fMatrix1(1,i) = {mfccf(ncoeff,speechIn1,FS1)};
end
s1 = cell2struct(fMatrix1, fields, 2);         %fields项作为行
save Vectors29.mat -struct s1;

for i = 1:10
    q = ['D:\2020-fall\DSP\homework3\DTW\SpeechData\p30\' num2str(i-1) '.wav'];
    [speechIn1,FS1] = audioread(q);
    speechIn1 = my_vad(speechIn1);
    fMatrix1(1,i) = {mfccf(ncoeff,speechIn1,FS1)};
end
s1 = cell2struct(fMatrix1, fields, 2);         %fields项作为行
save Vectors30.mat -struct s1;