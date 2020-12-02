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