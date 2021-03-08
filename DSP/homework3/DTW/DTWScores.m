function AllScores = DTWScores(rMatrix,N)
%动态时间规整（DTW）寻找最小失真
%输入参数：rMatrix为当前读入语音的MFCC参数矩阵,N为每个模板数量词汇数
%输出参数：AllScores为

%初始化DTW判别矩阵
Scores1 = zeros(1,N);
s1 = load('Vectors1.mat');
fMatrixall1 = struct2cell(s1);
for i = 1:N
    fMatrix1 = fMatrixall1{i,1};
    fMatrix1 = CMN(fMatrix1);
    Scores1(i) = myDTW(fMatrix1,rMatrix);
end
AllScores(:,1) = Scores1;

Scores1 = zeros(1,N);
s1 = load('Vectors2.mat');
fMatrixall1 = struct2cell(s1);
for i = 1:N
    fMatrix1 = fMatrixall1{i,1};
    fMatrix1 = CMN(fMatrix1);
    Scores1(i) = myDTW(fMatrix1,rMatrix);
end
AllScores(:,2) = Scores1;

Scores1 = zeros(1,N);
s1 = load('Vectors3.mat');
fMatrixall1 = struct2cell(s1);
for i = 1:N
    fMatrix1 = fMatrixall1{i,1};
    fMatrix1 = CMN(fMatrix1);
    Scores1(i) = myDTW(fMatrix1,rMatrix);
end
AllScores(:,3) = Scores1;

Scores1 = zeros(1,N);
s1 = load('Vectors4.mat');
fMatrixall1 = struct2cell(s1);
for i = 1:N
    fMatrix1 = fMatrixall1{i,1};
    fMatrix1 = CMN(fMatrix1);
    Scores1(i) = myDTW(fMatrix1,rMatrix);
end
AllScores(:,4) = Scores1;

Scores1 = zeros(1,N);
s1 = load('Vectors5.mat');
fMatrixall1 = struct2cell(s1);
for i = 1:N
    fMatrix1 = fMatrixall1{i,1};
    fMatrix1 = CMN(fMatrix1);
    Scores1(i) = myDTW(fMatrix1,rMatrix);
end
AllScores(:,5) = Scores1;

Scores1 = zeros(1,N);
s1 = load('Vectors6.mat');
fMatrixall1 = struct2cell(s1);
for i = 1:N
    fMatrix1 = fMatrixall1{i,1};
    fMatrix1 = CMN(fMatrix1);
    Scores1(i) = myDTW(fMatrix1,rMatrix);
end
AllScores(:,6) = Scores1;

Scores1 = zeros(1,N);
s1 = load('Vectors7.mat');
fMatrixall1 = struct2cell(s1);
for i = 1:N
    fMatrix1 = fMatrixall1{i,1};
    fMatrix1 = CMN(fMatrix1);
    Scores1(i) = myDTW(fMatrix1,rMatrix);
end
AllScores(:,7) = Scores1;

Scores1 = zeros(1,N);
s1 = load('Vectors8.mat');
fMatrixall1 = struct2cell(s1);
for i = 1:N
    fMatrix1 = fMatrixall1{i,1};
    fMatrix1 = CMN(fMatrix1);
    Scores1(i) = myDTW(fMatrix1,rMatrix);
end
AllScores(:,8) = Scores1;

Scores1 = zeros(1,N);
s1 = load('Vectors9.mat');
fMatrixall1 = struct2cell(s1);
for i = 1:N
    fMatrix1 = fMatrixall1{i,1};
    fMatrix1 = CMN(fMatrix1);
    Scores1(i) = myDTW(fMatrix1,rMatrix);
end
AllScores(:,9) = Scores1;

Scores1 = zeros(1,N);
s1 = load('Vectors10.mat');
fMatrixall1 = struct2cell(s1);
for i = 1:N
    fMatrix1 = fMatrixall1{i,1};
    fMatrix1 = CMN(fMatrix1);
    Scores1(i) = myDTW(fMatrix1,rMatrix);
end
AllScores(:,10) = Scores1;

Scores1 = zeros(1,N);
s1 = load('Vectors11.mat');
fMatrixall1 = struct2cell(s1);
for i = 1:N
    fMatrix1 = fMatrixall1{i,1};
    fMatrix1 = CMN(fMatrix1);
    Scores1(i) = myDTW(fMatrix1,rMatrix);
end
AllScores(:,11) = Scores1;

Scores1 = zeros(1,N);
s1 = load('Vectors12.mat');
fMatrixall1 = struct2cell(s1);
for i = 1:N
    fMatrix1 = fMatrixall1{i,1};
    fMatrix1 = CMN(fMatrix1);
    Scores1(i) = myDTW(fMatrix1,rMatrix);
end
AllScores(:,12) = Scores1;

Scores1 = zeros(1,N);
s1 = load('Vectors13.mat');
fMatrixall1 = struct2cell(s1);
for i = 1:N
    fMatrix1 = fMatrixall1{i,1};
    fMatrix1 = CMN(fMatrix1);
    Scores1(i) = myDTW(fMatrix1,rMatrix);
end
AllScores(:,13) = Scores1;

Scores1 = zeros(1,N);
s1 = load('Vectors14.mat');
fMatrixall1 = struct2cell(s1);
for i = 1:N
    fMatrix1 = fMatrixall1{i,1};
    fMatrix1 = CMN(fMatrix1);
    Scores1(i) = myDTW(fMatrix1,rMatrix);
end
AllScores(:,14) = Scores1;

Scores1 = zeros(1,N);
s1 = load('Vectors15.mat');
fMatrixall1 = struct2cell(s1);
for i = 1:N
    fMatrix1 = fMatrixall1{i,1};
    fMatrix1 = CMN(fMatrix1);
    Scores1(i) = myDTW(fMatrix1,rMatrix);
end
AllScores(:,15) = Scores1;

Scores1 = zeros(1,N);
s1 = load('Vectors16.mat');
fMatrixall1 = struct2cell(s1);
for i = 1:N
    fMatrix1 = fMatrixall1{i,1};
    fMatrix1 = CMN(fMatrix1);
    Scores1(i) = myDTW(fMatrix1,rMatrix);
end
AllScores(:,16) = Scores1;

Scores1 = zeros(1,N);
s1 = load('Vectors17.mat');
fMatrixall1 = struct2cell(s1);
for i = 1:N
    fMatrix1 = fMatrixall1{i,1};
    fMatrix1 = CMN(fMatrix1);
    Scores1(i) = myDTW(fMatrix1,rMatrix);
end
AllScores(:,17) = Scores1;

Scores1 = zeros(1,N);
s1 = load('Vectors18.mat');
fMatrixall1 = struct2cell(s1);
for i = 1:N
    fMatrix1 = fMatrixall1{i,1};
    fMatrix1 = CMN(fMatrix1);
    Scores1(i) = myDTW(fMatrix1,rMatrix);
end
AllScores(:,18) = Scores1;

Scores1 = zeros(1,N);
s1 = load('Vectors19.mat');
fMatrixall1 = struct2cell(s1);
for i = 1:N
    fMatrix1 = fMatrixall1{i,1};
    fMatrix1 = CMN(fMatrix1);
    Scores1(i) = myDTW(fMatrix1,rMatrix);
end
AllScores(:,19) = Scores1;

Scores1 = zeros(1,N);
s1 = load('Vectors20.mat');
fMatrixall1 = struct2cell(s1);
for i = 1:N
    fMatrix1 = fMatrixall1{i,1};
    fMatrix1 = CMN(fMatrix1);
    Scores1(i) = myDTW(fMatrix1,rMatrix);
end
AllScores(:,20) = Scores1;

Scores1 = zeros(1,N);
s1 = load('Vectors21.mat');
fMatrixall1 = struct2cell(s1);
for i = 1:N
    fMatrix1 = fMatrixall1{i,1};
    fMatrix1 = CMN(fMatrix1);
    Scores1(i) = myDTW(fMatrix1,rMatrix);
end
AllScores(:,21) = Scores1;

Scores1 = zeros(1,N);
s1 = load('Vectors22.mat');
fMatrixall1 = struct2cell(s1);
for i = 1:N
    fMatrix1 = fMatrixall1{i,1};
    fMatrix1 = CMN(fMatrix1);
    Scores1(i) = myDTW(fMatrix1,rMatrix);
end
AllScores(:,22) = Scores1;

Scores1 = zeros(1,N);
s1 = load('Vectors23.mat');
fMatrixall1 = struct2cell(s1);
for i = 1:N
    fMatrix1 = fMatrixall1{i,1};
    fMatrix1 = CMN(fMatrix1);
    Scores1(i) = myDTW(fMatrix1,rMatrix);
end
AllScores(:,23) = Scores1;

Scores1 = zeros(1,N);
s1 = load('Vectors24.mat');
fMatrixall1 = struct2cell(s1);
for i = 1:N
    fMatrix1 = fMatrixall1{i,1};
    fMatrix1 = CMN(fMatrix1);
    Scores1(i) = myDTW(fMatrix1,rMatrix);
end
AllScores(:,24) = Scores1;

Scores1 = zeros(1,N);
s1 = load('Vectors25.mat');
fMatrixall1 = struct2cell(s1);
for i = 1:N
    fMatrix1 = fMatrixall1{i,1};
    fMatrix1 = CMN(fMatrix1);
    Scores1(i) = myDTW(fMatrix1,rMatrix);
end
AllScores(:,25) = Scores1;

Scores1 = zeros(1,N);
s1 = load('Vectors26.mat');
fMatrixall1 = struct2cell(s1);
for i = 1:N
    fMatrix1 = fMatrixall1{i,1};
    fMatrix1 = CMN(fMatrix1);
    Scores1(i) = myDTW(fMatrix1,rMatrix);
end
AllScores(:,26) = Scores1;

Scores1 = zeros(1,N);
s1 = load('Vectors27.mat');
fMatrixall1 = struct2cell(s1);
for i = 1:N
    fMatrix1 = fMatrixall1{i,1};
    fMatrix1 = CMN(fMatrix1);
    Scores1(i) = myDTW(fMatrix1,rMatrix);
end
AllScores(:,27) = Scores1;

Scores1 = zeros(1,N);
s1 = load('Vectors28.mat');
fMatrixall1 = struct2cell(s1);
for i = 1:N
    fMatrix1 = fMatrixall1{i,1};
    fMatrix1 = CMN(fMatrix1);
    Scores1(i) = myDTW(fMatrix1,rMatrix);
end
AllScores(:,28) = Scores1;

Scores1 = zeros(1,N);
s1 = load('Vectors29.mat');
fMatrixall1 = struct2cell(s1);
for i = 1:N
    fMatrix1 = fMatrixall1{i,1};
    fMatrix1 = CMN(fMatrix1);
    Scores1(i) = myDTW(fMatrix1,rMatrix);
end
AllScores(:,29) = Scores1;

Scores1 = zeros(1,N);
s1 = load('Vectors30.mat');
fMatrixall1 = struct2cell(s1);
for i = 1:N
    fMatrix1 = fMatrixall1{i,1};
    fMatrix1 = CMN(fMatrix1);
    Scores1(i) = myDTW(fMatrix1,rMatrix);
end
AllScores(:,30) = Scores1;