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