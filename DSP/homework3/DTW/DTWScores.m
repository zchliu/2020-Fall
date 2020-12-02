function AllScores = DTWScores(rMatrix,N)
%��̬ʱ�������DTW��Ѱ����Сʧ��
%���������rMatrixΪ��ǰ����������MFCC��������,NΪÿ��ģ�������ʻ���
%���������AllScoresΪ

%��ʼ��DTW�б����
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