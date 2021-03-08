function [cost] = myDTW(F,R)
%���������FΪģ��MFCC��������RΪ��ǰ����MFCC��������
%���������costΪ���ƥ�����

%��Сƥ��İٷֱ�

alpha = 0.2;

[r1,c1]=size(F);         %ģ���ά��
[r2,c2]=size(R);         %��ǰ����ά��
distance = Inf(r1,r2);

for n=1:r1
    for m=1:r2
        if (r2/r1*n - alpha*r2) < m && m < (r2/r1*n + alpha*r2)
            FR=F(n,:)-R(m,:);
            FR=FR.^2;
            distance(n,m)=sqrt(sum(FR))/c1;     %����ŷ�Ͼ���
        end
    end
end

D = Inf(r1+1,r2+1);
D(1,1) = 0;
D(2:(r1+1), 2:(r2+1)) = distance;

%Ѱ���������̵����ƥ�����
for i = 1:r1
    for j = 1:r2
        if (r2/r1*i - alpha*r2) < j && j < (r2/r1*i + alpha*r2)
            [dmin] = min([D(i, j), D(i, j+1), D(i+1, j)]);
            D(i+1,j+1) = D(i+1,j+1)+dmin;
        end
    end
end

cost = D(r1+1,r2+1);    %�������վ���

