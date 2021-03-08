clc;
clear;
close all;

s1 = load('Vectors1.mat');
fMatrixall1 = struct2cell(s1);

species = {};

max_len = 0;
for i = 1:10
    for j = 1:30
        s1 = load(['Vectors', num2str(j), '.mat']);
        fMatrixall1 = struct2cell(s1);
        fMatrix1 = fMatrixall1{i,1};
        fMatrix1 = CMN(fMatrix1);
        a = reshape(fMatrix1,1,length(fMatrix1(:)));
        len = length(a);
        if len > max_len
            max_len = len;
        end
    end
end

M = [];
for i = 1:10
    for j = 1:30
        s1 = load(['Vectors', num2str(j), '.mat']);
        species{j+(i-1)*30,1} = num2str(i - 1);
        fMatrixall1 = struct2cell(s1);
        fMatrix1 = fMatrixall1{i,1};
        fMatrix1 = CMN(fMatrix1);
        a = reshape(fMatrix1,1,length(fMatrix1(:)));
        M = [M;a,zeros(1,max_len-length(a))];
    end
end

Y = tsne(M,'Algorithm','exact','Distance','cosine','NumPCAComponents',3500);
gscatter(Y(:,1),Y(:,2),species)

% Y = tsne(M,'Algorithm','barneshut','NumPCAComponents',2500,'NumDimensions',3);
% v = double(categorical(species));
% c = full(sparse(1:numel(v),v,ones(size(v)),numel(v),3));
% scatter3(Y(:,1),Y(:,2),Y(:,3),15,c,'filled')

% v = double(categorical(species));
% c = full(sparse(1:numel(v),v,ones(size(v)),numel(v),3));
% scatter3(Y(:,1),Y(:,2),Y(:,3),30,c,'filled')
% title('3-D Embedding')
% view(-50,8)


% load fisheriris
% rng default % for reproducibility
% [Y,loss] = tsne(meas,'Algorithm','exact');
% rng default % for fair comparison
% [Y2,loss2] = tsne(meas,'Algorithm','exact','NumDimensions',3);
% fprintf('2-D embedding has loss %g, and 3-D embedding has loss %g.\n',loss,loss2)
% 
% gscatter(Y(:,1),Y(:,2),species,eye(3))
% title('2-D Embedding')
% 
% figure
% v = double(categorical(species));
% c = full(sparse(1:numel(v),v,ones(size(v)),numel(v),3));
% scatter3(Y2(:,1),Y2(:,2),Y2(:,3),15,c,'filled')
% title('3-D Embedding')
% view(-50,8)

% Y = tsne(M,'Algorithm','barneshut','NumPCAComponents',50,'Exaggeration',1.5);
% figure
% gscatter(Y(:,1),Y(:,2),species)
% title('Default Figure')
