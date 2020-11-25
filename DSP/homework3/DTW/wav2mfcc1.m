function [rMatrix] = wav2mfcc1(path, ncoeff)
%WAV2MFCC Summary of this function goes here
% path: .wav格式的路径
% ncoeff: 梅尔滤波器个数 一般取12
[speech,fs] = audioread(path);
rMatrix1 = mfccf(ncoeff,speech,fs);            %采用MFCC系数作为特征矢量
rMatrix = CMN(rMatrix1); 
end

