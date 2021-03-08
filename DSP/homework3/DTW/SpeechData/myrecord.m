Record = [];
mkdir 语音库
fprintf("这次测试一共三组，每组用不同语调分别说出中文从0-9十个数字，每次采样持续时间2s，\n最后得到我们的语料库\n")
name = input('请输入你的名字的首字母小写(e.g.dxd,hdq,pym)\n','s');
fprintf("按下空格直接开始\n")
pause

for i = 1:10
    fprintf(['\n','文件名：',num2str(i-1),'\n'])
    record(2,num2str(i-1),i-1);
    ok = input('录音结束,按回车继续，如果想要重新录入请输入：yes\n','s');
    ok = [ok,' '];
    while ok == 'yes '
        record(2,num2str(i-1),i-1);
        ok = input('录音结束,按回车继续，如果想要重新录入请输入：yes\n','s');
        ok = [ok,' '];
    end
end

disp("语音库录入完成，谢谢您的配合。请把语音库后面加上名字首字母缩写‘.\语音库_xxx’ 打包发给学委。")
