file = 'C:\Users\janrei\Desktop\delete\test.tiff';
folder = 'C:\Users\janrei\Desktop\delete\test\';

fprintf("\nAQUA code:\n");
tic;
[a_dat,maxImg] = io.readTiffSeq(file);
toc;

fprintf("\nMultipage TIFF:\n");
tic;
m_dat = tiff.loadtiff(file);
toc;


fprintf("\nFrom folder TIFF:\n");
tic;
[f_dat, f_maxImg] = load_from_folder(folder, 1, 1);
toc;