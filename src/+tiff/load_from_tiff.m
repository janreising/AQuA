function [img, opts] = load_from_tiff(path, opts)
    
    % set parameters 
    [filepath,name,ext] = fileparts(path);
    disp(path);
    
    opts.filePath = filepath;
    opts.fileName = name;
    opts.fileType = ext; 
    
    % load data fast
    img = tiff.loadtiff(path);

    % set image info
    info = imfinfo(path);
    
    opts.sz = size(img);
    opts.maxValueDepth = info.BitDepth;
    opts.maxValueDat = info.MaxSampleValue;
    
    % necessary conversion
    img = double(img);
end