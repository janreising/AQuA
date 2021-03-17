%reset environment
clearvars;
clear options;

% arguments
channels = 2;

% get files
folder = "F:\JR\data\ca_imaging\III_09.03.2021_Dual\calcium\#2\2-20X-loc1.long\";
%folder = "F:\JR\data\ca_imaging\III_09.03.2021_Dual\calcium\#2\1-20X-loc1\";
files = dir(strcat(folder,"*tif*"));

% extract frame number
for k=1:numel(files)
    name = files(k).name;
    
    % convert name to location in array
    name = split(name, "_");
    name = name(end);
    name = split(name, ".");
    frame = name(1);
    frame = str2double(frame);
    
    files(k).frame = frame;
end

%sort files by frame number
files = struct2table(files);
files = sortrows(files, 'frame');
files = table2struct(files);

% get dimensions and create empty array
info = imfinfo(strcat(folder,files(1).name));

filetype = strcat('uint',int2str(info.BitDepth));
img = zeros(info.Width, info.Height, uint8(numel(files)/channels), filetype);

% load frames and fill array
for c=1:channels
    
    i=1;
    for k=c:channels:numel(files)

        frame = imread(strcat(folder,files(k).name));
        img(:,:, i) = frame;

        i = i+1;
    end

    % set options
    options.overwrite = true;
    options.big = true;
    options.compress = 'lzw';

    % save file
    saveastiff(img, strcat('seq_img_compress_#',num2str(c),'.tif'), options);

end

% load file
% x = loadtiff('dummy.tiff');