function [img, opts] = load_from_folder(path, channel, total_channels, opts, scale)

    clear options;
    
    % save file information
    parts = split(path, filesep);
    
    opts.filePath = strcat(join(parts(1:end-2), filesep),filesep);
    opts.fileName = parts{end-1};
    opts.fileType = '';
    
    % get tiff files in folder
    files = dir(strcat(path,"*tif*"));
    
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
    info = imfinfo(strcat(path,files(1).name));

    filetype = strcat('uint',int2str(info.BitDepth));
    MaxSampleValue = info.MaxSampleValue;
    img = zeros(info.Width*scale, info.Height*scale, uint8(numel(files)/total_channels), filetype);

    % save image info
    opts.sz = [info.Width, info.Height, uint8(numel(files)/total_channels)];
    opts.maxValueDepth = info.BitDepth;
    opts.maxValueDat = MaxSampleValue;
    
    % load frames and fill array
    i=1;
    frame = zeros(info.Width*scale, info.Height*scale, filetype);
    for k=channel:total_channels:numel(files)

        frame = imread(strcat(path,files(k).name));
	if not(scale== 1)
		frame = imresize(frame, [info.Width*scale, info.Height*scale]);
	end
        img(:,:, i) = frame;

        i = i+1;
    end
    
    % necessary conversion
    img = double(img);
    
end