% clc;
% % 
% folder = 'C:\Users\janrei\Desktop\delete\';
% file = 'save.h5';
% % 
% % % delete file if necessary
% out_path = strcat(folder,file);
% % 
% % disp(out_path);
% % 
% % %% save data
% % tic;
% % rec_saving(out_path, datOrg, '/datOrg');
% % rec_saving(out_path, opts, '/opts');
% % rec_saving(out_path, evtLstE, '/evtLstE');
% % rec_saving(out_path, ftsLstE, '/ftsLstE');
% % rec_saving(out_path, dffMatE, '/dffMatE');
% % rec_saving(out_path, dMatE, '/dMatE');
% % rec_saving(out_path, riseLstE, '/riseLstE');
% % rec_saving(out_path, datRE, '/datRE');
% % toc;
% % 
% % % ,,,,
% % 
% % %display
% % h5disp(path);
% 
% saveToh5(out_path, datOrg, opts, evtLstE, ftsLstE, dffMatE, dMatE, riseLstE, datRE);
% 
% function r = saveToh5(varargin)
% 
%     out_path = varargin{1};
%     disp(nargin);
%     
%     if exist(out_path, 'file') == 2
%         fprintf("\nFile already exists. Choosing new name:\n");
%         
%         [folder, name, ext] = fileparts(out_path);
%         out_path = [tempname(folder),'_',name,ext];
%         
%         disp(out_path);
%         
%     end
%     
%     fprintf("\nStarting to save ... \n");
%     tic;
%     for i=2:nargin
% %         rec_saving(out_path, varargin{i}, getVarName(varargin{i}));
%         disp(i);
%         disp(getVarName(varargin{i}));
%         disp(class(varargin{i}));
%         
%     end
%     
%     
% %     rec_saving(out_path, datOrg, '/datOrg');
% %     rec_saving(out_path, opts, '/opts');
% %     rec_saving(out_path, evtLstE, '/evtLstE');
% %     rec_saving(out_path, ftsLstE, '/ftsLstE');
% %     rec_saving(out_path, dffMatE, '/dffMatE');
% %     rec_saving(out_path, dMatE, '/dMatE');
% %     rec_saving(out_path, riseLstE, '/riseLstE');
% %     rec_saving(out_path, datRE, '/datRE');
%     toc;
%     
%     r = true;
% 
% end
% 
% function out = getVarName(var)
%     out = inputname(1);
% end

function r = save_to_h5(out_path, obj, loc)
    
    if isstruct(obj)
        
        fields = fieldnames(obj);
        
        for i=1:length(fields)
            field = fields{i};
            nested_obj = obj.(field);
            
            save_to_h5(out_path, nested_obj, strcat(loc,"/",field));
        end
        
    elseif isa(obj, 'cell')
        
        for i=1:numel(obj)
            save_to_h5(out_path, obj{i}, strcat(loc, "/", num2str(i)));
        end
        
    elseif isa(obj, 'double')

        h5create(out_path, loc, size(obj), 'Datatype', 'double');
        h5write(out_path, loc, obj);

        return
        
    elseif isa(obj, 'single')
        
        h5create(out_path, loc, size(obj), 'Datatype', 'single');
        h5write(out_path, loc, obj);
        
    elseif isa(obj, 'char')
        
        % convert to string since char is not supported
        obj = convertCharsToStrings(obj);
        
        h5create(out_path, loc, size(obj), 'Datatype', 'string');
        h5write(out_path, loc, obj);
        
    elseif isinteger(obj)
        
        h5create(out_path, loc, size(obj), 'Datatype', class(obj));
        h5write(out_path, loc, obj);
        
    else
        fprintf("\nThis is an unknown type:\n");
        disp(loc);
        disp(class(obj));
    end
    
    r = true;

end