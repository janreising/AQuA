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
        try
            
            if prod(size(obj)) > 100*100*100
                h5create(out_path, loc, size(obj), 'Datatype', 'double', 'ChunkSize', [100 100 100]); % try to fix saving problem with double size
            else
                h5create(out_path, loc, size(obj), 'Datatype', 'double'); % try to fix saving problem with double size
            end
            
            h5write(out_path, loc, obj);
        catch
            warning("There was a problem writing a double");
            disp(loc);
            disp(size(obj));
            disp(class(obj));
        end
        
        return
        
    elseif isa(obj, 'single')
        
        if prod(size(obj)) > 100*100*100
            h5create(out_path, loc, size(obj), 'Datatype', 'single', 'ChunkSize', [100 100 100]);
        else
            h5create(out_path, loc, size(obj), 'Datatype', 'single');
        end
        	
        h5write(out_path, loc, obj);
        
    elseif isa(obj, 'char') || isa(obj, 'string')
        
        % convert to string since char is not supported
        if isa(obj, 'char')
            obj = convertCharsToStrings(obj);
        end
        obj = strcat(obj);
        
        try
                h5create(out_path, loc, size(obj), 'Datatype', 'string');
                h5write(out_path, loc, obj);
        catch
            warning("There was a problem writing a string");
            disp(loc);
            disp(obj);
            disp(class(obj));

            temp_name = [out_path,loc,'.m']
            save temp_name obj

        end
        
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
