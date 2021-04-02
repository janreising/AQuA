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
	try
        	h5create(out_path, loc, size(obj), 'Datatype', 'string');
        	h5write(out_path, loc, obj);
	catch
		warning("There was a problem writing a string");
		disp(loc);
		disp(obj);
		disp(class(obj));
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
