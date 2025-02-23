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
            
            s = size(obj);
            z = size(s);
            chunk = zeros(z);
            for i=1:z(2)
                if s(i) > 100
                    c=100;
                else
                    c=s(i);
                end
                
                if c == 0
                    c=1;
                end
                
                chunk(i)=c;
                
            end
            
            h5create(out_path, loc, size(obj), 'Datatype', 'double', 'ChunkSize', chunk); 
            h5write(out_path, loc, obj);
        catch e
            warning("There was a problem writing a double");
            fprintf(1,'The identifier was:\n%s',e.identifier);
            fprintf(1,'There was an error! The message was:\n%s',e.message);
            disp(loc);
            disp(size(obj));
            disp(class(obj));
            
            temporary_name = strcat(out_path,strrep(loc,"/","-"),'.mat');
            disp(temporary_name);
            save temporary_name obj;
        end
        
        return
        
    elseif isa(obj, 'single')
        
        try
                        s = size(obj);
            z = size(s);
            chunk = zeros(z);
            for i=1:z(2)
                if s(i) > 100
                    c=100;
                else
                    c=s(i);
                end
                
                if c == 0
                    c=1;
                end
                
                chunk(i)=c;
            end
            
            h5create(out_path, loc, size(obj), 'Datatype', 'single', 'ChunkSize', chunk);
            h5write(out_path, loc, obj);
        
        catch e
           
            warning("There was a problem writing a single");
            fprintf(1,'The identifier was:\n%s',e.identifier);
            fprintf(1,'There was an error! The message was:\n%s',e.message);
            disp(loc);
            disp(size(obj));
            disp(class(obj));
            
            temporary_name = strcat(out_path,strrep(loc,"/","-"),'.mat');
            disp(temporary_name);
            save temporary_name obj;
            
        end
            
    elseif isa(obj, 'char')
        
        % convert to string since char is not supported
        if isa(obj, 'char')
            obj = convertCharsToStrings(obj);
        end
        obj = strcat(obj);
        
        try
                h5create(out_path, loc, size(obj), 'Datatype', 'string');
                h5write(out_path, loc, obj);
        catch e
            warning("There was a problem writing a string:\n");
            fprintf(1,'The identifier was:\n%s',e.identifier);
            fprintf(1,'There was an error! The message was:\n%s',e.message);
            disp(loc);
            disp(obj);
            disp(class(obj));

            temporary_name = strcat(out_path,strrep(loc,"/","-"),'.mat');
            disp(temporary_name);
            save temporary_name obj;

        end
        
    elseif isa(obj, 'string')
        
        fprintf("Can't save strings:");
        disp(obj);
        
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