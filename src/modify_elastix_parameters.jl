"""
    function modify_parameter_file(param_in::String, param_out::String, substitutions::Dict; is_universal=false)

Modifies an elastix transform parameter file.

# Arguments
 - `param_in::String`: Path to parameter file to modify
 - `param_out::String`: Path to modified parameter file output
 - `substitutions::Dict`: Dictionary of substitutions. For each key in `substitutions`, if it is a key in the parameter file, replace its value with the value in `substitutions`.
 - `is_universal::Bool` (optional):  If set to true, instead find/replace all instances of keys in `substitutions` regardless of if it is a key in the parameter file
"""
function modify_parameter_file(param_in::String, param_out::String, substitutions::Dict; is_universal::Bool=false)
    result_str = ""
    open(param_in, "r") do f
        for line in eachline(f)
            if length(line) == 0
                result_str *= line*"\n"
                continue
            end

            line_key = split(line)[1][2:end]
            found = false
            for key in keys(substitutions)
                if is_universal
                    result_str *= replace(line, key => substitutions[key]) * "\n"
                elseif key == line_key
                    result_str *= "($key $(substitutions[key]))\n"
                    found = true
                    break
                end
            end
            if !is_universal && !found
                result_str *= line*"\n"
            end
        end
    end
    open(param_out, "w") do f
        write(f, result_str)
    end
end
