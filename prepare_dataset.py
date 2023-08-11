f = open("paths.txt", "r")
paths = f.readlines()
f.close()

output_file = open("data.txt", "w")

# For each path, split along ':' character
for path in paths:
    result = path.split(':')
    if len(result) == 2:
        mutations = result[1]
        
        if len(mutations.split(',')) > 2:
            output_file.write(mutations)