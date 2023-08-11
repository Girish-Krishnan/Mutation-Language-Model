data_file = open('data.txt', 'r')
data_file_lines = data_file.readlines()
data_file.close()
mutations_data = []

for line in data_file_lines:
    line = line.strip()
    muts = line.split(',')
    muts_list = []
    for mut in muts:
        substitution = mut[-1]
        position = int(mut[1:-1])
        mut = (position, substitution)
        muts_list.append(mut)

    mutations_data.append(muts_list)