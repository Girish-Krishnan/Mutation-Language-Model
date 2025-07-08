from typing import List, Tuple


def load_dataset(path: str = "data.txt") -> List[List[Tuple[int, str]]]:
    """Load mutation sequences from a file."""
    mutations_data: List[List[Tuple[int, str]]] = []
    with open(path, "r") as data_file:
        for line in data_file:
            line = line.strip()
            if not line:
                continue
            muts_list = []
            for mut in line.split(','):
                substitution = mut[-1]
                position = int(mut[1:-1])
                muts_list.append((position, substitution))
            mutations_data.append(muts_list)
    return mutations_data
