from mutations_data import *

TOTAL_BASE_PAIRS = 29903 # number of base pairs in the SARS-CoV-2 genome
NUMBER_OF_BASES = 4 # A, C, G, T

vocab_size = NUMBER_OF_BASES * TOTAL_BASE_PAIRS
base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

encode = lambda mutations: [NUMBER_OF_BASES*mut[0] + base_map[mut[1]] for mut in mutations]
decode = lambda indices: [((index // NUMBER_OF_BASES),list(base_map.keys())[index % NUMBER_OF_BASES]) for index in indices]