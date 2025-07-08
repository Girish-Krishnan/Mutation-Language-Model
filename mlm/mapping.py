TOTAL_BASE_PAIRS = 29903  # number of base pairs in the SARS-CoV-2 genome
NUMBER_OF_BASES = 4  # A, C, G, T

vocab_size = NUMBER_OF_BASES * TOTAL_BASE_PAIRS
base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


def encode(mutations):
    """Encode list of (position, base) tuples to indices."""
    return [NUMBER_OF_BASES * pos + base_map[base] for pos, base in mutations]


def decode(indices):
    """Decode indices back to (position, base) tuples."""
    bases = list(base_map.keys())
    return [(index // NUMBER_OF_BASES, bases[index % NUMBER_OF_BASES]) for index in indices]
