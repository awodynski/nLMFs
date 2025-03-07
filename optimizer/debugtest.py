class debugtest:
    """
    A class to store molecular data and corresponding calculations for debugging and testing purposes.

    Attributes
    ----------
    molecules : list[str]
        List of molecule identifiers used in test calculations.

    exp_values : dict[str, float]
        Dictionary mapping test identifiers to their expected (experimental) values.

    testset_calculations : dict[str, str]
        Dictionary describing molecular calculations in symbolic format, 
        defining the reaction or calculation for each test identifier.

    """

    def __init__(self):
        self.molecules = [
            'b2', 'b', 'sih', 'si', 'h', 'ch4', 'c', 'ch'
        ]

        self.exp_values = {
            '133': 67.459,
            '7': 73.921,
            '10': 420.420,
            '17': 84.221
        }

        self.testset_calculations = {
            '133': "-1 * ['b2'] + 2 * ['b']",
            '7': "-1 * ['sih'] + 1 * ['si'] + 1 * ['h']",
            '10': "-1 * ['ch4'] + 1 * ['c'] + 4 * ['h']",
            '17': "-1 * ['ch'] + 1 * ['c'] + 1 * ['h']"
        }
