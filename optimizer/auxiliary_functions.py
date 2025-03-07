import numpy as np
import csv
from debugtest import debugtest
from BH76 import BH76
from W417 import W417

"""
This module provides auxiliary functions for handling testset data,
assembling features for machine learning, and exporting neural network
weights in a TurboMole-friendly format.

Functions
---------
- _unstack_array(a, axis):
    Moves the specified axis of array 'a' to axis 0.

- read_molecule(path, molecule, MP2):
    Reads grid data from .grid files (features, weights, e1e+coulomb, optionally 
    MP2 info) for a given molecule.

- select_class(type_class):
    Returns an instance of a testset class, given its name (debugtest, W417,
    BH76).

- create_testset_dict(testset_names):
    Generates a dictionary of testset instances.

- collect_data(testset_names, testset_paths, scales, logical_features, 
               MP2=False):
    Collects data from the specified testsets, merges them into feature arrays,
    and returns the resulting (x_train, y_train, x_x, x_features).

- save_weights_turbomole(model, output_path='weights2.csv'):
    Extracts neural network weights/biases and writes them to a CSV file in a
    TurboMole-friendly single-column format.
"""

def _unstack_array(a: np.ndarray, axis: int) -> np.ndarray:
    """
    Moves the specified axis of array 'a' to axis 0.

    Parameters
    ----------
    a : np.ndarray
        The input array.
    axis : int
        The axis to move to the front.

    Returns
    -------
    np.ndarray
        A new array with the selected axis moved to 0.
    """
    return np.moveaxis(a, axis, 0)

def _read_molecule(path: str, molecule: str, MP2: bool
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                              np.ndarray | None, np.ndarray | None, ]:
    """
    Reads grid data for the specified molecule from .grid files.

    The function loads:
      - features: (N,  ...), shape read from `<molecule>_features.grid`
      - weights:  (N,  ...), shape read from `<molecule>_weights.grid`
      - e_1e:     scalar energy from `<molecule>_e1e.grid` includes 1e energy +
                  2-electron Coulomb interaction
      - optional MP2 data: e_mp2opp (total energy not density)
                           e_mp2par (total energy not density),
                           if MP2=True.

    Parameters
    ----------
    path : str
        Base path to the directory containing .grid files.
    molecule : str
        The molecule name (file prefix).
    MP2 : bool
        If True, also load e_mp2opp and e_mp2par.

    Returns
    -------
    tuple
        (e_1e, features, weights, e_mp2opp, e_mp2par)
    """
    # Build the base file name
    base_name = path + molecule

    # Load features
    fname = base_name + "_features.grid"
    features = (np.loadtxt(fname, dtype=float).T).astype(np.float32)

    # Load weights
    fname = base_name + "_weights.grid"
    weights = (np.loadtxt(fname, dtype=float).T).astype(np.float32)

    # Load e_1e
    fname = base_name + "_e1e.grid"
    e_1e =  (np.loadtxt(fname, dtype=float).T).astype(np.float32)

    # If MP2 data is requested, load e_mp2opp and e_mp2par
    if MP2:
        fname = base_name + "_emp2opp.grid"
        e_mp2opp = (np.loadtxt(fname, dtype=float).T).astype(np.float32)

        fname = base_name + "_emp2par.grid"
        e_mp2par = (np.loadtxt(fname, dtype=float).T).astype(np.float32)
    else:
        e_mp2opp, e_mp2par = None, None

    return e_1e, features, weights, e_mp2opp, e_mp2par


def _select_class(type_class: str) -> object:
    """
    Returns an instance of a testset class identified by the string 'type_class'.

    Currently, only the following testset names are supported:
    - 'debugtest'
    - 'W417'
    - 'BH76'

    Parameters
    ----------
    type_class : str
        The name of the testset class to instantiate.

    Returns
    -------
    object
        An instance of the requested testset class.
    """
    available_classes = {
        "debugtest": debugtest,
        "W417": W417,
        "BH76": BH76,
    }
    if type_class not in available_classes:
        raise ValueError("Unknown testset class")
    return available_classes[type_class]()

def _create_testset_dict(testset_names: list[str]) -> dict:
    """
    Creates a dictionary of testset objects.

    Parameters
    ----------
    testset_names : list of str
        Names of the testsets to load (e.g. ["W417", "BH76", "debugtest"]).

    Returns
    -------
    dict
        A dictionary mapping testset_name -> testset instance.
    """
    testsets = {}
    for testset_name in testset_names:
        testsets[testset_name] = _select_class(testset_name)
    return testsets

def collect_data(
    testset_names: list[str],
    testset_paths: dict,
    scales: object,
    logical_features: list[bool],
    MP2: bool = False
) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray]:
    """
    Collect data from specified testsets, assemble feature arrays, 
    and return them.

    The method does the following:
    1. Loads the specified testsets ("debugtest", "W417", or "BH76").
    2. Reads molecule data (features, energies, weights, etc.) 
       via `_read_molecule()`.
    3. Slices the first 28 columns of features for each molecule.
    4. Concatenates all features into a single array.
    5. Computes additional derived quantities (like gaa, gab, gbb).
    6. Builds a final feature matrix x_train based on `logical_features`.
    7. Builds a full feature matrix x_features with all laplacian+mGGA data
       columns in alpha-beta and beta-alpha arrangement used in DFT class.

    Parameters
    ----------
    testset_names : list of str
        Names of the testsets to load.
    testset_paths : dict
        Mapping from testset name to file path, e.g. {"W417": "./path/to/W417"}.
    scales : object
        testset scaling parameters  used externally in loss function, 
        stored here for convenience.
    logical_features : list of bool
        Controls which features to include in the final x_train array.
        The order is typically [rho, gaa/gab/gbb, exx, t, lap, exx_sr].
    MP2 : bool
        If True, loads MP2 energies into e_mp2opp/e_mp2par.

    Returns
    -------
    x_train : np.ndarray
        Final stacked feature array for training.
    y_train : np.ndarray
        A dummy array of zeros. Real labels are not provided in loss function.
    x_x : dict
        A dictionary containing raw data structures: energies, features, 
        weights, etc.
    x_features : np.ndarray
        A larger feature matrix with all necessart columns in alpha-beta + 
        beta-alpha order necessary to construct total e_xc energie density
    """
    # Loads the specified testsets
    testsets = _create_testset_dict(testset_names)

    # Initialize dictionaries for collected data
    e_1e, e_xc, e_corr = {}, {}, {}
    features, weights = {}, {}
    e_mp2opp, e_mp2par = {}, {}

    # Loop over each testset, read data for each molecule
    for testset in testsets:
        (
            e_1e[testset],
            e_xc[testset],
            e_corr[testset],
            features[testset],
            weights[testset],
            e_mp2opp[testset],
            e_mp2par[testset]
        ) = {}, {}, {}, {}, {}, {}, {}

        #Reads molecule data
        for molecule in testsets[testset].molecules:
            (
                e1e_val,
                feat,
                wgt,
                mp2opp,
                mp2par
            ) = _read_molecule(
                path=testset_paths[testset],
                molecule=molecule,
                MP2=MP2
            )

            # Keep only the first 28 columns of features
            feat = feat[0:28, :]

            e_1e[testset][molecule] = e1e_val
            features[testset][molecule] = feat
            weights[testset][molecule] = wgt
            e_mp2opp[testset][molecule] = mp2opp
            e_mp2par[testset][molecule] = mp2par


    # Combine all testset features into one large array
    columns = 28  # since we restricted to first 28 columns
    x_train_full = np.empty((0, columns))
    pos_iter = 0
    pos = {}

    for testset in testsets:
        pos[testset] = {}
        for molecule in testsets[testset].molecules:
            pos[testset][molecule] = pos_iter
            x_train_full = np.concatenate(
                (x_train_full, features[testset][molecule].T)
            )
            pos_iter += weights[testset][molecule].shape[0]

    # Store references in x_x dictionary
    x_x = {
        'e_1e': e_1e,
        'features': features,
        'weights': weights,
        'testsets': testsets,
        'pos': pos,
        'scales': scales,
        'e_mp2opp': e_mp2opp,
        'e_mp2par': e_mp2par
    }

    # Unstack the combined feature array into individual variables
    # x_train_full has shape (N, 28), we unstack along axis=1
    # so each variable is shape (N,)
    # Hessian is not used here
    (ra, gax, gay, gaz, la, ta, hxxa, hxya, hxza, hyya, hyza, hzza,
     rb, gbx, gby, gbz, lb, tb, hxxb, hxyb, hxzb, hyyb, hyzb, hzzb,
     ea, eb, easr, ebsr) = (_unstack_array(x_train_full, axis=1))

    # Compute squared gradients
    gaa = gax * gax + gay * gay + gaz * gaz
    gab = gax * gbx + gay * gby + gaz * gbz
    gbb = gbx * gbx + gby * gby + gbz * gbz

    # We define blocks of features for alpha-beta
    ab_features_blocks = [
        (ra, rb),               # density
        (gaa, gab, gbb),       # grads
        (ea, eb),              # exact exchange
        (ta, tb),              # kinetic energy density
        (la, lb),              # laplacian
        (easr, ebsr)           # sr exact exchange
    ]

    # Build alpha-beta and beta-alpha in a single pass
    # LMF is symmetrized on the basis of two passes (NN(a,b)+NN(b,a))/2
    def build_feature_matrices(blocks, logical_mask):
        """Helper to build alpha-beta & beta-alpha stacked arrays based on 
        `logical_mask`."""
        ab_list = []
        ba_list = []
        for (idx, block) in enumerate(blocks):
            # If we want this block according to logical_features
            if logical_mask[idx]:
                # alpha-beta as is
                ab_list.extend(block)
                # beta-alpha is reversed
                ba_list.extend(block[::-1])

        # Combine
        ab_array = np.array(ab_list)
        ba_array = np.array(ba_list)
        return np.concatenate((ab_array, ba_array), axis=1)

    # Build x_train
    # ab_ba_array has shape (used features, 2 * N)
    logical_mask = logical_features
    ab_ba_array = build_feature_matrices(ab_features_blocks, logical_mask)
    # We want shape (2 * N, used features)
    x_train = ab_ba_array.T

    # Build x_features with all 6 blocks included 
    # i.e. pretend logical_features = [True]*6
    full_mask = [True, True, True, True, True, True]
    ab_ba_array_full = build_feature_matrices(ab_features_blocks, full_mask)
    x_features = ab_ba_array_full.T

    # Currently, we have no real labels for every gridpoint, so y_train 
    # is just zeros
    y_train = np.zeros(x_train.shape[0])

    return x_train, y_train, x_x, x_features

def save_weights_turbomole(model, output_path: str = 'weights2.csv') -> None:
    """
    Extracts model weights and biases, then writes them to a CSV file
    in a TurboMole-friendly, single-column format.

    Parameters
    ----------
    model : tf.keras.Model
        The trained neural network model containing layers with weights & biases.
    output_path : str, optional
        The filename to write the CSV data to.
    """
    weights_and_biases = [layer.get_weights() for layer in model.layers]

    # Flatten all weights in single column format
    single_column_values = []

    for layer_index, layer_weights in enumerate(weights_and_biases, start=1):
        if len(layer_weights) == 2:
            w, b = layer_weights
            # Transpose weights so columns match nodes
            w_T = w.T
            # Flatten row by row (or col by col, depends on how you want it)
            for row in w_T:
                single_column_values.extend(row)
            # Then biases
            single_column_values.extend(b)
        else:
            print(f"Layer {layer_index} does not contain weights or biases.")

    # Write everything into one CSV column
    with open(output_path, mode='w', encoding='utf-8', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for val in single_column_values:
            writer.writerow([f"{val:24.17E}"])


