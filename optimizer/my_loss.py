import tensorflow as tf
from tensorflow.keras.losses import Loss


class MyLoss(Loss):
    """
    A custom TensorFlow loss that computes exchange-correlation (XC) energies
    for multiple test sets and molecules, using model predictions.

    The loss internally:
    1) Splits predicted tensors to compute average XC density over 
       two passes with spin flip (exc).
    2) Integrates the XC contribution across different molecules.
    3) Computes the mean absolute error (in kcal/mol) against experimental
       values, scaled by user-defined factors.
    4) Summarizes these errors as a final loss value.

    Typically used in a scenario where y_pred contains two sets of outputs
    (exc, LMF) for the same batch.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        dict_all: dict,
        print_logical: bool,
        c_mp2opp: float | None,
        c_mp2par: float | None
    ):
        """
        Initialize MyLoss.

        Parameters
        ----------
        model : tf.keras.Model
            Reference to the compiled Keras model using this loss.
        dict_all : dict
            Dictionary storing dataset info, features, weights, etc.
        print_logical : bool
            If True, logs additional debug info via tf.print.
        c_mp2opp : float or None
            Optional scaling factor for the MP2 opposite-spin term.
        c_mp2par : float or None
            Optional scaling factor for the MP2 parallel-spin term.
        """
        super().__init__()
        self.print = print_logical
        self.dict_all = dict_all
        self.model = model
        self.c_mp2opp = c_mp2opp
        self.c_mp2par = c_mp2par

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute the loss value.

        Parameters
        ----------
        y_true : tf.Tensor
            Fake truth labels (not used in this implementation).
        y_pred : tf.Tensor
            Predicted tensor. This is expected to have shape (batch_size, 2).
            - The first column is XC energy density (exc).
            - The second column is LMF.

        Returns
        -------
        tf.Tensor
            A scalar tensor representing the final loss value.
        """
        # Split the predicted XC densities into two halves, then average
        exc1, exc2 = tf.split(y_pred[:, 0:1], num_or_size_splits=2, axis=0)
        exc = (exc1 + exc2) / 2.0

        # Split the predicted LMF into two halves, then average
        lmf1, lmf2 = tf.split(y_pred[:, 1:2], num_or_size_splits=2, axis=0)
        lmf = (lmf1 + lmf2) / 2.0  # kept for possible future use

        # Dictionary to hold computed XC energies for each testset & molecule
        efull = {}
        for testset in self.dict_all['testsets']:
            efull[testset] = {}
            for molecule in self.dict_all['testsets'][testset].molecules:
                integrated = self._exc_intergrate(
                    dict_all=self.dict_all,
                    testset=testset,
                    molecule=molecule,
                    y_pred=exc,
                    c_mp2opp=self.c_mp2opp,
                    c_mp2par=self.c_mp2par
                )
                efull[testset][molecule] = integrated

                if self.print:
                    tf.print("raw", molecule, integrated)

        # Evaluate each testset expression, compare to experiment, accumulate error
        energy_results = {}
        mae_train = {}
        for testset in self.dict_all['testsets']:
            mae_train[testset] = 0.0
            energy_results[testset] = {}

            for key in self.dict_all['testsets'][testset].testset_calculations:
                line_expr = self.dict_all['testsets'][testset] \
                    .testset_calculations[key]
                # Evaluate expression using integrated energies in efull[testset]
                energy_results[testset][key] = self.string_to_expression(
                    line_expr,
                    efull[testset]
                )

                # Convert from Hartrees to kcal/mol and compare to experiment
                pred_kcal = energy_results[testset][key] * 627.5096080305927
                exp_kcal = self.dict_all['testsets'][testset].exp_values[key]
                tmp_error = tf.math.abs(pred_kcal - exp_kcal)

                n_calcs = len(self.dict_all['testsets'][testset]
                              .testset_calculations)
                mae_train[testset] += tmp_error / n_calcs

                if self.print:
                    tf.print(key, pred_kcal)
            self.model.MAE_train[testset].assign(mae_train[testset])

        # Combine testset errors with user-defined scaling factors
        scales = self.dict_all['scales']
        scaled_losses = [mae * scale for mae, scale
                         in zip(mae_train.values(), scales)]
        # Final loss is the sum across testsets
        return tf.add_n(scaled_losses)

    def string_to_expression(self, s: str, var_dict: dict) -> tf.Tensor:
        """
        Evaluate a string expression referencing integrated energies.

        Replaces placeholders like ['MoleculeName'] with var_dict['MoleculeName'],
        then calls Python's eval().

        Parameters
        ----------
        s : str
            A string expression containing placeholders for molecule energies,
            e.g. \"var_dict['H2O'] + var_dict['CO2']\".
        var_dict : dict
            A dictionary mapping 'MoleculeName' -> tf.Tensor (integrated energy).

        Returns
        -------
        tf.Tensor
            The evaluated expression as a scalar tensor.
        """
        # Replace placeholders of the form ['someMolecule'] with var_dict['someMolecule']
        for var_name in var_dict:
            s = s.replace(f"['{var_name}']", f"var_dict['{var_name}']")
        return eval(s)

    @classmethod
    def _exc_intergrate(
        cls,
        dict_all: dict,
        testset: str,
        molecule: str,
        y_pred: tf.Tensor,
        c_mp2opp: float | None,
        c_mp2par: float | None
    ) -> tf.Tensor:
        """
        Integrate the predicted XC energy for a single molecule, optionally
        adding MP2 correlation terms if provided.

        Parameters
        ----------
        dict_all : dict
            Master dictionary containing relevant data.
        testset : str
            Name of the current testset in dict_all.
        molecule : str
            Name of the molecule whose energy is integrated.
        y_pred : tf.Tensor
            Predicted XC energy for all grid points (1D or 2D tensor).
        c_mp2opp : float or None
            Scaling factor for the MP2 opposite-spin term.
        c_mp2par : float or None
            Scaling factor for the MP2 parallel-spin term.

        Returns
        -------
        tf.Tensor
            Integrated energy for the specified molecule.
        """
        features = dict_all['features'][testset][molecule]
        w_array = dict_all['weights'][testset][molecule]
        e_1e = dict_all['e_1e'][testset][molecule]
        e_mp2par = dict_all['e_mp2par'][testset][molecule]
        e_mp2opp = dict_all['e_mp2opp'][testset][molecule]
        pos = dict_all['pos'][testset][molecule]

        # Convert weights to TF tensor
        weights = tf.convert_to_tensor(w_array, dtype=tf.float32)

        # Extract predicted XC for this molecule from the concatenated prediction
        e_xc = tf.transpose(y_pred[pos : pos + features.shape[1]])

        # Integrate over the grid
        integrated_exc =  e_1e + tf.reduce_sum(e_xc * weights)
        # Optionally add MP2 correlation terms
        if c_mp2par is not None:
            integrated_exc += c_mp2par * e_mp2par
        if c_mp2opp is not None:
            integrated_exc += c_mp2opp * e_mp2opp

        return integrated_exc
