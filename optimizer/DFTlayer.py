import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from dft import make_exc_density_calc


class DFTLayer(Layer):
    """
    Custom Keras layer that computes exchange-correlation (XC) density based on
    features and a neural network output using an external 'exc_density_calc' function.

    The layer defines trainable or non-trainable parameters corresponding
    to correlation scaling factors, mixing fractions, etc.
    """

    def __init__(
        self,
        scal_opp: np.ndarray,
        scal_ss: np.ndarray,
        c_opp: float,
        c_ss: float,
        nlx: float,
        train: bool,
        x_model: str,
        c_model: str,
        rs_model: bool,
        **kwargs
    ):
        """
        Initialize DFTLayer with the initial values of the correlation/exchange parameters.

        Parameters
        ----------
        scal_opp : np.ndarray
            Initial array of scaling coefficients for the opposite-spin correlation term.
        scal_ss : np.ndarray
            Initial array of scaling coefficients for the same-spin correlation term.
        c_opp : float
            Opposite-spin correlation parameter.
        c_ss : float
            Same-spin correlation parameter.
        nlx : float
            Mixing fraction for exchange (0 <= nlx <= 1).
        train : bool
            If True, parameters are trainable; otherwise they're frozen.
        x_model : str
            Exchange functional model (e.g. "PBE").
        c_model : str
            Correlation functional model (e.g. "B95" or "B97").
        rs_model : bool
            Whether to apply range-separated modifications.
        **kwargs : dict
            Additional keyword arguments for the Layer base class.
        """
        super().__init__(**kwargs)

        # Save initial values
        self.initial_scal_opp = scal_opp
        self.initial_scal_ss = scal_ss
        self.initial_c_opp = c_opp
        self.initial_c_ss = c_ss
        self.initial_nlx = nlx

        # Flags & functional choices
        self.trainable_params = train
        self.x_model = x_model
        self.c_model = c_model
        self.rs_model = rs_model

        # We'll create actual tf.Variables in build()
        self.scal_opp = None
        self.scal_ss = None
        self.c_opp = None
        self.c_ss = None
        self.nlx = None


    def build(self, input_shape):
        """
        Creates (trainable or not) tf.Variables for the various parameters 
        required by the DFT-based XC functional.

        Parameters
        ----------
        input_shape : tf.TensorShape or list of tf.TensorShape
            Shapes of the incoming tensors.
        """
        # Opposite-spin scaling factors (array)
        self.scal_opp = self.add_weight(
            name='scal_opp',
            shape=(len(self.initial_scal_opp),),
            initializer=tf.constant_initializer(self.initial_scal_opp),
            trainable=self.trainable_params
        )

        # Same-spin scaling factors (array)
        self.scal_ss = self.add_weight(
            name='scal_ss',
            shape=(len(self.initial_scal_ss),),
            initializer=tf.constant_initializer(self.initial_scal_ss),
            trainable=self.trainable_params
        )

        # Opposite-spin correlation parameter (float)
        self.c_opp = self.add_weight(
            name='c_opp',
            shape=(1,),
            initializer=tf.constant_initializer(self.initial_c_opp),
            trainable=self.trainable_params
        )

        # Same-spin correlation parameter (float)
        self.c_ss = self.add_weight(
            name='c_ss',
            shape=(1,),
            initializer=tf.constant_initializer(self.initial_c_ss),
            trainable=self.trainable_params
        )

        # Mixing fraction, here kept non-trainable
        self.nlx = self.add_weight(
            name='nlx',
            shape=(1,),
            initializer=tf.constant_initializer(self.initial_nlx),
            trainable=False
        )

        super().build(input_shape)

    @tf.function
    def call(self, inputs: tuple[tf.Tensor, tf.Tensor], **kwargs) -> tf.Tensor:
        """
        Forward pass. Computes the XC density using external 'exc_density_calc'
        and concatenates it with the original neural network output.

        Parameters
        ----------
        inputs : tuple of (tf.Tensor, tf.Tensor)
            (nn_output, features)
            - nn_output: Local Mixing Function from previous NN layer (shape: (batch, 1) typically).
            - features: Additional density-derived features required for the DFT calculation.

        Returns
        -------
        tf.Tensor
            Concatenation of [xc_density, nn_output] along axis=1.
        """
        nn_output, features = inputs

        # Use the external function for XC density calculation

        self.exc_density_calc = make_exc_density_calc(
            x_model=self.x_model, 
            c_model=self.c_model,
            rs_model=self.rs_model,  
            scal_opp=self.scal_opp,
            scal_ss=self.scal_ss,
            c_ss=self.c_ss,
            c_opp=self.c_opp,
            nlx=self.nlx
        )

        xc_density = self.exc_density_calc(
            Neural_=nn_output,
            features=features
        )
        # Concatenate XC density with the original NN output
        return tf.concat([xc_density, nn_output], axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 2)    
