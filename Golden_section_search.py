# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 14:17:16 2024

@author: presvotscor
"""
import numpy as np
import copy as copy

from Normalize import normalize
from codage_model import Model_Encoder, Model_Decoder
from codage_residu import Residual_Encoder, Residual_Decoder
from Measures import get_quality, my_bin, my_inv_bin

class Encode_one_window(Model_Encoder, Residual_Encoder):
    def __init__(self, fn=50, fs=6400, N=128, Model_used={}, Residual_used={}):
        """
        Constructor for the Encode_one_window class, which combines model-based encoding 
        and residual encoding techniques.

        Parameters:
        - fn: Fundamental frequency (default: 50 Hz).
        - fs: Sampling frequency (default: 6400 Hz).
        - N: Number of samples in the window (default: 128).
        - Model_used: Dictionary of model configurations.
        - Residual_used: Dictionary of residual encoding configurations.
        """
        self.Model_used = Model_used  # Set of models used for encoding.
        self.Residual_used = Residual_used  # Set of residual encoding techniques used.

        # Initialize parent classes for model and residual encoding.
        Model_Encoder.__init__(self, fn, fs, N, False) 
        Residual_Encoder.__init__(self, N)   

        ##################### Initialization of header parameters
        
        self.nm = np.max([1, int(np.ceil(np.log2(len(Model_used))))])  # Number of bits to encode the model ID.
        self.nl = np.max([1, int(np.ceil(np.log2(len(Residual_used))))])  # Number of bits to encode the residual ID.
        self.n_kx = 5  # Number of bits to encode the normalization factor `kx`.
        self.n_kr = 5  # Number of bits to encode the normalization factor `kr`, or 0 if no residual encoding is used.

        self.nb_min_bit_theta = 2  # Minimum number of bits per parameter.
        self.nb_max_bit_theta = 10  # Maximum number of bits per parameter.
        self.n_sym = 10  # Maximum number of bits to reconstruct the residual.
        
        self.max_size_Nx = 0 # Count the number of DCT and DWT evalueted
        self.stop_GSS = 1 # Maximum size of the search interval for nx; determines when the Golden Section search stops.

        
    def ini_MMC_enc(self):
        """
        Initialize the best model and residual encoding configurations.
        This creates a deep copy of the provided `Model_used` and `Residual_used`.
        """
        self.best_Model_used = copy.deepcopy(self.Model_used)
        self.best_Residual_used = copy.deepcopy(self.Residual_used)

    def get_header(self, x, x_p):
        """
        Calculate the header information, including normalization factors (`kx` and `kx_p`)
        and initialize the parameters for each model and residual method.

        Parameters:
        - x: Current signal window.
        - x_p: Previous signal window (used for predictive models).

        Returns:
        - kx: Normalization factor for the current window.
        - kx_p: Normalization factor for the previous window.
        """
        # Normalize the current window and calculate its normalization factor `kx`.
        _, kx = normalize(x)
        if kx >= 2**self.n_kx:
            kx = 2**self.n_kx - 1
        if kx < 0:
            kx = 0

        # Normalize the previous window and calculate its normalization factor `kx_p`.
        _, kx_p = normalize(x_p[self.N:])
        if kx_p >= 2**self.n_kx:
            kx_p = 2**self.n_kx - 1
        if kx_p < 0:
            kx_p = 0

        # Loop through all models and set their header parameters.
        for id_model in self.Model_used:
            model = self.best_Model_used[id_model]
            if model["family"] == "sin":  # Sine model
                model["n nx"] = int(np.ceil(np.log2(3 * self.nb_max_bit_theta)))
                model["n kx"] = self.n_kx
                model["kx"] = kx
                model["xn"] = x * 2**(-kx)

            elif model["family"] == "poly":  # Polynomial model
                model["n nx"] = int(np.ceil(np.log2((model["order"] + 1) * self.nb_max_bit_theta)))
                model["n kx"] = self.n_kx
                model["kx"] = kx
                model["xn"] = x * 2**(-kx)

            elif model["family"] == "pred samples":  # Predictive model based on samples
                model["n nx"] = int(np.ceil(np.log2(model["order"] * self.nb_max_bit_theta)))
                model["n kx"] = self.n_kx
                model["kx"] = kx
                model["xn"] = x * 2**(-kx)
                model["xn previous"] = x_p * 2**(-kx)

            elif model["family"] == "pred para":  # Predictive model based on parameters
                id_previous_model = model["model used"]
                model["n kx"] = self.n_kx
                model["kx"] = kx
                model["xn"] = x * 2**(-kx)
                model["xn previous"] = x_p * 2**(-kx)
                if self.Model_used[id_previous_model]["family"] == "sin":
                    model["n nx"] = int(np.ceil(np.log2(3 * self.nb_max_bit_theta)))
                elif self.Model_used[id_previous_model]["family"] == "poly":
                    model["n nx"] = int(np.ceil(np.log2((self.best_Model_used[id_previous_model]["order"] + 1) * self.nb_max_bit_theta)))
                elif self.Model_used[id_previous_model]["family"] == "pred samples":
                    model["n nx"] = int(np.ceil(np.log2(self.best_Model_used[id_previous_model]["order"] * self.nb_max_bit_theta)))
                elif self.Model_used[id_previous_model]["family"] == "none":
                    model["n nx"] = 0

            elif model["family"] == "none":  # No model
                model["n nx"] = 0
                model["n kx"] = self.n_kx
                model["kx"] = kx
                model["xn"] = x * 2**(-kx)

        # Loop through all residual methods and set their header parameters.
        for id_residual in self.Residual_used:
            residual = self.best_Residual_used[id_residual]
            if residual["name"] == "DCT+BPC":
                residual["n kr"] = self.n_kr
                residual["n nr"] = self.n_sym
            elif residual["name"] == "DWT+BPC":
                residual["n kr"] = self.n_kr
                residual["n nr"] = self.n_sym
            elif residual["name"] == "none":
                residual["n kr"] = 0
                residual["n nr"] = 0
            else:
                print(f"Error: The method {id_residual} does not exist.")

        return kx, kx_p

    def get_theta(self):
        """
        Compute the estimated parameters (theta) for all models in `Model_used`.
    
        The method iterates over each model, determines its family type (e.g., sine, polynomial, predictive), 
        and calculates the appropriate theta values. The results are stored in the `theta hat` field 
        of each model configuration.
    
        Raises:
        - Prints an error message if a model family does not exist.
        """
        for id_model in self.Model_used:
            if self.best_Model_used[id_model]["family"] == "sin":
                # Calculate estimated parameters for the sine model
                theta_sin_hat = self.get_theta_sin(
                    self.best_Model_used[id_model]["xn"], 
                    self.best_Model_used[id_model]["m theta"], 
                    self.best_Model_used[id_model]["w theta"]
                )
                self.best_Model_used[id_model]["theta hat"] = theta_sin_hat
    
            elif self.best_Model_used[id_model]["family"] == "poly":
                # Calculate estimated parameters for the polynomial model
                theta_poly_hat = self.get_theta_poly(
                    self.best_Model_used[id_model]["xn"], 
                    self.best_Model_used[id_model]["m theta"], 
                    self.best_Model_used[id_model]["w theta"], 
                    self.best_Model_used[id_model]["order"]
                )
                self.best_Model_used[id_model]["theta hat"] = theta_poly_hat
    
            elif self.best_Model_used[id_model]["family"] == "pred samples":
                # Predictive model based on samples
                m_theta_pred_samples = self.get_m_theta_pred_samples(
                    self.best_Model_used[id_model]["order"], 
                    self.best_Model_used[id_model]["eta"], 
                    0, 
                    [0] * self.best_Model_used[id_model]["order"], 
                    [10] * self.best_Model_used[id_model]["order"]
                )
                self.best_Model_used[id_model]["m theta"] = m_theta_pred_samples
                
                X_pred_samples = self.get_X(
                    self.best_Model_used[id_model]["xn previous"], 
                    self.best_Model_used[id_model]["order"], 
                    self.best_Model_used[id_model]["eta"]
                )
                theta_pred_samples_hat = self.get_theta_pred_samples(
                    X_pred_samples, 
                    self.best_Model_used[id_model]["xn"], 
                    self.best_Model_used[id_model]["m theta"], 
                    self.best_Model_used[id_model]["w theta"]
                )
                
                self.best_Model_used[id_model]["X"] = X_pred_samples
                self.best_Model_used[id_model]["theta hat"] = theta_pred_samples_hat
    
            elif self.best_Model_used[id_model]["family"] == "pred para":
                # Predictive model based on parameters from a previous model
                id_previous_model = self.best_Model_used[id_model]["model used"]
                if self.Model_used[id_previous_model]["family"] == "sin":
                    theta_sin_hat = self.get_theta_sin(
                        self.best_Model_used[id_model]["xn"], 
                        self.best_Model_used[id_model]["m theta"], 
                        self.best_Model_used[id_model]["w theta"]
                    )
                    self.best_Model_used[id_model]["theta hat"] = theta_sin_hat
    
                elif self.Model_used[id_previous_model]["family"] == "poly":
                    theta_poly_hat = self.get_theta_poly(
                        self.best_Model_used[id_model]["xn"], 
                        self.best_Model_used[id_model]["m theta"], 
                        self.best_Model_used[id_model]["w theta"], 
                        self.best_Model_used[id_previous_model]["order"]
                    )
                    self.best_Model_used[id_model]["theta hat"] = theta_poly_hat 
    
                elif self.Model_used[id_previous_model]["family"] == "pred samples":
                    X_pred_samples = self.get_X(
                        self.best_Model_used[id_model]["xn previous"], 
                        self.best_Model_used[id_previous_model]["order"], 
                        self.best_Model_used[id_previous_model]["eta"]
                    )
                    theta_pred_samples_hat = self.get_theta_pred_samples(
                        X_pred_samples, 
                        self.best_Model_used[id_model]["xn"], 
                        self.best_Model_used[id_model]["m theta"], 
                        self.best_Model_used[id_model]["w theta"]
                    )
                    self.best_Model_used[id_model]["X"] = self.best_Model_used[id_previous_model]["X"]
                    self.best_Model_used[id_model]["theta hat"] = theta_pred_samples_hat
    
                elif self.best_Model_used[id_previous_model]["family"] == "none":
                    self.best_Model_used[id_model]["theta hat"] = []
    
            elif self.best_Model_used[id_model]["family"] == "none":
                # No model used
                self.best_Model_used[id_model]["theta hat"] = []
    
            else:
                # Handle unknown model family
                print(f"Error: The model {id_model} does not exist.")
    
    def enc_model(self, id_model, nx):
        """
        Encode the parameters (theta) and reconstruct the signal for a specific model.
    
        Parameters:
        - id_model: Identifier for the model to be encoded.
        - nx: Number of bits allocated for encoding the model parameters.
    
        Returns:
        - theta_tilde: Encoded parameters.
        - code_theta_tilde: Binary representation of the encoded parameters.
        - x_rec: Reconstructed signal from the model.
        """
        if self.best_Model_used[id_model]["family"] == "pred samples":
            # Encode predictive model based on samples
            theta_tilde, code_theta_tilde = self.get_theta_pred_samples_tilde(
                self.best_Model_used[id_model]["theta hat"], 
                nx, 
                self.best_Model_used[id_model]["m theta"], 
                self.best_Model_used[id_model]["w theta"]
            )
            x_rec = self.get_model_pred_samples(self.best_Model_used[id_model]["X"], *theta_tilde)
    
        elif self.best_Model_used[id_model]["family"] == "pred para":
            # Encode predictive model based on parameters of a previous model
            id_previous_model = self.best_Model_used[id_model]["model used"]
            if self.Model_used[id_previous_model]["family"] == "sin":
                theta_tilde, code_theta_tilde = self.get_theta_sin_tilde(
                    self.best_Model_used[id_model]["theta hat"], 
                    nx, 
                    self.best_Model_used[id_model]["m theta"], 
                    self.best_Model_used[id_model]["w theta"]
                )
                x_rec = self.get_model_sin(self.t, *theta_tilde)
    
            elif self.Model_used[id_previous_model]["family"] == "pred samples":
                theta_tilde, code_theta_tilde = self.get_theta_pred_samples_tilde(
                    self.best_Model_used[id_model]["theta hat"], 
                    nx, 
                    self.best_Model_used[id_model]["m theta"], 
                    self.best_Model_used[id_model]["w theta"]
                )
                x_rec = self.get_model_pred_samples(self.best_Model_used[id_model]["X"], *theta_tilde)
    
            elif self.Model_used[id_previous_model]["family"] == "poly":
                theta_tilde, code_theta_tilde = self.get_theta_poly_tilde(
                    self.best_Model_used[id_model]["theta hat"], 
                    nx, 
                    self.best_Model_used[id_model]["m theta"], 
                    self.best_Model_used[id_model]["w theta"]
                )
                x_rec = self.get_model_poly(self.t, *theta_tilde)
    
            elif self.best_Model_used[id_previous_model]["family"] == "none":
                theta_tilde = []
                code_theta_tilde = []
                x_rec = np.zeros(self.N)
    
        elif self.best_Model_used[id_model]["family"] == "sin":
            # Encode sine model
            theta_tilde, code_theta_tilde = self.get_theta_sin_tilde(
                self.best_Model_used[id_model]["theta hat"], 
                nx, 
                self.best_Model_used[id_model]["m theta"], 
                self.best_Model_used[id_model]["w theta"]
            )
            x_rec = self.get_model_sin(self.t, *theta_tilde)
    
        elif self.best_Model_used[id_model]["family"] == "poly":
            # Encode polynomial model
            theta_tilde, code_theta_tilde = self.get_theta_poly_tilde(
                self.best_Model_used[id_model]["theta hat"], 
                nx, 
                self.best_Model_used[id_model]["m theta"], 
                self.best_Model_used[id_model]["w theta"]
            )
            x_rec = self.get_model_poly(self.t, *theta_tilde)
    
        elif self.best_Model_used[id_model]["family"] == "none":
            # No model used
            theta_tilde = []
            code_theta_tilde = []
            x_rec = np.zeros(self.N)
    
        else:
            print(f"Error: The model {id_model} does not exist.")
    
        return theta_tilde, code_theta_tilde, x_rec
            
    
    def enc_residual(self, id_residual, r,metric,quality_r):
        """
        Encode the residual signal using a specified method.
    
        Parameters:
        - id_residual: Identifier for the residual encoding method.
        - r: The residual signal to be encoded.
        - n_r: Number of bits allocated for encoding the residual.
    
        Returns:
        - r_rec: Reconstructed residual signal.
        - code_r: Binary representation of the encoded residual.
        - kr: Normalization factor for the residual.
        - nb_sym: Number of symbols used for encoding the residual.
        """
      
        if self.best_Residual_used[id_residual]["name"] == "DCT+BPC":
            # Discrete Cosine Transform + Binary Plane Coding
            r_rec, code_r, kr, nb_sym = self.get_r_DCT_BPC_tilde(r,metric, quality_r,np.infty, self.n_sym)
    
        elif self.best_Residual_used[id_residual]["name"] == "DWT+BPC":
            # Discrete Wavelet Transform + Binary Plane Coding
            r_rec, code_r, kr, nb_sym = self.get_r_DWT_BPC_tilde(r,metric, quality_r,np.infty, self.n_sym)
    
        elif self.best_Residual_used[id_residual]["name"] == "none":
            # No residual encoding
            r_rec = np.zeros(self.N)
            code_r = []
            kr = 0
            nb_sym = 0
    
        else:
            # Error handling for unknown residual methods
            print(f"Error: The method {id_residual} does not exist.")
    
        return r_rec, code_r, kr, nb_sym
    
   
    def f(self, id_model, nx, metric, quality):
        """
        This function evaluates the quality of a given model and its residual by quantizing the model parameters 
        and testing different residual compression methods. It identifies the optimal combination that meets 
        the quality constraints.
    
        Parameters:
        - id_model: Identifier for the model to be encoded.
        - nx: Number of bits allocated to quantize the model parameters.
        - metric: The quality metric to be used (e.g., SNR, RMSE, MSE).
        - quality: Target quality for the encoded signal.
    
        Returns:
        - quality_model: The quality (e.g., SNR) of the quantized model.
        - quality_residual: The quality of the best residual compression method.
        - theta_tilde: Quantized model parameters.
        - code_model: Binary representation of the quantized model parameters.
        - x_model: Reconstructed signal from the quantized model parameters.
        - id_residual: Identifier of the best residual compression method.
        - kr: Key parameter for the residual compression.
        - nb_sym: Number of symbols used in residual compression.
        - code_residual: Compressed residual data.
        - x_residual: Reconstructed residual signal.
        """
        # Step 1: Quantize the model parameters and encode them using nx bits
        theta_tilde, code_model, x_model = self.enc_model(id_model, nx)
        
        # Step 2: Calculate the quality (e.g., SNR) of the quantized model
        quality_model = get_quality(self.best_Model_used[id_model]["xn"], x_model, "SNR")
        
        # Convert the target quality to the appropriate scale based on the metric
        if metric == "RMSE":
            quality_n = quality * 2**(-self.best_Model_used[id_model]["kx"])
        elif metric == "MSE":
            quality_n = quality * 2**(-2*self.best_Model_used[id_model]["kx"])
        elif metric == "SNR":
            quality_n = quality
        else:
            print("error, the metric {} does not exist".format(metric))
            return
    
        # Step 3: Compute the residual signal as the difference between the original signal and the model approximation
        r = self.best_Model_used[id_model]["xn"] - x_model
        
        # Determine the target quality for the residual based on the metric
        if metric == "SNR":
            quality_r_target = (quality_n - quality_model)
        elif metric in ["RMSE", "MSE"]:
            quality_r_target = quality_n
    
        # Initialize variables to find the best residual compression method
        nr_min = np.infty
        id_residual = 0
        x_residual = []
        code_residual = []
        kr = 0
        nb_sym = 0
        quality_residual = np.infty
    
        # Step 4: Test all available residual compression methods
        for id_residual_test in self.best_Residual_used:
            # Encode the residual using the current method
            x_residual_test, code_residual_test, kr_test, nb_sym_test = self.enc_residual(id_residual_test, r, metric, quality_r_target)
            
            # Increment a counter if certain residual methods are used (for tracking complexity)
            if self.best_Residual_used[id_residual_test]["name"] in ["DCT+BPC", "DWT+BPC"]:
                self.max_size_Nx += 1
    
            # Calculate the quality of the encoded residual
            quality_residual_test = get_quality(r, x_residual_test, metric)
            nr_test = len(code_residual_test)
    
            # Update the best method if it meets the quality constraint and uses fewer bits
            if nr_test < nr_min and quality_residual_test <= quality_r_target:
                nr_min = nr_test
                id_residual = id_residual_test
                x_residual = copy.copy(x_residual_test)
                code_residual = copy.copy(code_residual_test)
                kr = kr_test
                nb_sym = nb_sym_test
                quality_residual = quality_residual_test
    
                # Stop searching if the "none" method (no compression) is the best
                if self.best_Residual_used[id_residual_test]["name"] == "none":
                    break
    
        # Return the results: the model's quality, the residual's quality, and all related encoding information
        return quality_model, quality_residual, theta_tilde, code_model, x_model, id_residual, kr, nb_sym, code_residual, x_residual

        

    def get_n_max(self, x, metric, quality):
        """
        Determines the minimum number of bits required for residual compression methods
        to achieve a given quality level for a given input signal.
    
        Parameters:
        - x: Input signal to be compressed.
        - metric: Quality metric to consider (e.g., SNR, MSE, RMSE).
        - quality: Target quality level to meet.
    
        Returns:
        - nr_min: The minimum number of bits required for the residual compression.
        """
        
        # Step 1: Initialize the minimum bit count as infinity
        nr_min = np.infty
        
        # Step 2: Iterate through each available residual compression method
        for id_residual_test in self.best_Residual_used:
            # Encode the residual using the current method
            x_residual_test, code_residual_test, kr_test, nb_sym_test = self.enc_residual(
                id_residual_test, x, metric, quality
            )
            
            # Check if the current method is DCT+BPC or DWT+BPC and increment a counter if true
            if (
                self.best_Residual_used[id_residual_test]["name"] == "DCT+BPC"
                or self.best_Residual_used[id_residual_test]["name"] == "DWT+BPC"
            ):
                self.max_size_Nx += 1
            
            # Compute the quality metric (e.g., SNR) of the compressed residual
            quality_residual_test = get_quality(x, x_residual_test, metric)
            
            # Determine the number of bits required by this residual method
            nr_test = len(code_residual_test)
            
            # If this method achieves the target quality with fewer bits, update nr_min
            if nr_test < nr_min and quality_residual_test <= quality:
                nr_min = nr_test
                
                # If the method is "none" (no residual compression), stop testing further methods
                # as "none" is the most straightforward approach requiring the least bits
                if self.best_Residual_used[id_residual_test]["name"] == "none":
                    break
        
        # Return the minimum number of bits required
        return nr_min

    
    def get_n_min_m(self, id_model, metric, quality):
        """
        Determines the minimum number of bits required for a residual compression method
        to achieve a given quality level for a specific model.
    
        Parameters:
        - id_model: Identifier of the model to be evaluated.
        - metric: Quality metric to consider, such as SNR, MSE, or RMSE.
        - quality: Target quality level to meet.
    
        Returns:
        - nr_min: The minimum number of bits needed for the residual compression method.
        """
        
        # Step 1: Encode the model parameters using the maximum allowed bits
        _, _, x_model_tilde = self.enc_model(
            id_model,
            self.nb_max_bit_theta * len(self.best_Model_used[id_model]["w theta"])
        )
        
        # Step 2: Calculate the residual signal by subtracting the model's reconstruction
        # from the original signal and scaling it according to kx.
        x_residual = (
            self.best_Model_used[id_model]["xn"] - x_model_tilde
        ) * 2 ** (self.best_Model_used[id_model]["kx"])
        
        # Initialize the minimum number of bits for the residual as infinity.
        nr_min = np.infty
        
        # Step 3: Iterate through each residual compression method
        for id_residual_test in self.best_Residual_used:
            # Encode the residual using the current method
            x_residual_test, code_residual_test, kr_test, nb_sym_test = self.enc_residual(
                id_residual_test, x_residual, metric, quality
            )
            
            # If the current method is DCT+BPC or DWT+BPC, increment a counter
            if (
                self.best_Residual_used[id_residual_test]["name"] == "DCT+BPC"
                or self.best_Residual_used[id_residual_test]["name"] == "DWT+BPC"
            ):
                self.max_size_Nx += 1
            
            # Compute the quality (e.g., SNR) of the compressed residual
            quality_residual_test = get_quality(x_residual, x_residual_test, metric)
            
            # Check the length of the binary representation for this residual method
            nr_test = len(code_residual_test)
            
            # If this method achieves the target quality with fewer bits, update nr_min
            if nr_test < nr_min and quality_residual_test <= quality:
                nr_min = nr_test
                
                # If the method is "none" (no residual compression), stop further testing
                # since it requires the least number of bits
                if self.best_Residual_used[id_residual_test]["name"] == "none":
                    break
        
        # Return the minimum number of bits needed for the residual compression
        return nr_min

            
    def MMC_enc(self, x, x_previous, metric, quality):
        """
        Perform a Golden Section search-based Model-Residual Compression (MMC) encoding.
    
        This function employs the Golden Section search algorithm to find the optimal
        bit allocation for encoding the model parameters and the residual. It iteratively
        refines the interval of candidate bitrates to minimize the total number of bits used
        while meeting the specified quality target.
    
        Parameters:
        - x: The current input signal to encode.
        - x_previous: The previous signal window, used for predictive models.
        - metric: Quality metric (e.g., SNR or RMSE) to evaluate compression performance.
        - quality: Target quality value to be achieved.
    
        Returns:
        - Updates internal attributes to store the optimal compressed representation.
        """
        # Step 1: Initialize the encoding process and setup variables
        self.ini_MMC_enc()
    
        # Step 2: Compute scaling factors for the current and previous signals
        kx, kx_previous = self.get_header(x, x_previous)
    
        # Step 3: Estimate initial model parameters
        self.get_theta()
    
        # Step 4: Set up the Golden ratio for partitioning the search interval
        alpha = (np.sqrt(5) - 1) / 2  # Golden ratio proportion
    
        # Determine the maximum number of bits allowed based on the target quality
        n_tot =np.infty # self.get_n_max(x, metric, quality)+ self.n_kx + self.n_kr + self.nm + self.nl + self.n_sym + 1 # Placeholder, potentially computed based on target quality
    
        # Step 5: Iterate over each candidate model
        for id_model_test in self.best_Model_used:
            # Calculate the minimum bits required for the current model to reach the target quality
            n_min = self.get_n_min_m(id_model_test, metric, quality) + \
                    self.n_kx + self.n_kr + self.nm + self.nl + self.n_sym + 1
    
            # Determine the range of bitrates to test for the current model
            min_rate_per_parameter = self.nb_min_bit_theta * len(self.best_Model_used[id_model_test]["w theta"])
            max_rate_per_parameter = min(
                max(min_rate_per_parameter, n_tot - n_min),
                min(self.nb_max_bit_theta * len(self.best_Model_used[id_model_test]["w theta"]),
                    2**self.best_Model_used[id_model_test]["n nx"] - 1)
            )
    
            # Step 6: Initialize two test points within the search range using the Golden ratio
            n_xc = int(np.floor(alpha * min_rate_per_parameter + (1 - alpha) * max_rate_per_parameter))
            n_xd = min_rate_per_parameter + max_rate_per_parameter - n_xc
    
            # Step 7: Evaluate quality metrics at the initial test points
            quality_model_c, quality_residual_c, theta_tilde_c, code_model_c, x_model_c, id_residual_c, \
            kr_c, nb_sym_c, code_residual_c, x_residual_c = self.f(id_model_test, n_xc, metric, quality)
    
            if n_xc == n_xd:
                # If both points are identical, reuse the same evaluation results
                quality_model_d, quality_residual_d, theta_tilde_d, code_model_d, x_model_d, id_residual_d, \
                kr_d, nb_sym_d, code_residual_d, x_residual_d = \
                    quality_model_c, quality_residual_c, theta_tilde_c, code_model_c, \
                    x_model_c, id_residual_c, kr_c, nb_sym_c, code_residual_c, x_residual_c
            else:
                # Otherwise, evaluate the quality metrics at the second point
                quality_model_d, quality_residual_d, theta_tilde_d, code_model_d, x_model_d, id_residual_d, \
                kr_d, nb_sym_d, code_residual_d, x_residual_d = self.f(id_model_test, n_xd, metric, quality)
    
            # Compute the total number of bits used at each test point
            n_tot_c = self.nm + self.best_Model_used[id_model_test]["n kx"] + \
                      self.best_Model_used[id_model_test]["n nx"] + len(code_model_c) + self.nl + \
                      self.best_Residual_used[id_residual_c]["n kr"] + self.best_Residual_used[id_residual_c]["n nr"] + \
                      len(code_residual_c)
    
            n_tot_d = self.nm + self.best_Model_used[id_model_test]["n kx"] + \
                      self.best_Model_used[id_model_test]["n nx"] + len(code_model_d) + self.nl + \
                      self.best_Residual_used[id_residual_d]["n kr"] + self.best_Residual_used[id_residual_d]["n nr"] + \
                      len(code_residual_d)
    
            # Step 8: Begin iteratively refining the search interval
            while True:
                
                if n_tot_c < n_tot_d:
                
                    max_rate_per_parameter = n_xd
                    
                    if n_tot_c < n_tot:
                        # Update the best model and residual if this configuration is better
                        n_tot = n_tot_c
                        self.best_Model_used[id_model_test].update({
                            "theta tilde": theta_tilde_c,
                            "code model": code_model_c,
                            "x model": x_model_c,
                            "quality model": quality_model_c,
                            "nx": n_xc,
                            "id residual": id_residual_c,
                            "name residual": self.best_Residual_used[id_residual_c]['name'],
                            "x residual": x_residual_c,
                            "nb sym residual": nb_sym_c,
                            "nr": len(code_residual_c),
                            "code residual": code_residual_c
                        })
                        self.best_Residual_used[id_residual_c]["kr"] = kr_c
                        self.best_model_used = id_model_test
                        self.best_residual_used = id_residual_c
                        
                        max_rate_per_parameter = min(max_rate_per_parameter,max([min_rate_per_parameter,n_tot-n_min]))
                        
       
                    # Update test points
                    n_xd = n_xc
                    n_xc = int(np.floor(alpha * min_rate_per_parameter + (1 - alpha) * max_rate_per_parameter))
    
    
                    # Break if the interval becomes too small
                    if n_xd - n_xc <= self.stop_GSS:
                        break
                    
                    # Shift values for the next iteration
                    n_tot_d = n_tot_c
                    quality_model_c, quality_residual_c, theta_tilde_c, code_model_c, x_model_c, id_residual_c, \
                        kr_c, nb_sym_c, code_residual_c, x_residual_c = self.f(id_model_test, n_xc, metric,quality)
                    n_tot_c=self.nm+self.best_Model_used[id_model_test]["n kx"]+self.best_Model_used[id_model_test]["n nx"]+len(code_model_c)+self.nl+self.best_Residual_used[id_residual_c]["n kr"]+self.best_Residual_used[id_residual_c]["n nr"]+len(code_residual_c)
                    
                
                else:
                    # If the combined SNR at n_xd is worse, shift the interval to the left
                    min_rate_per_parameter = n_xc
                    
                    if n_tot_d < n_tot:
                        # Update the best model and residual if this configuration is better
                        n_tot = n_tot_d
                        self.best_Model_used[id_model_test].update({
                            "theta tilde": theta_tilde_d,
                            "code model": code_model_d,
                            "x model": x_model_d,
                            "quality model": quality_model_d,
                            "nx": n_xd,
                            "id residual": id_residual_d,
                            "name residual": self.best_Residual_used[id_residual_d]['name'],
                            "x residual": x_residual_d,
                            "nb sym residual": nb_sym_d,
                            "nr": len(code_residual_d),
                            "code residual": code_residual_d
                        })
                        self.best_Residual_used[id_residual_d]["kr"] = kr_d
                        self.best_model_used = id_model_test
                        self.best_residual_used = id_residual_d
                        
                        max_rate_per_parameter = min(max_rate_per_parameter,max([min_rate_per_parameter,n_tot-n_min]))
                        
               
                    # Update test points
                    n_xc = n_xd
                    n_xd = min_rate_per_parameter + max_rate_per_parameter - n_xc
                    
                    # Break if the interval becomes too small
                    if n_xd - n_xc <= self.stop_GSS:
                        break
    
                    # Shift values for the next iteration
                    n_tot_c = n_tot_d
                    quality_model_d, quality_residual_d, theta_tilde_d, code_model_d, x_model_d, id_residual_d, \
                        kr_d, nb_sym_d, code_residual_d, x_residual_d = self.f(id_model_test, n_xd, metric,quality)
                    
                    n_tot_d=self.nm+self.best_Model_used[id_model_test]["n kx"]+self.best_Model_used[id_model_test]["n nx"]+len(code_model_d)+self.nl+self.best_Residual_used[id_residual_d]["n kr"]+self.best_Residual_used[id_residual_d]["n nr"]+len(code_residual_d)
                 
                                  
        
        # If the best model is not "none", update dependent models with the current best model
        if self.best_Model_used[self.best_model_used]["name"] != "none":
            for id_model in self.Model_used:
                # If the model uses predictive samples
                if self.Model_used[id_model]["family"] == "pred samples":
                    # Update the model used if the best model is not a "predictive parameter"
                    if self.best_Model_used[self.best_model_used]["family"] != "pred para":
                        self.Model_used[id_model]["model used"] = self.best_model_used
        
                # If the model uses predictive parameters
                elif self.Model_used[id_model]["family"] == "pred para":
                    if self.best_Model_used[self.best_model_used]["family"] != "pred para":
                        self.Model_used[id_model]["model used"] = self.best_model_used
                        # Update the model's parameters
                        self.Model_used[id_model]["m theta"] = self.best_Model_used[self.best_model_used]["theta tilde"]
                        factor = self.Model_used[id_model]["factor"]
                        self.Model_used[id_model]["w theta"] = [
                            self.Model_used[self.best_model_used]["w theta"][i] / factor
                            for i in range(len(self.best_Model_used[self.best_model_used]["w theta"]))
                        ]
                        self.Model_used[id_model]["n nx"] = self.best_Model_used[self.best_model_used]["n nx"]
        
        # Encode the signal and store encoding metadata
        self.id_model_enc = self.best_model_used  # Identifier for the encoded model
        self.id_residual_enc = self.best_residual_used  # Identifier for the encoded residual
        
        # Metadata for encoding
        self.nm_enc = self.nm  # Number of bits to encode the model index
        self.nl_enc = self.nl  # Number of bits to encode the residual index
        
        # Encoded model and residual names
        self.m_enc = self.best_Model_used[self.best_model_used]["name"]
        self.l_enc = self.best_Model_used[self.best_model_used]["name residual"]
        
        # Encoding parameters for the model and residual
        self.n_nx_enc = self.best_Model_used[self.best_model_used]["n nx"]
        self.nx_enc = self.best_Model_used[self.best_model_used]["nx"]
        
        self.n_nr_enc = self.best_Residual_used[self.best_residual_used]["n nr"]
        self.nb_sym_residual_enc = self.best_Model_used[self.best_model_used]["nb sym residual"]
        self.nr_enc = self.best_Model_used[self.best_model_used]["nr"]
        
        self.n_kx_enc = self.best_Model_used[self.best_model_used]["n kx"]
        self.n_kr_enc = self.best_Residual_used[self.best_residual_used]["n kr"]
        
        # Normalization factors for encoding
        self.kx_enc = self.best_Model_used[self.best_model_used]["kx"]
        self.kr_enc = self.best_Residual_used[self.best_residual_used]["kr"]
        
        # Reconstructed model and residual signals after encoding
        self.x_model_enc = self.best_Model_used[self.best_model_used]["x model"] * 2 ** (self.kx_enc)
        self.x_residual_enc = self.best_Model_used[self.best_model_used]["x residual"] * 2 ** (self.kx_enc)
        self.x_rec_enc = self.x_model_enc + self.x_residual_enc  # Final reconstructed signal
        
        # First stage encoding: encode the model identifier, normalization, and parameters
        code_m = my_bin(self.best_model_used, self.nm)  # Binary representation of the model index
        code_kx = my_bin(self.kx_enc, self.n_kx_enc)  # Binary representation of kx
        code_nx = my_bin(self.nx_enc, self.n_nx_enc)  # Binary representation of nx
        
        # Second stage encoding: encode residual-related parameters
        code_l = my_bin(self.best_Model_used[self.best_model_used]["id residual"], self.nl)  # Binary for residual index
        code_kr = my_bin(-self.kr_enc, self.n_kr_enc)  # Binary representation of kr
        code_nr = my_bin(self.nb_sym_residual_enc, self.n_nr_enc)  # Binary for residual symbols count
        
        # Combine all encoding stages to produce the final encoded signal
        code = (
            code_m
            + code_kx
            + code_nx
            + self.best_Model_used[self.best_model_used]["code model"]
            + code_l
            + code_kr
            + code_nr
            + self.best_Model_used[self.best_model_used]["code residual"]
        )
        
        # Store the final encoded signal
        self.code = code
        return code  # Return the complete encoded binary representation


class Decode_one_window(Model_Decoder, Residual_Decoder):
    """
    Class to decode a single window of data, implementing both model and residual decoding.
    """

    def __init__(self, fn=50, fs=6400, N=128, Model_used={}, Residual_used={}):
        """
        Initialize the decoder with model and residual configurations.
        
        Parameters:
        - fn: Base frequency of the signal.
        - fs: Sampling frequency of the signal.
        - N: Length of the signal window.
        - Model_used: Dictionary defining the set of models to be used in decoding.
        - Residual_used: Dictionary defining the set of residuals to be used in decoding.
        """
        self.Model_used = Model_used  # Set of models used for decoding.
        self.Residual_used = Residual_used  # Set of residuals used for decoding.

        # Initialize parent classes for model and residual decoding.
        Model_Decoder.__init__(self, fn, fs, N, False)
        Residual_Decoder.__init__(self, N)

        # Header parameter initialization for decoding.
        self.nm = np.max([1, int(np.ceil(np.log2(len(Model_used))))])  # Bits for model encoding.
        self.nl = np.max([1, int(np.ceil(np.log2(len(Residual_used))))])  # Bits for residual encoding.
        self.n_kx = 5  # Bits for kx encoding.
        self.n_kr = 5  # Bits for kr encoding.
        self.nb_max_bit_theta = 10  # Maximum number of bits per parameter for model coefficients.
        self.n_sym = 10  # Bits for residual symbols.

    def ini_MMC_dec(self):
        """
        Initialize the decoding process by creating deep copies of model and residual configurations.
        """
        self.best_Model_used = copy.deepcopy(self.Model_used)  # Deep copy of model configurations.
        self.best_Residual_used = copy.deepcopy(self.Residual_used)  # Deep copy of residual configurations.

    def dec_header(self):
        """
        Decode the header of the encoded signal to initialize model and residual configurations.
        """
        for id_model in self.Model_used:  # Iterate through each model in the used models set.

            if self.best_Model_used[id_model]["family"] == "pred samples":
                # Calculate the number of bits required for the model based on its order.
                self.best_Model_used[id_model]["n nx"] = int(np.ceil(np.log2(
                    self.best_Model_used[id_model]["order"] * self.nb_max_bit_theta)))
                self.best_Model_used[id_model]["n kx"] = self.n_kx

            elif self.best_Model_used[id_model]["family"] == "pred para":
                # If the model is a predictive parameter, calculate bits based on its previous model.
                id_previous_model = self.Model_used[id_model]["model used"]
                self.best_Model_used[id_model]["n kx"] = self.n_kx

                if self.Model_used[id_previous_model]["family"] == "sin":
                    self.best_Model_used[id_model]["n nx"] = int(np.ceil(np.log2(3 * self.nb_max_bit_theta)))

                elif self.Model_used[id_previous_model]["family"] == "pred samples":
                    self.best_Model_used[id_model]["n nx"] = int(np.ceil(np.log2(
                        self.Model_used[id_previous_model]["order"] * self.nb_max_bit_theta)))

                elif self.Model_used[id_previous_model]["family"] == "poly":
                    self.best_Model_used[id_model]["n nx"] = int(np.ceil(np.log2(
                        (self.Model_used[id_previous_model]["order"] + 1) * self.nb_max_bit_theta)))

                elif self.Model_used[id_previous_model]["family"] == "none":
                    self.best_Model_used[id_model]["n nx"] = 0

            elif self.best_Model_used[id_model]["family"] == "sin":
                # For sinusoidal models, calculate based on the maximum bits per parameter.
                self.best_Model_used[id_model]["n nx"] = int(np.ceil(np.log2(3 * self.nb_max_bit_theta)))
                self.best_Model_used[id_model]["n kx"] = self.n_kx

            elif self.best_Model_used[id_model]["family"] == "poly":
                # For polynomial models, calculate based on the order of the polynomial.
                self.best_Model_used[id_model]["n nx"] = int(np.ceil(np.log2(
                    (self.best_Model_used[id_model]["order"] + 1) * self.nb_max_bit_theta)))
                self.best_Model_used[id_model]["n kx"] = self.n_kx

            elif self.best_Model_used[id_model]["family"] == "none":
                # For models that don't involve encoding, set nx to 0.
                self.best_Model_used[id_model]["n nx"] = 0
                self.best_Model_used[id_model]["n kx"] = self.n_kx

        # Configure residual properties for each residual type.
        for id_residual in self.Residual_used:
            if self.best_Residual_used[id_residual]["name"] == "DCT+BPC":
                self.best_Residual_used[id_residual]["n kr"] = self.n_kr
                self.best_Residual_used[id_residual]["n nr"] = self.n_sym

            elif self.best_Residual_used[id_residual]["name"] == "DWT+BPC":
                self.best_Residual_used[id_residual]["n kr"] = self.n_kr
                self.best_Residual_used[id_residual]["n nr"] = self.n_sym

            elif self.best_Residual_used[id_residual]["name"] == "none":
                # If no residual is used, set kr and nr to 0.
                self.best_Residual_used[id_residual]["n kr"] = 0
                self.best_Residual_used[id_residual]["n nr"] = 0

            else:
                # Handle invalid or unknown residual types.
                print("error: the method {} does not exist".format(id_residual))


    def dec_model(self, id_model, code_m, x_previous_n):
        """
        Decode the model parameters and reconstruct the model signal.
    
        Parameters:
        - id_model: Identifier of the model to be decoded.
        - code_m: Encoded model parameters as a binary string.
        - x_previous_n: Normalized version of the previous signal (for predictive models).
        """
        n_x = len(code_m)  # Length of the encoded model parameters.
    
        if self.best_Model_used[id_model]["family"] == "pred samples":
            # Predictive model using sample values
            self.best_Model_used[id_model]["m theta"] = self.get_m_theta_pred_samples(
                self.best_Model_used[id_model]["order"],
                self.best_Model_used[id_model]["eta"],
                0,
                [0] * self.best_Model_used[id_model]["order"],
                [10] * self.best_Model_used[id_model]["order"]
            )
            self.best_Model_used[id_model]["X"] = self.get_X(
                x_previous_n[:2 * self.N],
                self.best_Model_used[id_model]["order"],
                self.best_Model_used[id_model]["eta"]
            )
            self.best_Model_used[id_model]["theta tilde"] = self.get_theta_pred_samples_tilde(
                code_m,
                n_x,
                self.best_Model_used[id_model]["m theta"],
                self.best_Model_used[id_model]["w theta"]
            )
            self.best_Model_used[id_model]["x model"] = self.get_model_pred_samples(
                self.best_Model_used[id_model]["X"],
                *self.best_Model_used[id_model]["theta tilde"]
            ) * 2**self.kx_dec
    
        elif self.best_Model_used[id_model]["family"] == "pred para":
            # Predictive model based on parameters
            id_previous_model = self.Model_used[id_model]["model used"]
    
            if self.Model_used[id_previous_model]["family"] == "sin":
                self.best_Model_used[id_model]["theta tilde"] = self.get_theta_sin_tilde(
                    code_m, n_x,
                    self.best_Model_used[id_model]["m theta"],
                    self.best_Model_used[id_model]["w theta"]
                )
                self.best_Model_used[id_model]["x model"] = self.get_model_sin(
                    self.t,
                    *self.best_Model_used[id_model]["theta tilde"]
                ) * 2**self.kx_dec
    
            elif self.Model_used[id_previous_model]["family"] == "pred samples":
                self.best_Model_used[id_model]["X"] = self.get_X(
                    x_previous_n[:2 * self.N],
                    self.Model_used[id_previous_model]["order"],
                    self.Model_used[id_previous_model]["eta"]
                )
                self.best_Model_used[id_model]["theta tilde"] = self.get_theta_pred_samples_tilde(
                    code_m, n_x,
                    self.best_Model_used[id_model]["m theta"],
                    self.best_Model_used[id_model]["w theta"]
                )
                self.best_Model_used[id_model]["x model"] = self.get_model_pred_samples(
                    self.best_Model_used[id_model]["X"],
                    *self.best_Model_used[id_model]["theta tilde"]
                ) * 2**self.kx_dec
    
            elif self.Model_used[id_previous_model]["family"] == "poly":
                self.best_Model_used[id_model]["theta tilde"] = self.get_theta_poly_tilde(
                    code_m, n_x,
                    self.best_Model_used[id_model]["m theta"],
                    self.best_Model_used[id_model]["w theta"]
                )
                self.best_Model_used[id_model]["x model"] = self.get_model_poly(
                    self.t,
                    *self.best_Model_used[id_model]["theta tilde"]
                ) * 2**self.kx_dec
    
            elif self.Model_used[id_previous_model]["family"] == "none":
                self.best_Model_used[id_model]["theta tilde"] = []
                self.best_Model_used[id_model]["x model"] = np.zeros(self.N)
    
        elif self.best_Model_used[id_model]["family"] == "sin":
            # Sinusoidal model
            self.best_Model_used[id_model]["theta tilde"] = self.get_theta_sin_tilde(
                code_m, n_x,
                self.best_Model_used[id_model]["m theta"],
                self.best_Model_used[id_model]["w theta"]
            )
            self.best_Model_used[id_model]["x model"] = self.get_model_sin(
                self.t,
                *self.best_Model_used[id_model]["theta tilde"]
            ) * 2**self.kx_dec
    
        elif self.best_Model_used[id_model]["family"] == "poly":
            # Polynomial model
            self.best_Model_used[id_model]["theta tilde"] = self.get_theta_poly_tilde(
                code_m, n_x,
                self.best_Model_used[id_model]["m theta"],
                self.best_Model_used[id_model]["w theta"]
            )
            self.best_Model_used[id_model]["x model"] = self.get_model_poly(
                self.t,
                *self.best_Model_used[id_model]["theta tilde"]
            ) * 2**self.kx_dec
    
        elif self.best_Model_used[id_model]["family"] == "none":
            # No model
            self.best_Model_used[id_model]["theta tilde"] = []
            self.best_Model_used[id_model]["x model"] = np.zeros(self.N)
    
    def dec_residual(self, id_residual, code_r, nb_sym_residual):
        """
        Decode the residual signal using the specified residual method.
    
        Parameters:
        - id_residual: Identifier of the residual decoding method.
        - code_r: Encoded residual as a binary string.
        - nb_sym_residual: Number of symbols used for the residual encoding.
    
        Returns:
        - Decoded residual signal.
        - Number of bits used in the residual decoding.
        """
        if self.best_Residual_used[id_residual]["name"] == "DCT+BPC":
            x_residual, nr_dec = self.get_r_DCT_BPC_tilde(code_r, nb_sym_residual, self.kr_dec)
            return x_residual * 2**(self.kx_dec), nr_dec
    
        elif self.best_Residual_used[id_residual]["name"] == "DWT+BPC":
            x_residual, nr_dec = self.get_r_DWT_BPC_tilde(code_r, nb_sym_residual, self.kr_dec)
            return x_residual * 2**(self.kx_dec), nr_dec
    
        elif self.best_Residual_used[id_residual]["name"] == "none":
            return np.zeros(self.N), 0
    
        else:
            print("Error in residual decoding: method not recognized.")
    
    
    def MMC_dec(self, code, x_previous):
        """
        Perform Multi-Model Coding (MMC) decoding.
    
        Parameters:
        - code: Encoded signal as a binary string.
        - x_previous: Previous signal used for prediction in some models.
        """
        self.ini_MMC_dec()  # Initialize model and residual configurations.
        self.dec_header()  # Decode header information.
    
        ptr = 0  # Pointer for traversing the binary code.
    
        # Decode model information.
        self.id_model_dec = int(my_inv_bin(code[ptr:ptr + self.nm]))
        ptr += self.nm
    
        self.kx_dec = int(my_inv_bin(code[ptr:ptr + self.n_kx]))
        ptr += self.n_kx
    
        self.n_nx_dec = self.best_Model_used[self.id_model_dec]["n nx"]
        self.nx_dec = int(my_inv_bin(code[ptr:ptr + self.n_nx_dec]))
        ptr += self.n_nx_dec
    
        self.dec_model(self.id_model_dec, code[ptr:ptr + self.nx_dec], x_previous * 2**(-self.kx_dec))
        ptr += self.nx_dec
    
        # Decode residual information.
        self.id_residual_dec = int(my_inv_bin(code[ptr:ptr + self.nl]))
        ptr += self.nl
    
        self.kr_dec = -int(my_inv_bin(code[ptr:ptr + self.best_Residual_used[self.id_residual_dec]["n kr"]]))
        ptr += self.best_Residual_used[self.id_residual_dec]["n kr"]
    
        self.n_nr_dec = self.best_Residual_used[self.id_residual_dec]["n nr"]
        self.nb_sym_residual = int(my_inv_bin(code[ptr:ptr + self.n_nr_dec]))
        ptr += self.n_nr_dec
    
        self.best_Model_used[self.id_model_dec]["x residual"], self.nr_dec = self.dec_residual(
            self.id_residual_dec, code[ptr:], self.nb_sym_residual)
    
        # Combine model and residual to reconstruct the signal.
        self.x_rec_dec = self.best_Model_used[self.id_model_dec]["x model"] + self.best_Model_used[self.id_model_dec]["x residual"]
    
        # Update models for predictive decoding.
        if self.best_Model_used[self.id_model_dec]["name"] != "none":
            for name in self.Model_used:
                if self.Model_used[name]["family"] == "pred samples" and self.best_Model_used[self.id_model_dec]["family"] != "pred para":
                    self.Model_used[name]["model used"] = self.id_model_dec
    
                elif self.Model_used[name]["family"] == "pred para" and self.best_Model_used[self.id_model_dec]["family"] != "pred para":
                    self.Model_used[name]["model used"] = self.id_model_dec
                    self.Model_used[name]["m theta"] = self.best_Model_used[self.id_model_dec]["theta tilde"]
                    factor = self.Model_used[name]["factor"]
                    self.Model_used[name]["w theta"] = [
                        self.Model_used[self.id_model_dec]["w theta"][i] / factor
                        for i in range(len(self.best_Model_used[self.id_model_dec]["w theta"]))
                    ]
                    self.Model_used[name]["n nx"] = self.best_Model_used[self.id_model_dec]["n nx"]
