            # Weight by confidence
            weights = np.array([est['confidence'] for est in windowed_estimates])
            weights = weights / np.sum(weights)  # Normalize weights
            
            # Compute weighted average of psi estimates
            weighted_psi = np.zeros_like(windowed_estimates[0]['psi'])
            for est, weight in zip(windowed_estimates, weights):
                weighted_psi += weight * est['psi']
            
            # Compute weighted average confidence
            weighted_confidence = np.sum(weights * [est['confidence'] for est in windowed_estimates])
            
            return weighted_psi, weighted_confidence
        else:
            # Fallback to full estimate if no windowed estimates
            return psi_full, confidence_full

    def fit(self, data: np.ndarray) -> None:
        """
        Fit the Koopman estimator to data.
        
        Args:
            data: Input data with shape (n_samples, n_features)
        """
        # Store data for analysis
        self.data = data
        
        # Compute basic eigenmodes (simplified implementation)
        try:
            # Use SVD for dimensionality reduction and mode extraction
            U, s, Vt = np.linalg.svd(data, full_matrices=False)
            
            # Create eigenmodes from SVD components
            self.eigenmodes = []
            for i in range(min(5, len(s))):  # Top 5 modes
                eigenvalue = s[i] / s[0]  # Normalize by largest singular value
                eigenfunction = Vt[i]
                
                mode = KoopmanEigenMode(
                    eigenvalue=eigenvalue,
                    eigenfunction=eigenfunction,
                    mode_index=i
                )
                self.eigenmodes.append(mode)
                
        except Exception as e:
            logger.warning(f"Error in Koopman fit: {e}")
            # Create default mode if fitting fails
            self.eigenmodes = [
                KoopmanEigenMode(
                    eigenvalue=1.0,
                    eigenfunction=np.ones(data.shape[1]),
                    mode_index=0
                )
            ]
    
    def basis_function(self, x: np.ndarray) -> np.ndarray:
        """
        Apply basis function transformation.
        
        Args:
            x: Input data
            
        Returns:
            Transformed data
        """
        # Simple polynomial basis for now
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        # Return augmented features (original + squares)
        x_aug = np.hstack([x, x**2])
        return x_aug
