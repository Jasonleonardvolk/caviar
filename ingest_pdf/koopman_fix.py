            # Weight by confidence
            weights = np.array([est[1] for est in windowed_estimates])
            weights = weights / np.sum(weights)  # Normalize weights
            
            # Compute weighted average of psi estimates
            weighted_psi = np.zeros_like(windowed_estimates[0][0])
            for (psi_est, conf), weight in zip(windowed_estimates, weights):
                weighted_psi += weight * psi_est
            
            # Compute weighted average confidence
            weighted_confidence = np.sum(weights * [est[1] for est in windowed_estimates])
            
            return weighted_psi, weighted_confidence
        else:
            # Fallback to full estimate if no windowed estimates
            return psi_full, confidence_full

    def fit_generator(self, data: np.ndarray) -> None:
        """
        Fit the Koopman generator (alternative to standard fit method).
        
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
