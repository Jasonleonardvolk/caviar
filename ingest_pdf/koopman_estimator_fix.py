        # If we have windowed estimates, compute weighted average
        if windowed_estimates:
            # Weight by confidence
            weights = np.array([conf for _, conf in windowed_estimates])
            weights = weights / np.sum(weights)
            
            # Weighted average of eigenfunctions
            weighted_psi = np.zeros_like(psi_full)
            for i, (psi_est, conf) in enumerate(windowed_estimates):
                weighted_psi += weights[i] * psi_est
                
            # Average confidence
            avg_confidence = np.mean([conf for _, conf in windowed_estimates])
            
            return weighted_psi, avg_confidence
        else:
            # Use full estimate
            return psi_full, confidence_full
