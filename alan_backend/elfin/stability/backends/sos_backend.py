"""
SOS Backend Module for Lyapunov Verification.

This module provides a SOSTOOLS-based backend for verifying
Lyapunov functions using Sum-of-Squares programming.
"""

import os
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import tempfile
import subprocess
import json

from alan_backend.elfin.stability.constraint_ir import (
    ConstraintIR, VerificationResult, VerificationStatus, ConstraintType
)

logger = logging.getLogger(__name__)


class SOSToolsBackend:
    """
    SOSTOOLS wrapper for SOS programming based verification.
    
    This provides an interface to SOSTOOLS for verifying polynomial
    Lyapunov functions using Sum-of-Squares programming.
    """
    
    def __init__(
        self,
        matlab_path: Optional[str] = None,
        sostools_path: Optional[str] = None,
        use_matlab_engine: bool = True,
        timeout: int = 120
    ):
        """
        Initialize the SOSTOOLS backend.
        
        Args:
            matlab_path: Path to MATLAB executable
            sostools_path: Path to SOSTOOLS installation
            use_matlab_engine: Whether to use the MATLAB Engine API
            timeout: Timeout for verification in seconds
        """
        self.matlab_path = matlab_path
        self.sostools_path = sostools_path
        self.use_matlab_engine = use_matlab_engine
        self.timeout = timeout
        self.matlab = None
        
        # Try to find paths if not provided
        if not self.matlab_path:
            self._find_matlab_path()
            
        if not self.sostools_path:
            self._find_sostools_path()
        
    def _find_matlab_path(self):
        """Find MATLAB executable on common paths."""
        common_paths = [
            r"C:\Program Files\MATLAB\R20*\bin\matlab.exe",
            r"/usr/local/MATLAB/R20*/bin/matlab",
            r"/Applications/MATLAB_R20*.app/bin/matlab"
        ]
        
        for path in common_paths:
            # In a real implementation, this would use glob
            if os.path.exists(path):
                self.matlab_path = path
                break
                
    def _find_sostools_path(self):
        """Find SOSTOOLS installation on common paths."""
        common_paths = [
            os.path.expanduser("~/Documents/MATLAB/sostools"),
            os.path.expanduser("~/matlab/sostools"),
            r"C:\Users\Public\Documents\MATLAB\sostools"
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                self.sostools_path = path
                break
    
    def _start_matlab_engine(self):
        """Start the MATLAB engine if using Engine API."""
        if not self.use_matlab_engine:
            return
            
        if self.matlab is not None:
            return
            
        try:
            import matlab.engine
            self.matlab = matlab.engine.start_matlab()
            
            # Add SOSTOOLS to path
            if self.sostools_path:
                self.matlab.addpath(self.sostools_path)
                self.matlab.addpath(os.path.join(self.sostools_path, 'multipoly'))
                
            # Check if SOSTOOLS is available
            res = self.matlab.which('sossetup')
            if not res:
                logger.error("SOSTOOLS not found in MATLAB path")
                self.matlab = None
                
        except ImportError:
            logger.error("MATLAB Engine API not available. Install matlab.engine package.")
            self.use_matlab_engine = False
            
        except Exception as e:
            logger.error(f"Error starting MATLAB: {e}")
            self.matlab = None
    
    def _generate_matlab_script(
        self,
        constraint: ConstraintIR,
        workdir: str
    ) -> str:
        """
        Generate MATLAB script for SOS verification.
        
        Args:
            constraint: Constraint to verify
            workdir: Working directory for script and results
            
        Returns:
            Path to generated script
        """
        if constraint.constraint_type == ConstraintType.POSITIVE:
            template = self._generate_pd_template(constraint)
        elif constraint.constraint_type == ConstraintType.NEGATIVE:
            template = self._generate_decreasing_template(constraint)
        else:
            raise ValueError(f"Unsupported constraint type: {constraint.constraint_type}")
            
        script_path = os.path.join(workdir, f"sos_verify_{constraint.id}.m")
        result_path = os.path.join(workdir, f"sos_result_{constraint.id}.json")
        
        # Add result output
        template += f"""
% Save result to JSON
result = struct();
result.status = status;
result.verification_time = verification_time;

if exist('decomposition', 'var')
    result.certificate = decomposition;
end

if exist('counterexample', 'var')
    result.counterexample = counterexample;
end

result_json = jsonencode(result);
fid = fopen('{result_path}', 'w');
fprintf(fid, '%s', result_json);
fclose(fid);
"""
        
        # Write script
        with open(script_path, 'w') as f:
            f.write(template)
            
        return script_path
    
    def _generate_pd_template(self, constraint: ConstraintIR) -> str:
        """
        Generate MATLAB script for positive definiteness verification.
        
        Args:
            constraint: Constraint to verify
            
        Returns:
            MATLAB script as string
        """
        # Extract matrix from context if available
        q_matrix = constraint.context.get("q_matrix")
        dim = constraint.context.get("dimension", len(constraint.variables))
        
        if q_matrix is not None:
            # Direct matrix check
            template = f"""
% Positive definiteness verification for {constraint.id}
disp('Verifying positive definiteness...');
start_time = tic;

% Define Q matrix
Q = {self._matrix_to_matlab(q_matrix)};

% Check eigenvalues
eigenvalues = eig(Q);
min_eigenvalue = min(eigenvalues);

if min_eigenvalue > 0
    status = 'VERIFIED';
    decomposition = struct('type', 'eigenvalue', 'min_eigenvalue', min_eigenvalue);
    disp('Verified: Q is positive definite');
else
    status = 'REFUTED';
    % Find eigenvector for counterexample
    [V, D] = eig(Q);
    [~, idx] = min(diag(D));
    counterexample = V(:, idx);
    disp('Refuted: Q is not positive definite');
end

verification_time = toc(start_time);
disp(['Verification time: ', num2str(verification_time), ' seconds']);
"""
        else:
            # Use SOS programming
            template = f"""
% Positive definiteness verification for {constraint.id}
disp('Verifying positive definiteness using SOS...');
start_time = tic;

% Initialize SOS program
dim = {dim};
prog = sosprogram(dim);

% Define variables
[prog, vars] = sospolyvar(prog, monomials(dim, 0:2));

% Extract expression
expr = {self._parse_expression(constraint.expression)};

% Add SOS constraint
prog = sosineq(prog, expr);

% Solve
prog = sossolve(prog);

% Check result
if ~prog.feasible
    status = 'REFUTED';
    disp('Refuted: Expression is not SOS');
else
    status = 'VERIFIED';
    decomposition = struct('type', 'sos', 'decomposition', char(sosgetsol(prog, expr)));
    disp('Verified: Expression is SOS');
end

verification_time = toc(start_time);
disp(['Verification time: ', num2str(verification_time), ' seconds']);
"""
        return template
    
    def _generate_decreasing_template(self, constraint: ConstraintIR) -> str:
        """
        Generate MATLAB script for decreasing condition verification.
        
        Args:
            constraint: Constraint to verify
            
        Returns:
            MATLAB script as string
        """
        dim = constraint.context.get("dimension", len(constraint.variables))
        dynamics_function = constraint.context.get("dynamics_function", "")
        
        template = f"""
% Decreasing condition verification for {constraint.id}
disp('Verifying decreasing condition using SOS...');
start_time = tic;

% Initialize SOS program
dim = {dim};
prog = sosprogram(dim);

% Define variables
[prog, x] = sospolyvar(prog, monomials(dim, 1));

% Define dynamics
{dynamics_function}

% Define Lyapunov function
{self._parse_expression(constraint.context.get("lyapunov_function", ""))}

% Compute Lie derivative
lie_derivative = jacobian(V, x) * f(x);

% Add SOS constraint: -lie_derivative is SOS
prog = sosineq(prog, -lie_derivative);

% Solve
prog = sossolve(prog);

% Check result
if ~prog.feasible
    status = 'REFUTED';
    disp('Refuted: -lie_derivative is not SOS');
else
    status = 'VERIFIED';
    decomposition = struct('type', 'sos', 'decomposition', char(sosgetsol(prog, -lie_derivative)));
    disp('Verified: -lie_derivative is SOS');
end

verification_time = toc(start_time);
disp(['Verification time: ', num2str(verification_time), ' seconds']);
"""
        return template
    
    def _matrix_to_matlab(self, matrix: Union[List[List[float]], np.ndarray]) -> str:
        """
        Convert matrix to MATLAB syntax.
        
        Args:
            matrix: Matrix as 2D list or numpy array
            
        Returns:
            MATLAB matrix string
        """
        if isinstance(matrix, np.ndarray):
            matrix = matrix.tolist()
            
        rows = []
        for row in matrix:
            rows.append("[" + ", ".join(str(x) for x in row) + "]")
            
        return "[" + "; ".join(rows) + "]"
    
    def _parse_expression(self, expression: str) -> str:
        """
        Parse constraint expression to MATLAB syntax.
        
        This is a simplified implementation. A full implementation would
        parse the constraint expression properly.
        
        Args:
            expression: Constraint expression
            
        Returns:
            MATLAB expression
        """
        # For now, just return a placeholder
        # In a real implementation, this would parse the expression
        return "x' * Q * x"
    
    def verify(self, constraint: ConstraintIR) -> VerificationResult:
        """
        Verify a constraint using SOSTOOLS.
        
        Args:
            constraint: Constraint to verify
            
        Returns:
            Verification result
        """
        start_time = time.time()
        
        # Create temp directory for scripts/results
        with tempfile.TemporaryDirectory() as workdir:
            script_path = self._generate_matlab_script(constraint, workdir)
            result_path = os.path.join(workdir, f"sos_result_{constraint.id}.json")
            
            if self.use_matlab_engine and self.matlab:
                # Use MATLAB Engine API
                self._start_matlab_engine()
                if not self.matlab:
                    return self._create_error_result(constraint, start_time, 
                                                   "Failed to start MATLAB")
                
                try:
                    self.matlab.run(script_path, nargout=0)
                except Exception as e:
                    return self._create_error_result(constraint, start_time, 
                                                   f"MATLAB execution error: {e}")
            else:
                # Use subprocess
                cmd = [
                    self.matlab_path, 
                    "-batch", 
                    f"run('{script_path}')"
                ]
                
                try:
                    subprocess.run(
                        cmd, 
                        check=True, 
                        timeout=self.timeout,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                except subprocess.TimeoutExpired:
                    return self._create_error_result(constraint, start_time, 
                                                   "Verification timeout")
                except subprocess.CalledProcessError as e:
                    return self._create_error_result(constraint, start_time, 
                                                   f"MATLAB process error: {e}")
            
            # Parse result
            if os.path.exists(result_path):
                try:
                    with open(result_path, 'r') as f:
                        result_data = json.load(f)
                        
                    status_str = result_data.get("status", "ERROR")
                    status = {
                        "VERIFIED": VerificationStatus.VERIFIED,
                        "REFUTED": VerificationStatus.REFUTED,
                        "UNKNOWN": VerificationStatus.UNKNOWN
                    }.get(status_str, VerificationStatus.ERROR)
                    
                    verification_time = result_data.get("verification_time", 
                                                     time.time() - start_time)
                    
                    certificate = result_data.get("certificate")
                    counterexample = result_data.get("counterexample")
                    
                    return VerificationResult(
                        constraint_id=constraint.id,
                        status=status,
                        proof_hash=constraint.compute_hash(),
                        verification_time=verification_time,
                        certificate=certificate,
                        counterexample=counterexample,
                        solver_info={"solver": "sostools"}
                    )
                    
                except Exception as e:
                    return self._create_error_result(constraint, start_time, 
                                                   f"Error parsing result: {e}")
            else:
                return self._create_error_result(constraint, start_time, 
                                               "No result file generated")
    
    def _create_error_result(
        self,
        constraint: ConstraintIR,
        start_time: float,
        error_message: str
    ) -> VerificationResult:
        """
        Create an error verification result.
        
        Args:
            constraint: Constraint being verified
            start_time: Start time of verification
            error_message: Error message
            
        Returns:
            Error verification result
        """
        logger.error(f"Verification error for {constraint.id}: {error_message}")
        
        return VerificationResult(
            constraint_id=constraint.id,
            status=VerificationStatus.ERROR,
            proof_hash=constraint.compute_hash(),
            verification_time=time.time() - start_time,
            solver_info={
                "solver": "sostools",
                "error": error_message
            }
        )


class SOSVerifier:
    """
    Simplified SOS verifier with direct computation.
    
    This provides a pure Python implementation for verifying
    quadratic Lyapunov functions without requiring MATLAB.
    """
    
    def verify_pd(
        self,
        q_matrix: np.ndarray,
        variables: Optional[List[str]] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify positive definiteness of a quadratic form V(x) = x^T Q x.
        
        Args:
            q_matrix: Q matrix defining the quadratic form
            variables: Optional variable names (for documentation)
            
        Returns:
            Tuple of (success, certificate)
        """
        start_time = time.time()
        
        # Check symmetric part
        q_symmetric = (q_matrix + q_matrix.T) / 2
        
        try:
            # Check eigenvalues
            eigenvalues = np.linalg.eigvalsh(q_symmetric)
            min_eigenvalue = np.min(eigenvalues)
            
            success = min_eigenvalue > 0
            
            if success:
                # Generate certificate
                certificate = {
                    "type": "eigenvalue",
                    "min_eigenvalue": float(min_eigenvalue),
                    "eigenvalues": eigenvalues.tolist(),
                }
            else:
                # Find eigenvector for counterexample
                eigenvalues, eigenvectors = np.linalg.eigh(q_symmetric)
                min_idx = np.argmin(eigenvalues)
                counterexample = eigenvectors[:, min_idx]
                
                certificate = {
                    "type": "counterexample",
                    "counterexample": counterexample.tolist(),
                    "min_eigenvalue": float(min_eigenvalue)
                }
                
            return success, {
                "certificate": certificate,
                "verification_time": time.time() - start_time
            }
            
        except Exception as e:
            return False, {
                "error": str(e),
                "verification_time": time.time() - start_time
            }
    
    def verify_decreasing(
        self,
        q_matrix: np.ndarray,
        dynamics_matrix: np.ndarray,
        variables: Optional[List[str]] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify decreasing condition for a linear system.
        
        For a linear system dx/dt = Ax and V(x) = x^T Q x,
        the decreasing condition is A^T Q + Q A negative definite.
        
        Args:
            q_matrix: Q matrix defining the quadratic form
            dynamics_matrix: A matrix defining the linear dynamics
            variables: Optional variable names (for documentation)
            
        Returns:
            Tuple of (success, certificate)
        """
        start_time = time.time()
        
        try:
            # Compute Lyapunov equation result: A^T Q + Q A
            lyap_eq = dynamics_matrix.T @ q_matrix + q_matrix @ dynamics_matrix
            
            # Check eigenvalues of -lyap_eq (should be positive)
            eigenvalues = np.linalg.eigvalsh(-lyap_eq)
            min_eigenvalue = np.min(eigenvalues)
            
            success = min_eigenvalue > 0
            
            if success:
                certificate = {
                    "type": "linear_lyapunov",
                    "min_eigenvalue": float(min_eigenvalue),
                    "eigenvalues": eigenvalues.tolist(),
                }
            else:
                # Find eigenvector for counterexample
                eigenvalues, eigenvectors = np.linalg.eigh(-lyap_eq)
                min_idx = np.argmin(eigenvalues)
                counterexample = eigenvectors[:, min_idx]
                
                certificate = {
                    "type": "counterexample",
                    "counterexample": counterexample.tolist(),
                    "min_eigenvalue": float(min_eigenvalue)
                }
                
            return success, {
                "certificate": certificate,
                "verification_time": time.time() - start_time
            }
            
        except Exception as e:
            return False, {
                "error": str(e),
                "verification_time": time.time() - start_time
            }
