"""
Unit tests for the LyapunovNet and NeuralLyapunovTrainer classes.

Tests the functionality of the neural network architecture and training
algorithm for Lyapunov functions.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from ..training.neural_lyapunov_trainer import LyapunovNet, NeuralLyapunovTrainer
from ..samplers.trajectory_sampler import TrajectorySampler


def linear_system(x):
    """Linear stable system dx/dt = -0.5*x."""
    return -0.5 * x


def test_lyapunov_net_architecture():
    """Test the LyapunovNet architecture."""
    dim = 2
    hidden_dims = (32, 32)
    alpha = 0.01
    
    # Create network
    net = LyapunovNet(dim, hidden_dims, alpha)
    
    # Check parameters
    assert net.alpha == alpha
    assert net.hidden_dims == hidden_dims
    
    # Check that the architecture has the correct structure
    # phi should have len(hidden_dims)*2 + 1 layers
    # (linear + activation for each hidden dim, plus final linear)
    phi_layers = list(net.phi.children())
    assert len(phi_layers) == len(hidden_dims) * 2 + 1
    
    # Check that input and output dimensions are correct
    first_layer = phi_layers[0]
    last_layer = phi_layers[-1]
    assert first_layer.in_features == dim
    assert last_layer.out_features == 1
    
    # Check that hidden dimensions are correct
    for i, h_dim in enumerate(hidden_dims):
        layer_idx = i * 2  # Each hidden dim has a linear + activation
        assert phi_layers[layer_idx].out_features == h_dim
        if i > 0:
            assert phi_layers[layer_idx].in_features == hidden_dims[i-1]
    
    # Check that initialization works
    # Weights should be non-zero
    for m in net.modules():
        if isinstance(m, nn.Linear):
            assert torch.sum(torch.abs(m.weight)) > 0
            if m.bias is not None:
                assert torch.sum(torch.abs(m.bias)) >= 0


def test_lyapunov_net_positive_definite():
    """Test that LyapunovNet is always positive definite."""
    dim = 3
    hidden_dims = (16, 16)
    alpha = 0.01
    
    # Create network
    net = LyapunovNet(dim, hidden_dims, alpha)
    
    # Generate random inputs
    torch.manual_seed(42)
    num_samples = 100
    x = torch.randn(num_samples, dim)
    
    # Add origin
    x = torch.cat([x, torch.zeros(1, dim)])
    
    # Evaluate the network
    with torch.no_grad():
        values = net(x)
    
    # Check that values are positive for non-zero inputs
    assert torch.all(values[:-1] > 0)
    
    # Check that value at origin is exactly zero
    assert torch.abs(values[-1]) < 1e-6


def test_lyapunov_net_origin_gradient():
    """Test that gradient at origin is zero (important for positive definiteness)."""
    dim = 2
    hidden_dims = (16, 16)
    alpha = 0.01
    
    # Create network
    net = LyapunovNet(dim, hidden_dims, alpha)
    
    # Define origin
    x = torch.zeros(1, dim, requires_grad=True)
    
    # Forward pass
    y = net(x)
    
    # Compute gradient
    y.backward()
    grad = x.grad
    
    # Check that gradient has correct shape and is close to zero
    assert grad.shape == (1, dim)
    assert torch.all(torch.abs(grad) < 1e-5)


def test_compute_gradient():
    """Test the gradient computation."""
    dim = 2
    hidden_dims = (32, 32)
    alpha = 0.01
    
    # Create network
    net = LyapunovNet(dim, hidden_dims, alpha)
    
    # Generate random input
    torch.manual_seed(42)
    x = torch.randn(1, dim)
    
    # Compute gradient using the provided method
    grad1 = net.compute_gradient(x)
    
    # Compute gradient manually
    x_manual = x.detach().clone().requires_grad_(True)
    y = net(x_manual)
    y.backward()
    grad2 = x_manual.grad
    
    # Check that both methods give the same result
    assert torch.allclose(grad1, grad2)


def test_save_load_lyapunov_net(tmp_path):
    """Test saving and loading LyapunovNet."""
    dim = 2
    hidden_dims = (16, 16)
    alpha = 0.01
    
    # Create network
    net1 = LyapunovNet(dim, hidden_dims, alpha)
    
    # Generate random input
    torch.manual_seed(42)
    x = torch.randn(10, dim)
    
    # Get output before saving
    with torch.no_grad():
        y1 = net1(x)
    
    # Save network
    save_path = tmp_path / "test_lyapunov_net.pt"
    net1.save(str(save_path))
    
    # Load network
    net2 = LyapunovNet.load(str(save_path), dim)
    
    # Check that parameters are the same
    assert net2.alpha == alpha
    assert net2.hidden_dims == hidden_dims
    
    # Check that weights are the same
    for p1, p2 in zip(net1.parameters(), net2.parameters()):
        assert torch.allclose(p1, p2)
    
    # Check that outputs are the same
    with torch.no_grad():
        y2 = net2(x)
    
    assert torch.allclose(y1, y2)


def test_neural_lyapunov_trainer_initialization():
    """Test initialization of NeuralLyapunovTrainer."""
    dim = 2
    domain = (np.array([-1, -1]), np.array([1, 1]))
    batch_size = 64
    
    # Create sampler
    sampler = TrajectorySampler(linear_system, dim, domain, batch_size=batch_size)
    
    # Create network
    net = LyapunovNet(dim, (32, 32), 0.01)
    
    # Create trainer
    trainer = NeuralLyapunovTrainer(net, sampler, learning_rate=1e-3, gamma=0.1)
    
    # Check parameters
    assert trainer.gamma == 0.1
    assert trainer.sampler == sampler
    assert trainer.net == net
    assert len(trainer.history) > 0  # Should have initialized history


def test_neural_lyapunov_trainer_train_step():
    """Test a single training step of NeuralLyapunovTrainer."""
    dim = 2
    domain = (np.array([-1, -1]), np.array([1, 1]))
    batch_size = 64
    
    # Create sampler
    sampler = TrajectorySampler(linear_system, dim, domain, batch_size=batch_size)
    
    # Create network
    torch.manual_seed(42)
    net = LyapunovNet(dim, (32, 32), 0.01)
    
    # Create trainer
    trainer = NeuralLyapunovTrainer(net, sampler, learning_rate=1e-3, gamma=0.1)
    
    # Get initial parameters
    initial_params = [p.clone() for p in net.parameters()]
    
    # Execute a single training step
    metrics = trainer.train_step()
    
    # Check metrics
    assert 'loss' in metrics
    assert 'decrease_violations' in metrics
    assert 'violation_rate' in metrics
    assert metrics['loss'] >= 0
    assert 0 <= metrics['violation_rate'] <= 1
    
    # Check that parameters have changed
    current_params = list(net.parameters())
    assert len(current_params) == len(initial_params)
    assert not all(torch.allclose(p1, p2) for p1, p2 in zip(initial_params, current_params))


def test_neural_lyapunov_trainer_fit():
    """Test the fit method of NeuralLyapunovTrainer."""
    dim = 2
    domain = (np.array([-1, -1]), np.array([1, 1]))
    batch_size = 64
    
    # Create sampler
    sampler = TrajectorySampler(linear_system, dim, domain, batch_size=batch_size)
    
    # Create network
    torch.manual_seed(42)
    net = LyapunovNet(dim, (16, 16), 0.01)
    
    # Create trainer
    trainer = NeuralLyapunovTrainer(net, sampler, learning_rate=1e-3, gamma=0.1)
    
    # Execute training
    steps = 10
    history = trainer.fit(steps=steps, log_every=5)
    
    # Check history
    assert 'loss' in history
    assert 'steps' in history
    assert 'decrease_violations' in history
    assert 'time' in history
    assert len(history['loss']) == steps
    assert len(history['steps']) == steps
    assert len(history['decrease_violations']) == steps
    assert len(history['time']) == steps


def test_neural_lyapunov_trainer_evaluate():
    """Test the evaluate method of NeuralLyapunovTrainer."""
    dim = 2
    domain = (np.array([-1, -1]), np.array([1, 1]))
    batch_size = 64
    
    # Create sampler
    sampler = TrajectorySampler(linear_system, dim, domain, batch_size=batch_size)
    
    # Create network
    torch.manual_seed(42)
    net = LyapunovNet(dim, (16, 16), 0.01)
    
    # Create trainer
    trainer = NeuralLyapunovTrainer(net, sampler, learning_rate=1e-3, gamma=0.1)
    
    # Evaluate the model
    n_samples = 100
    metrics = trainer.evaluate(n_samples=n_samples)
    
    # Check metrics
    assert 'decrease_violations' in metrics
    assert 'decrease_violation_rate' in metrics
    assert 'margin_violations' in metrics
    assert 'margin_violation_rate' in metrics
    assert 'V_min' in metrics
    assert 'V_max' in metrics
    assert 'V_mean' in metrics
    assert 'Vdot_min' in metrics
    assert 'Vdot_max' in metrics
    assert 'Vdot_mean' in metrics
    
    assert 0 <= metrics['decrease_violation_rate'] <= 1
    assert 0 <= metrics['margin_violation_rate'] <= 1
    assert metrics['V_min'] >= 0  # Lyapunov function should be non-negative
    assert metrics['V_max'] >= metrics['V_min']
    assert metrics['V_mean'] >= metrics['V_min']


def test_stable_system_training():
    """Test that training on a stable system reduces Lyapunov violations."""
    dim = 2
    domain = (np.array([-1, -1]), np.array([1, 1]))
    batch_size = 64
    
    # Create sampler for stable linear system
    sampler = TrajectorySampler(linear_system, dim, domain, batch_size=batch_size)
    
    # Create network
    torch.manual_seed(42)
    net = LyapunovNet(dim, (32, 32), 0.01)
    
    # Create trainer
    trainer = NeuralLyapunovTrainer(net, sampler, learning_rate=1e-3, gamma=0.1)
    
    # Get initial evaluation
    initial_metrics = trainer.evaluate(n_samples=500)
    
    # Train the model
    trainer.fit(steps=100, log_every=50)
    
    # Get final evaluation
    final_metrics = trainer.evaluate(n_samples=500)
    
    # Check that training has improved the Lyapunov function
    # For a stable linear system, we should be able to reduce violations
    assert final_metrics['decrease_violation_rate'] <= initial_metrics['decrease_violation_rate']
