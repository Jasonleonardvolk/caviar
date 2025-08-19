# Holographic Rendering Pipeline - Complete Shader Analysis

## Overview
This is a comprehensive holographic rendering pipeline consisting of four specialized shaders that work together to create high-quality multi-view holograms for Looking Glass displays and other holographic systems.

## Shader Components

### 1. Velocity Field Shader (`velocityField.wgsl`)
**Purpose**: Computes and visualizes phase velocity fields from wavefield data

**Key Features**:
- **Phase Gradient Computation**: Uses finite differences with proper phase unwrapping
- **Theoretical Velocity Calculation**: Derives velocity from oscillator parameters
- **Vorticity Effects**: Adds artistic vortex patterns based on oscillator phases
- **Viscous Diffusion**: Smooths velocity field using Laplacian operator
- **Particle Advection**: RK2 integration for particle movement in velocity field
- **Debug Visualization**: Magnitude mapping for flow visualization

**Technical Details**:
- Handles phase wrapping correctly using atan2 and unwrapping logic
- Blends measured and theoretical velocities based on coherence
- Implements damping and velocity clamping for stability
- Workgroup size: 8x8 for fiel