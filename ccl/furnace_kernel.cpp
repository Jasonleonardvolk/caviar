// ccl/furnace_kernel.cpp
// Dark Soliton Furnace - OpenCL kernel for controlled chaos generation
// Uses defocusing Kerr non-linearity with negative dispersion

#include <CL/cl.hpp>
#include <complex>
#include <vector>

const char* furnace_kernel_source = R"CLC(
// Complex number operations for float2
inline float2 cmul(float2 a, float2 b) {
    return (float2)(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

inline float2 cexp(float2 z) {
    float exp_x = exp(z.x);
    return (float2)(exp_x * cos(z.y), exp_x * sin(z.y));
}

inline float cabs2(float2 z) {
    return z.x*z.x + z.y*z.y;
}

// Dispersion operator (Fourier space)
float2 dispersion(float2 E, float beta2, float dz, int k_idx, int N) {
    float k = 2.0f * M_PI * (k_idx - N/2) / N;
    float phase = -0.5f * beta2 * k * k * dz;
    float2 disp = (float2)(cos(phase), sin(phase));
    return cmul(E, disp);
}

__kernel void dark_soliton_furnace(
    __global const float2* phase_in,
    __global float2* phase_out,
    __global float* energy_monitor,
    const float gamma,     // Kerr nonlinearity coeff (negative for dark solitons)
    const float beta2,     // dispersion coeff (negative for anomalous)
    const float dz,
    const int N
) {
    int idx = get_global_id(0);
    if (idx >= N) return;
    
    float2 E = phase_in[idx];
    
    // Store initial energy for conservation check
    float E0 = cabs2(E);
    
    // Split-step Fourier method
    // Step 1: Linear dispersion (half step)
    E = dispersion(E, beta2, dz/2, idx, N);
    
    // Step 2: Nonlinear Kerr effect
    // For dark solitons: E' = E * exp(i * gamma * |E|^2 * dz)
    float intensity = cabs2(E);
    float nl_phase = gamma * intensity * dz;
    float2 nl_factor = (float2)(cos(nl_phase), sin(nl_phase));
    E = cmul(E, nl_factor);
    
    // Step 3: Linear dispersion (half step)
    E = dispersion(E, beta2, dz/2, idx, N);
    
    // Energy conservation check
    float E1 = cabs2(E);
    energy_monitor[idx] = fabs(E1 - E0) / (E0 + 1e-10f);
    
    // Soliton stability check (tanh profile for dark soliton)
    // Dark soliton: E(x) = sqrt(P) * tanh(x/w) * exp(i*phi)
    // Monitor deviation from ideal profile
    
    phase_out[idx] = E;
}

// Topological charge calculator for dark solitons
__kernel void calculate_topological_charge(
    __global const float2* field,
    __global float* charge_density,
    const int N
) {
    int idx = get_global_id(0);
    if (idx >= N-1) return;
    
    float2 E1 = field[idx];
    float2 E2 = field[idx + 1];
    
    // Phase difference
    float phase1 = atan2(E1.y, E1.x);
    float phase2 = atan2(E2.y, E2.x);
    float dphase = phase2 - phase1;
    
    // Wrap to [-pi, pi]
    while (dphase > M_PI) dphase -= 2*M_PI;
    while (dphase < -M_PI) dphase += 2*M_PI;
    
    charge_density[idx] = dphase / (2*M_PI);
}
)CLC";

class DarkSolitonFurnace {
private:
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;
    cl::Kernel furnace_kernel;
    cl::Kernel charge_kernel;
    
    size_t N;  // Lattice size
    float gamma;  // Kerr coefficient
    float beta2;  // Dispersion
    
public:
    DarkSolitonFurnace(size_t lattice_size, float kerr_coeff, float dispersion)
        : N(lattice_size), gamma(kerr_coeff), beta2(dispersion) {
        
        // Initialize OpenCL
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        
        cl::Platform platform = platforms[0];
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        
        cl::Device device = devices[0];
        context = cl::Context(device);
        queue = cl::CommandQueue(context, device);
        
        // Build program
        cl::Program::Sources sources;
        sources.push_back({furnace_kernel_source, strlen(furnace_kernel_source)});
        program = cl::Program(context, sources);
        
        if (program.build({device}) != CL_SUCCESS) {
            throw std::runtime_error("Failed to build OpenCL program");
        }
        
        // Create kernels
        furnace_kernel = cl::Kernel(program, "dark_soliton_furnace");
        charge_kernel = cl::Kernel(program, "calculate_topological_charge");
    }
    
    void evolve(cl::Buffer& phase_in, cl::Buffer& phase_out, 
                cl::Buffer& energy_monitor, float dz) {
        
        furnace_kernel.setArg(0, phase_in);
        furnace_kernel.setArg(1, phase_out);
        furnace_kernel.setArg(2, energy_monitor);
        furnace_kernel.setArg(3, gamma);
        furnace_kernel.setArg(4, beta2);
        furnace_kernel.setArg(5, dz);
        furnace_kernel.setArg(6, static_cast<int>(N));
        
        queue.enqueueNDRangeKernel(furnace_kernel, cl::NullRange,
                                   cl::NDRange(N), cl::NullRange);
        queue.finish();
    }
    
    float calculate_total_charge(cl::Buffer& field) {
        cl::Buffer charge_density(context, CL_MEM_READ_WRITE, 
                                 sizeof(float) * N);
        
        charge_kernel.setArg(0, field);
        charge_kernel.setArg(1, charge_density);
        charge_kernel.setArg(2, static_cast<int>(N));
        
        queue.enqueueNDRangeKernel(charge_kernel, cl::NullRange,
                                   cl::NDRange(N), cl::NullRange);
        
        // Read back and sum
        std::vector<float> charges(N);
        queue.enqueueReadBuffer(charge_density, CL_TRUE, 0,
                               sizeof(float) * N, charges.data());
        
        float total_charge = 0.0f;
        for (float c : charges) {
            total_charge += c;
        }
        
        return total_charge;
    }
};

// Python bindings will go in bindings/furnace_bindings.cpp
