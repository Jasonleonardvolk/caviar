// furnace_bindings.cpp - Python bindings for Dark Soliton Furnace
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include "../ccl/furnace_kernel.cpp"

namespace py = pybind11;

// Python-friendly wrapper for numpy arrays
class PyDarkSolitonFurnace {
private:
    std::unique_ptr<DarkSolitonFurnace> furnace;
    size_t N;
    
public:
    PyDarkSolitonFurnace(size_t lattice_size, float kerr_coeff, float dispersion)
        : N(lattice_size) {
        furnace = std::make_unique<DarkSolitonFurnace>(lattice_size, kerr_coeff, dispersion);
    }
    
    py::array_t<std::complex<float>> evolve(
        py::array_t<std::complex<float>> phase_in,
        float dz,
        int steps = 1
    ) {
        // Ensure C-contiguous
        auto buf_in = phase_in.request();
        if (buf_in.ndim != 1 || buf_in.size != N) {
            throw std::runtime_error("Input must be 1D array of size " + std::to_string(N));
        }
        
        // Allocate output
        auto phase_out = py::array_t<std::complex<float>>(N);
        auto energy_monitor = py::array_t<float>(N);
        
        // Create OpenCL buffers (simplified - production would reuse)
        cl::Buffer cl_in(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(cl_float2) * N, buf_in.ptr);
        cl::Buffer cl_out(context, CL_MEM_WRITE_ONLY, sizeof(cl_float2) * N);
        cl::Buffer cl_energy(context, CL_MEM_WRITE_ONLY, sizeof(float) * N);
        
        // Evolve multiple steps
        for (int i = 0; i < steps; i++) {
            furnace->evolve(cl_in, cl_out, cl_energy, dz);
            std::swap(cl_in, cl_out);
        }
        
        // Read back result
        queue.enqueueReadBuffer(cl_in, CL_TRUE, 0,
                               sizeof(cl_float2) * N, phase_out.mutable_unchecked<1>().data());
        
        return phase_out;
    }
    
    float get_topological_charge(py::array_t<std::complex<float>> field) {
        auto buf = field.request();
        if (buf.ndim != 1 || buf.size != N) {
            throw std::runtime_error("Input must be 1D array of size " + std::to_string(N));
        }
        
        cl::Buffer cl_field(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           sizeof(cl_float2) * N, buf.ptr);
        
        return furnace->calculate_total_charge(cl_field);
    }
    
    // Efficiency-maximizing batch evolution
    std::tuple<py::array_t<std::complex<float>>, py::array_t<float>> 
    evolve_batch_efficient(
        py::array_t<std::complex<float>> phase_in,
        float dz,
        int steps,
        bool track_energy = false
    ) {
        auto buf_in = phase_in.request();
        size_t batch_size = buf_in.shape[0];
        
        // Pre-allocate all buffers
        std::vector<cl::Buffer> buffers_in, buffers_out, energy_monitors;
        for (size_t b = 0; b < batch_size; b++) {
            buffers_in.emplace_back(context, CL_MEM_READ_WRITE, sizeof(cl_float2) * N);
            buffers_out.emplace_back(context, CL_MEM_READ_WRITE, sizeof(cl_float2) * N);
            energy_monitors.emplace_back(context, CL_MEM_WRITE_ONLY, sizeof(float) * N);
        }
        
        // Asynchronous batch processing
        std::vector<cl::Event> events;
        for (size_t b = 0; b < batch_size; b++) {
            cl::Event write_event;
            queue.enqueueWriteBuffer(buffers_in[b], CL_FALSE, 0, sizeof(cl_float2) * N,
                                    buf_in.unchecked<2>().data(b, 0), nullptr, &write_event);
            events.push_back(write_event);
        }
        
        // Parallel evolution
        for (int step = 0; step < steps; step++) {
            std::vector<cl::Event> step_events;
            for (size_t b = 0; b < batch_size; b++) {
                furnace_kernel.setArg(0, buffers_in[b]);
                furnace_kernel.setArg(1, buffers_out[b]);
                furnace_kernel.setArg(2, energy_monitors[b]);
                
                cl::Event kernel_event;
                queue.enqueueNDRangeKernel(furnace_kernel, cl::NullRange,
                                          cl::NDRange(N), cl::NullRange,
                                          &events, &kernel_event);
                step_events.push_back(kernel_event);
            }
            events = step_events;
            std::swap(buffers_in, buffers_out);
        }
        
        // Async readback
        auto result = py::array_t<std::complex<float>>({batch_size, N});
        auto energy = py::array_t<float>({batch_size, N});
        
        for (size_t b = 0; b < batch_size; b++) {
            queue.enqueueReadBuffer(buffers_in[b], CL_FALSE, 0, sizeof(cl_float2) * N,
                                   result.mutable_unchecked<2>().data(b, 0), &events);
            if (track_energy) {
                queue.enqueueReadBuffer(energy_monitors[b], CL_FALSE, 0, sizeof(float) * N,
                                       energy.mutable_unchecked<2>().data(b, 0));
            }
        }
        
        queue.finish();
        return std::make_tuple(result, energy);
    }
};

PYBIND11_MODULE(dark_soliton_furnace, m) {
    m.doc() = "Dark Soliton Furnace - GPU-accelerated chaos generation";
    
    py::class_<PyDarkSolitonFurnace>(m, "DarkSolitonFurnace")
        .def(py::init<size_t, float, float>(),
             py::arg("lattice_size"),
             py::arg("kerr_coefficient") = -1.0,
             py::arg("dispersion") = -0.5)
        .def("evolve", &PyDarkSolitonFurnace::evolve,
             py::arg("phase_in"),
             py::arg("dz") = 0.01,
             py::arg("steps") = 1,
             "Evolve dark soliton dynamics")
        .def("get_topological_charge", &PyDarkSolitonFurnace::get_topological_charge,
             "Calculate total topological charge")
        .def("evolve_batch_efficient", &PyDarkSolitonFurnace::evolve_batch_efficient,
             py::arg("phase_in"),
             py::arg("dz") = 0.01,
             py::arg("steps") = 100,
             py::arg("track_energy") = false,
             "Efficient batch evolution for maximum throughput");
    
    // Efficiency utilities
    m.def("get_opencl_info", []() {
        py::dict info;
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        
        py::list platform_list;
        for (auto& platform : platforms) {
            py::dict p;
            p["name"] = platform.getInfo<CL_PLATFORM_NAME>();
            p["vendor"] = platform.getInfo<CL_PLATFORM_VENDOR>();
            
            std::vector<cl::Device> devices;
            platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
            
            py::list device_list;
            for (auto& device : devices) {
                py::dict d;
                d["name"] = device.getInfo<CL_DEVICE_NAME>();
                d["type"] = device.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU ? "GPU" : "CPU";
                d["compute_units"] = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
                d["memory_mb"] = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() / (1024*1024);
                device_list.append(d);
            }
            p["devices"] = device_list;
            platform_list.append(p);
        }
        info["platforms"] = platform_list;
        return info;
    });
}
