#include <chrono>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <tuple>
#include <vector>
// #include <thread>
// #include <mutex>
#include <map>
#include <filesystem>
#include <algorithm>
#include "CImg/CImg.h"
#ifdef __APPLE__
    #include <OpenCL/opencl.h>
#else
    #include <CL/cl.h>
#endif

using namespace std::chrono;
namespace fs = std::filesystem;

// Mutex for console output synchronization
// std::mutex console_mutex;
// std::mutex metrics_mutex;
// std::mutex workload_mutex;

// Global program start time for workload tracking
auto t_program_start = high_resolution_clock::now();

// Structure to hold detailed metrics per image
struct ImageMetrics {
    std::string filename;
    int gpu_id;
    double total_time_ms;      // Total processing time
    double kernel_time_ms;     // Kernel execution only
    double h2d_time_ms;        // Host to Device transfer
    double d2h_time_ms;        // Device to Host transfer
    double io_time_ms;         // Image load/save time
    int width;
    int height;
};

// Structure to track GPU workload
struct GPUWorkloadMetrics {
    int gpu_id = -1;
    int images_processed = 0;
    double total_active_time_ms = 0;
    double first_image_start_ms = 0;
    double last_image_end_ms = 0;
    double total_span_ms = 0;
};

// Global storage for metrics
std::vector<ImageMetrics> all_metrics;
std::map<int, GPUWorkloadMetrics> gpu_workload;

// check error, in such a case, it exits
void cl_error(cl_int code, const char *string){
    if (code != CL_SUCCESS){
        // std::lock_guard<std::mutex> lock(console_mutex);
        printf("%d - %s\n", code, string);
        exit(-1);
    }
}

// Create gaussian mask
float * createMask(float sigma, int * maskSizePointer) {
    int maskSize = (int)ceil(3.0f*sigma);
    float * mask = new float[(maskSize*2+1)*(maskSize*2+1)];
    float sum = 0.0f;
    for(int a = -maskSize; a < maskSize+1; a++) {
        for(int b = -maskSize; b < maskSize+1; b++) {
            float temp = exp(-((float)(a*a+b*b) / (2*sigma*sigma)));
            sum += temp;
            mask[a+maskSize+(b+maskSize)*(maskSize*2+1)] = temp;
        }
    }
    // Normalize the mask
    for(int i = 0; i < (maskSize*2+1)*(maskSize*2+1); i++)
        mask[i] = mask[i] / sum;

    *maskSizePointer = maskSize;
    return mask;
}

// Structure to hold GPU context information
struct GPUContext {
    int gpu_id;
    cl_device_id device_id;
    cl_context context;
    cl_command_queue command_queue;
    cl_program program;
    cl_kernel kernel;
};

// Function to process a single image on a specific GPU
void processImage(GPUContext& gpu_ctx, const std::string& image_path, float sigma) {
    ImageMetrics metrics;
    metrics.gpu_id = gpu_ctx.gpu_id;
    metrics.filename = fs::path(image_path).filename().string();
    
    auto t_total_start = high_resolution_clock::now();
    
    // Track workload start
    {
        // std::lock_guard<std::mutex> lock(workload_mutex);
        auto& wl = gpu_workload[gpu_ctx.gpu_id];
        wl.gpu_id = gpu_ctx.gpu_id;
        if (wl.images_processed == 0) {
            wl.first_image_start_ms = duration<double, std::milli>(t_total_start - t_program_start).count();
        }
    }
    
    int err;
    
    {
        // std::lock_guard<std::mutex> lock(console_mutex);
        //std::cout << "[GPU " << gpu_ctx.gpu_id << "] Processing: " << metrics.filename << std::endl;
    }
    
    // Time image loading
    auto t_io_start = high_resolution_clock::now();
    cimg_library::CImg<unsigned char> img(image_path.c_str());
    auto t_io_end = high_resolution_clock::now();
    double io_load_ms = duration<double, std::milli>(t_io_end - t_io_start).count();
    
    metrics.width = img.width();
    metrics.height = img.height();
    int width = metrics.width;
    int height = metrics.height;
    double num_pixels = (double)width * (double)height;
    
    // Convert CImg planar RGB → interleaved RGBA (OpenCL format)
    std::vector<unsigned char> rgba_in(num_pixels*4);
    cimg_forXY(img, x, y){
        int idx = 4*(y*width + x);
        rgba_in[idx+0] = img(x,y,0);
        rgba_in[idx+1] = img(x,y,1);
        rgba_in[idx+2] = img(x,y,2);
        rgba_in[idx+3] = 255;
    }
    
    // Create Gaussian mask
    int maskSize;
    float * mask = createMask(sigma, &maskSize);
    int side = 2*maskSize + 1;
    
    // Create OpenCL image format
    cl_image_format img_format;
    img_format.image_channel_order = CL_RGBA;
    img_format.image_channel_data_type = CL_UNORM_INT8;
    
    cl_image_desc img_desc;
    memset(&img_desc, 0, sizeof(img_desc));
    img_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    img_desc.image_width = width;
    img_desc.image_height = height;
    img_desc.image_depth = 1;
    img_desc.image_array_size = 1;
    
    // Create input image buffer
    cl_mem clImage_In = clCreateImage(gpu_ctx.context, CL_MEM_READ_ONLY, &img_format, &img_desc, NULL, &err);
    cl_error(err, "Failed to create input image at device");
    
    // Write input image and measure H2D transfer
    cl_event write_event;
    size_t origin[3] = {0,0,0};
    size_t region[3] = {(size_t)width, (size_t)height, 1};
    err = clEnqueueWriteImage(gpu_ctx.command_queue, clImage_In, CL_FALSE, origin, region, 0, 0, rgba_in.data(), 0, NULL, &write_event);
    cl_error(err, "Failed to write input image to device");
    clWaitForEvents(1, &write_event);
    
    cl_ulong w_start = 0, w_end = 0;
    clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_START, sizeof(w_start), &w_start, NULL);
    clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_END, sizeof(w_end), &w_end, NULL);
    metrics.h2d_time_ms = (w_end - w_start) * 1e-6;
    
    // Create output image buffer
    cl_mem clImage_Out = clCreateImage(gpu_ctx.context, CL_MEM_WRITE_ONLY, &img_format, &img_desc, NULL, &err);
    cl_error(err, "Failed to create output image at device");
    
    // Create mask buffer
    cl_mem clMask = clCreateBuffer(gpu_ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*side*side, mask, &err);
    cl_error(err, "Failed to create mask buffer at device");
    
    // Set kernel arguments
    err = clSetKernelArg(gpu_ctx.kernel, 0, sizeof(cl_mem), &clImage_In);
    cl_error(err, "Failed to set argument 0");
    err = clSetKernelArg(gpu_ctx.kernel, 1, sizeof(cl_mem), &clImage_Out);
    cl_error(err, "Failed to set argument 1");
    err = clSetKernelArg(gpu_ctx.kernel, 2, sizeof(cl_mem), &clMask);
    cl_error(err, "Failed to set argument 2");
    err = clSetKernelArg(gpu_ctx.kernel, 3, sizeof(int), &maskSize);
    cl_error(err, "Failed to set argument 3");
    err = clSetKernelArg(gpu_ctx.kernel, 4, sizeof(int), &width);
    cl_error(err, "Failed to set argument 4");
    err = clSetKernelArg(gpu_ctx.kernel, 5, sizeof(int), &height);
    cl_error(err, "Failed to set argument 5");
    
    // Launch kernel and measure execution time
    cl_event kernel_event;
    size_t global_size[2] = { (size_t)width, (size_t)height };
    err = clEnqueueNDRangeKernel(gpu_ctx.command_queue, gpu_ctx.kernel, 2, NULL, global_size, NULL, 0, NULL, &kernel_event);
    cl_error(err, "Failed to launch kernel to the device");
    clWaitForEvents(1, &kernel_event);
    
    cl_ulong k_start = 0, k_end = 0;
    clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(k_start), &k_start, NULL);
    clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(k_end), &k_end, NULL);
    metrics.kernel_time_ms = (k_end - k_start) * 1e-6;
    
    // Read output image and measure D2H transfer
    std::vector<unsigned char> outRGBA(width*height*4);
    cl_event read_event;
    err = clEnqueueReadImage(gpu_ctx.command_queue, clImage_Out, CL_FALSE, origin, region, 0, 0, outRGBA.data(), 0, NULL, &read_event);
    cl_error(err, "Failed to read output");
    clWaitForEvents(1, &read_event);
    
    cl_ulong r_start = 0, r_end = 0;
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(r_start), &r_start, NULL);
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(r_end), &r_end, NULL);
    metrics.d2h_time_ms = (r_end - r_start) * 1e-6;
    
    // Save output and measure I/O time
    cimg_library::CImg<unsigned char> outImg(width, height, 1, 3);
    cimg_forXY(outImg, x, y) {
        int i = 4*(y*width + x);
        outImg(x,y,0) = outRGBA[i+0];
        outImg(x,y,1) = outRGBA[i+1];
        outImg(x,y,2) = outRGBA[i+2];
    }
    
    // Create results directory if it doesn't exist
    std::string results_dir = "results";
    if (!fs::exists(results_dir)) {
        fs::create_directory(results_dir);
    }
    
    auto t_save_start = high_resolution_clock::now();
    std::string result = results_dir + "/result_gpu" + std::to_string(gpu_ctx.gpu_id) + "_" + metrics.filename;
    outImg.save(result.c_str());
    auto t_save_end = high_resolution_clock::now();
    double io_save_ms = duration<double, std::milli>(t_save_end - t_save_start).count();
    
    metrics.io_time_ms = io_load_ms + io_save_ms;
    
    // Calculate total time
    auto t_total_end = high_resolution_clock::now();
    metrics.total_time_ms = duration<double, std::milli>(t_total_end - t_total_start).count();
    
    // Store metrics thread-safely
    {
        // std::lock_guard<std::mutex> lock(metrics_mutex);
        all_metrics.push_back(metrics);
    }
    
    // Update workload tracking
    {
        // std::lock_guard<std::mutex> lock(workload_mutex);
        auto& wl = gpu_workload[gpu_ctx.gpu_id];
        wl.images_processed++;
        wl.total_active_time_ms += metrics.total_time_ms;
        wl.last_image_end_ms = duration<double, std::milli>(t_total_end - t_program_start).count();
        wl.total_span_ms = wl.last_image_end_ms - wl.first_image_start_ms;
    }
    
    {
        // std::lock_guard<std::mutex> lock(console_mutex);
        //std::cout << "[GPU " << gpu_ctx.gpu_id << "] Completed: " << metrics.filename 
                  //<< " | Total: " << metrics.total_time_ms << " ms | Kernel: " << metrics.kernel_time_ms << " ms" << std::endl;
    }
    
    // Cleanup
    clReleaseEvent(kernel_event);
    clReleaseEvent(write_event);
    clReleaseEvent(read_event);
    clReleaseMemObject(clMask);
    clReleaseMemObject(clImage_Out);
    clReleaseMemObject(clImage_In);
    delete[] mask;
}

// Export metrics to CSV
void exportToCSV(const std::vector<ImageMetrics>& metrics, const std::string& filename) {
    std::ofstream csv(filename);
    csv << "filename,gpu_id,width,height,total_ms,kernel_ms,h2d_ms,d2h_ms,io_ms\n";
    for (const auto& m : metrics) {
        csv << m.filename << "," << m.gpu_id << "," << m.width << "," << m.height << ","
            << m.total_time_ms << "," << m.kernel_time_ms << "," << m.h2d_time_ms << ","
            << m.d2h_time_ms << "," << m.io_time_ms << "\n";
    }
    std::cout << "\nMetrics exported to: " << filename << std::endl;
}

// Generate detailed performance report
void generatePerformanceReport(const std::vector<ImageMetrics>& metrics, double total_program_ms) {
    if (metrics.empty()) return;
    
    // Separate by GPU
    std::vector<ImageMetrics> gpu0_metrics, gpu1_metrics;
    for (const auto& m : metrics) {
        if (m.gpu_id == 0) gpu0_metrics.push_back(m);
        // else gpu1_metrics.push_back(m);
    }
    
    auto calcStats = [](const std::vector<ImageMetrics>& m, const std::string& name) {
        if (m.empty()) return;
        
        double total_time = 0, total_kernel = 0, total_h2d = 0, total_d2h = 0, total_io = 0;
        double min_time = 1e9, max_time = 0;
        
        for (const auto& metric : m) {
            total_time += metric.total_time_ms;
            total_kernel += metric.kernel_time_ms;
            total_h2d += metric.h2d_time_ms;
            total_d2h += metric.d2h_time_ms;
            total_io += metric.io_time_ms;
            min_time = std::min(min_time, metric.total_time_ms);
            max_time = std::max(max_time, metric.total_time_ms);
        }
        
        double avg_time = total_time / m.size();
        
        std::cout << "\n╔══════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║  " << name << std::string(54 - name.length(), ' ') << "║" << std::endl;
        std::cout << "╚══════════════════════════════════════════════════════════╝" << std::endl;
        std::cout << "Images processed:        " << m.size() << std::endl;
        std::cout << "Average time per image:  " << avg_time << " ms" << std::endl;
        std::cout << "Min time:                " << min_time << " ms" << std::endl;
        std::cout << "Max time:                " << max_time << " ms" << std::endl;
        std::cout << "Average kernel time:     " << total_kernel / m.size() << " ms (" 
                  << (total_kernel/total_time)*100 << "%)" << std::endl;
        std::cout << "Average H2D transfer:    " << total_h2d / m.size() << " ms (" 
                  << (total_h2d/total_time)*100 << "%)" << std::endl;
        std::cout << "Average D2H transfer:    " << total_d2h / m.size() << " ms (" 
                  << (total_d2h/total_time)*100 << "%)" << std::endl;
        std::cout << "Average I/O time:        " << total_io / m.size() << " ms (" 
                  << (total_io/total_time)*100 << "%)" << std::endl;
        std::cout << "Throughput:              " << (m.size() * 1000.0) / total_time << " images/sec" << std::endl;
    };
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "       DETAILED EXECUTION TIME AND PERFORMANCE ANALYSIS" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    calcStats(gpu0_metrics, "GPU 0 Performance");
    // calcStats(gpu1_metrics, "GPU 1 Performance");
    
    // Overall statistics
    std::cout << "\n╔══════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  Overall System Performance                              ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════╝" << std::endl;
    std::cout << "Total images processed:  " << metrics.size() << std::endl;
    std::cout << "Total program time:      " << total_program_ms << " ms" << std::endl;
    std::cout << "Overall throughput:      " << (metrics.size() * 1000.0) / total_program_ms << " images/sec" << std::endl;
}

// Analyze workload balance
void analyzeWorkloadBalance() {
    if (gpu_workload.size() < 2) return;
    
    auto& wl0 = gpu_workload[0];
    auto& wl1 = gpu_workload[1];
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "            WORKLOAD UNBALANCE MEASUREMENT" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    std::cout << "\nGPU 0: " << wl0.images_processed << " images, Total span: " << wl0.total_span_ms << " ms" << std::endl;
    std::cout << "GPU 1: " << wl1.images_processed << " images, Total span: " << wl1.total_span_ms << " ms" << std::endl;
    
    double time_diff = std::abs(wl0.total_span_ms - wl1.total_span_ms);
    double imbalance_percent = (time_diff / std::max(wl0.total_span_ms, wl1.total_span_ms)) * 100.0;
    
    std::cout << "\nTime difference:         " << time_diff << " ms" << std::endl;
    std::cout << "Workload imbalance:      " << imbalance_percent << "%" << std::endl;
    
    int slower_gpu = (wl0.total_span_ms > wl1.total_span_ms) ? 0 : 1;
    int faster_gpu = 1 - slower_gpu;
    std::cout << "Slower GPU:              GPU " << slower_gpu << std::endl;
    std::cout << "Faster GPU:              GPU " << faster_gpu << " (idle time: " << time_diff << " ms)" << std::endl;
    std::cout << "Load balance ratio:      " << (double)wl0.images_processed / wl1.images_processed << std::endl;
}

// Identify bottlenecks
void identifyBottlenecks(const std::vector<ImageMetrics>& metrics) {
    if (metrics.empty()) return;
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "              BOTTLENECK IDENTIFICATION" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    // 1. Find slower GPU (Inter-Device Bottleneck)
    double gpu0_avg = 0, gpu1_avg = 0;
    int gpu0_count = 0, gpu1_count = 0;
    
    for (const auto& m : metrics) {
        if (m.gpu_id == 0) {
            gpu0_avg += m.total_time_ms;
            gpu0_count++;
        } else {
            // gpu1_avg += m.total_time_ms;
            // gpu1_count++;
        }
    }
    
    if (gpu0_count > 0) gpu0_avg /= gpu0_count;
    // if (gpu1_count > 0) gpu1_avg /= gpu1_count;
    
    std::cout << "\n[A] Inter-Device Bottleneck (GPU vs GPU)" << std::endl;
    std::cout << "    GPU 0 avg time:      " << gpu0_avg << " ms" << std::endl;
    // std::cout << "    GPU 1 avg time:      " << gpu1_avg << " ms" << std::endl;
    
    // if (gpu0_count > 0 && gpu1_count > 0) {
    //     int slower_gpu = (gpu0_avg > gpu1_avg) ? 0 : 1;
    //     double performance_gap = std::abs(gpu0_avg - gpu1_avg) / std::min(gpu0_avg, gpu1_avg) * 100.0;
    //     std::cout << "    Slower GPU:          GPU " << slower_gpu << std::endl;
    //     std::cout << "    Performance gap:     " << performance_gap << "%" << std::endl;
    // }
    
    // 2. Analyze computation vs communication bottleneck (Intra-Device)
    auto analyzePhases = [&](int gpu_id) {
        double total_kernel = 0, total_h2d = 0, total_d2h = 0, total_io = 0, total_time = 0;
        int count = 0;
        
        for (const auto& m : metrics) {
            if (m.gpu_id == gpu_id) {
                total_kernel += m.kernel_time_ms;
                total_h2d += m.h2d_time_ms;
                total_d2h += m.d2h_time_ms;
                total_io += m.io_time_ms;
                total_time += m.total_time_ms;
                count++;
            }
        }
        
        if (count == 0) return;
        
        std::cout << "\n[B] GPU " << gpu_id << " - Intra-Device Bottleneck (Computation vs Communication)" << std::endl;
        std::cout << "    Computation (kernel): " << (total_kernel/total_time)*100 << "% (" << total_kernel/count << " ms avg)" << std::endl;
        std::cout << "    Communication H2D:    " << (total_h2d/total_time)*100 << "% (" << total_h2d/count << " ms avg)" << std::endl;
        std::cout << "    Communication D2H:    " << (total_d2h/total_time)*100 << "% (" << total_d2h/count << " ms avg)" << std::endl;
        std::cout << "    I/O (load/save):      " << (total_io/total_time)*100 << "% (" << total_io/count << " ms avg)" << std::endl;
        
        double communication_total = total_h2d + total_d2h;
        std::cout << "    → BOTTLENECK: ";
        if (total_kernel > communication_total) {
            std::cout << "Computation-bound (kernel is " 
                      << std::fixed << std::setprecision(2)
                      << (total_kernel/communication_total) << "x slower than transfers)" << std::endl;
        } else {
            std::cout << "Communication-bound (transfers are " 
                      << std::fixed << std::setprecision(2)
                      << (communication_total/total_kernel) << "x slower than kernel)" << std::endl;
        }
    };
    
    analyzePhases(0);
    // analyzePhases(1);
    
    // 3. System-level bottlenecks
    std::cout << "\n[C] System-Level Bottlenecks" << std::endl;
    
    double total_io = 0, total_processing = 0;
    for (const auto& m : metrics) {
        total_io += m.io_time_ms;
        total_processing += m.total_time_ms;
    }
    
    double io_percentage = (total_io / total_processing) * 100;
    std::cout << "    I/O percentage:       " << io_percentage << "%" << std::endl;
    
    if (io_percentage > 20) {
        std::cout << "    ⚠ WARNING: Disk I/O may be a significant bottleneck!" << std::endl;
    } else {
        std::cout << "    ✓ I/O overhead is acceptable" << std::endl;
    }
}

int main(int argc, const char *argv[]){
    
    // Check arguments
    if (argc != 2) {
        std::cerr << "Invalid syntax: environ_gaussian_filter <sigma>" << std::endl;
        std::exit(1);
    }
    
    float sigma = std::stof(argv[1]);
    
    t_program_start = high_resolution_clock::now();
    
    int err;
    size_t t_buf = 50;
    char str_buffer[t_buf];
    size_t e_buf;
    
    const cl_uint num_platforms_ids = 10;
    cl_platform_id platforms_ids[num_platforms_ids];
    cl_uint n_platforms;
    const cl_uint num_devices_ids = 10;
    cl_device_id devices_ids[num_platforms_ids][num_devices_ids];
    cl_uint n_devices[num_platforms_ids];
    
    // 1. Scan platforms
    err = clGetPlatformIDs(num_platforms_ids, platforms_ids, &n_platforms);
    cl_error(err, "Error: Failed to Scan for Platforms IDs");
    std::cout << "Number of available platforms: " << n_platforms << "\n" << std::endl;
    
    for (int i = 0; i < n_platforms; i++){
        err = clGetPlatformInfo(platforms_ids[i], CL_PLATFORM_NAME, t_buf, str_buffer, &e_buf);
        cl_error(err, "Error: Failed to get info of the platform");
        printf("\t[%d]-Platform Name: %s\n", i, str_buffer);
    }
    std::cout << std::endl;
    
    // 2. Scan devices
    for (int i = 0; i < n_platforms; i++){
        err = clGetDeviceIDs(platforms_ids[i], CL_DEVICE_TYPE_ALL, num_devices_ids, devices_ids[i], &(n_devices[i]));
        cl_error(err, "Error: Failed to Scan for Devices IDs");
        printf("\t[%d]-Platform. Number of available devices: %d\n", i, n_devices[i]);
        
        for (int j = 0; j < n_devices[i]; j++){
            err = clGetDeviceInfo(devices_ids[i][j], CL_DEVICE_NAME, sizeof(str_buffer), &str_buffer, NULL);
            cl_error(err, "clGetDeviceInfo: Getting device name");
            printf("\t\t[%d]-Platform [%d]-Device CL_DEVICE_NAME: %s\n", i, j, str_buffer);
            
            cl_uint max_compute_units_available;
            err = clGetDeviceInfo(devices_ids[i][j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(max_compute_units_available), &max_compute_units_available, NULL);
            cl_error(err, "clGetDeviceInfo: Getting device max compute units available");
            printf("\t\t[%d]-Platform [%d]-Device CL_DEVICE_MAX_COMPUTE_UNITS: %d\n\n", i, j, max_compute_units_available);
        }
    }
    
    // // Verify we have at least 2 GPUs
    // if (n_devices[0] < 2) {
    //     std::cerr << "Error: Need at least 2 GPU devices. Found only " << n_devices[0] << std::endl;
    //     exit(-1);
    // }
    
    // 3. Read kernel source
    FILE *fileHandler = fopen("kernel_gaussian_filter.cl", "r");
    if (!fileHandler) {
        std::cerr << "Error: Cannot open kernel_gaussian_filter.cl" << std::endl;
        exit(-1);
    }
    fseek(fileHandler, 0, SEEK_END);
    size_t fileSize = ftell(fileHandler);
    rewind(fileHandler);
    
    char * sourceCode = (char*) malloc(fileSize + 1);
    sourceCode[fileSize] = '\0';
    fread(sourceCode, sizeof(char), fileSize, fileHandler);
    fclose(fileHandler);
    
    // 4. Create contexts for both GPUs
    // GPUContext gpu0_ctx, gpu1_ctx;
    GPUContext gpu0_ctx;
    gpu0_ctx.gpu_id = 0;
    // gpu1_ctx.gpu_id = 1;
    
    gpu0_ctx.device_id = devices_ids[0][0];
    // gpu1_ctx.device_id = devices_ids[0][1];
    
    // Create context for GPU 0
    cl_context_properties properties0[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platforms_ids[0], 0};
    gpu0_ctx.context = clCreateContext(properties0, 1, &gpu0_ctx.device_id, NULL, NULL, &err);
    cl_error(err, "Failed to create context for GPU 0");
    
    // // Create context for GPU 1
    // cl_context_properties properties1[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platforms_ids[0], 0};
    // gpu1_ctx.context = clCreateContext(properties1, 1, &gpu1_ctx.device_id, NULL, NULL, &err);
    // cl_error(err, "Failed to create context for GPU 1");
    
    // Create command queues with profiling
    cl_command_queue_properties proprt[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    gpu0_ctx.command_queue = clCreateCommandQueueWithProperties(gpu0_ctx.context, gpu0_ctx.device_id, proprt, &err);
    cl_error(err, "Failed to create command queue for GPU 0");
    
    // gpu1_ctx.command_queue = clCreateCommandQueueWithProperties(gpu1_ctx.context, gpu1_ctx.device_id, proprt, &err);
    // cl_error(err, "Failed to create command queue for GPU 1");
    
    // Build program for GPU 0
    gpu0_ctx.program = clCreateProgramWithSource(gpu0_ctx.context, 1, (const char**)&sourceCode, &fileSize, &err);
    cl_error(err, "Failed to create program for GPU 0");
    
    err = clBuildProgram(gpu0_ctx.program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS){
        size_t len;
        char buffer[2048];
        std::cout << "Error: Building program for GPU 0" << std::endl;
        clGetProgramBuildInfo(gpu0_ctx.program, gpu0_ctx.device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        std::cout << buffer << std::endl;
        exit(-1);
    }
    
    // // Build program for GPU 1
    // gpu1_ctx.program = clCreateProgramWithSource(gpu1_ctx.context, 1, (const char**)&sourceCode, &fileSize, &err);
    // cl_error(err, "Failed to create program for GPU 1");
    
    // err = clBuildProgram(gpu1_ctx.program, 0, NULL, NULL, NULL, NULL);
    // if (err != CL_SUCCESS){
    //     size_t len;
    //     char buffer[2048];
    //     std::cout << "Error: Building program for GPU 1" << std::endl;
    //     clGetProgramBuildInfo(gpu1_ctx.program, gpu1_ctx.device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
    //     std::cout << buffer << std::endl;
    //     exit(-1);
    // }
    
    free(sourceCode);
    
    // Create kernels
    gpu0_ctx.kernel = clCreateKernel(gpu0_ctx.program, "gaussian_filter", &err);
    cl_error(err, "Failed to create kernel for GPU 0");
    
    // gpu1_ctx.kernel = clCreateKernel(gpu1_ctx.program, "gaussian_filter", &err);
    // cl_error(err, "Failed to create kernel for GPU 1");
    
    std::cout << "\n=== GPU Contexts Initialized on Berlin Server ===\n" << std::endl;
    
    // 5. Scan dataset folder for images
    std::vector<std::string> image_paths;
    std::string dataset_path = "dataset";
    
    if (!fs::exists(dataset_path)) {
        std::cerr << "Error: 'dataset' folder not found!" << std::endl;
        exit(-1);
    }
    
    for (const auto& entry : fs::directory_iterator(dataset_path)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".jpg" || ext == ".jpeg") {
                image_paths.push_back(entry.path().string());
            }
        }
    }
    
    if (image_paths.empty()) {
        std::cerr << "Error: No JPG images found in 'dataset' folder!" << std::endl;
        exit(-1);
    }
    
    std::cout << "Found " << image_paths.size() << " images to process\n" << std::endl;
    std::cout << "Starting concurrent processing \n" << std::endl;
    
    // 6. Process images 
    // std::vector<std::thread> threads;
    
    for (size_t i = 0; i < image_paths.size(); i++) {
        // Alternate between GPU 0 and GPU 1
        // GPUContext& gpu_ctx = (i % 2 == 0) ? gpu0_ctx : gpu1_ctx;
        // GPUContext& gpu_ctx = gpu0_ctx;
        
        // Launch thread for processing
        // threads.emplace_back(processImage, std::ref(gpu_ctx), image_paths[i], sigma);
        processImage(gpu0_ctx, image_paths[i], sigma);
        
        // // Process in pairs to keep both GPUs busy
        // if (threads.size() == 2 || i == image_paths.size() - 1) {
        //     for (auto& t : threads) {
        //         t.join();
        //     }
        //     threads.clear();
        // }
    }
    
    auto t_program_end = high_resolution_clock::now();
    double program_ms = duration<double, std::milli>(t_program_end - t_program_start).count();
    
    // 7. Generate comprehensive analysis reports
    generatePerformanceReport(all_metrics, program_ms);
    // analyzeWorkloadBalance();
    identifyBottlenecks(all_metrics);
    
    // 8. Export metrics to CSV
    exportToCSV(all_metrics, "performance_metrics.csv");
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "All " << image_paths.size() << " images processed successfully!" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    // 9. Cleanup
    clReleaseKernel(gpu0_ctx.kernel);
    // clReleaseKernel(gpu1_ctx.kernel);
    clReleaseProgram(gpu0_ctx.program);
    // clReleaseProgram(gpu1_ctx.program);
    clReleaseCommandQueue(gpu0_ctx.command_queue);
    // clReleaseCommandQueue(gpu1_ctx.command_queue);
    clReleaseContext(gpu0_ctx.context);
    // clReleaseContext(gpu1_ctx.context);
    
    return 0;
}
