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

struct Metrics {
    int iters = 0;
    double h2d_ms = 0.0;
    double kernel_ms = 0.0;
    double d2h_ms = 0.0;
};

// check error, in such a case, it exits
void cl_error(cl_int code, const char *string){
    if (code != CL_SUCCESS){
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


double eventDurationMs(cl_event evt) {
    cl_ulong start = 0, end = 0;
    clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START,
                            sizeof(start), &start, nullptr);
    clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END,
                            sizeof(end), &end, nullptr);
    return (end - start) * 1e-6; // ns → ms
}

int main(int argc, const char *argv[]) {

    if (argc != 2) {
        std::cerr << "Usage: gaussian_filter <sigma>\n";
        return -1;
    }

    float sigma = std::stof(argv[1]);

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
    
    // Verify we have at least 2 GPUs
    if (n_devices[0] < 2) {
        std::cerr << "Error: Need at least 2 GPU devices. Found only " << n_devices[0] << std::endl;
        exit(-1);
    }
    
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
    GPUContext gpu0_ctx, gpu1_ctx;
    gpu0_ctx.gpu_id = 0;
    gpu1_ctx.gpu_id = 1;
    
    gpu0_ctx.device_id = devices_ids[0][0];
    gpu1_ctx.device_id = devices_ids[0][1];
    
    // Create context for GPU 0
    cl_context_properties properties0[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platforms_ids[0], 0};
    gpu0_ctx.context = clCreateContext(properties0, 1, &gpu0_ctx.device_id, NULL, NULL, &err);
    cl_error(err, "Failed to create context for GPU 0");
    
    // Create context for GPU 1
    cl_context_properties properties1[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platforms_ids[0], 0};
    gpu1_ctx.context = clCreateContext(properties1, 1, &gpu1_ctx.device_id, NULL, NULL, &err);
    cl_error(err, "Failed to create context for GPU 1");
    
    // Create command queues with profiling
    cl_command_queue_properties proprt[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    gpu0_ctx.command_queue = clCreateCommandQueueWithProperties(gpu0_ctx.context, gpu0_ctx.device_id, proprt, &err);
    cl_error(err, "Failed to create command queue for GPU 0");
    
    gpu1_ctx.command_queue = clCreateCommandQueueWithProperties(gpu1_ctx.context, gpu1_ctx.device_id, proprt, &err);
    cl_error(err, "Failed to create command queue for GPU 1");
    
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
    
    // Build program for GPU 1
    gpu1_ctx.program = clCreateProgramWithSource(gpu1_ctx.context, 1, (const char**)&sourceCode, &fileSize, &err);
    cl_error(err, "Failed to create program for GPU 1");
    
    err = clBuildProgram(gpu1_ctx.program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS){
        size_t len;
        char buffer[2048];
        std::cout << "Error: Building program for GPU 1" << std::endl;
        clGetProgramBuildInfo(gpu1_ctx.program, gpu1_ctx.device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        std::cout << buffer << std::endl;
        exit(-1);
    }
    
    free(sourceCode);
    
    // Create kernels
    gpu0_ctx.kernel = clCreateKernel(gpu0_ctx.program, "gaussian_filter", &err);
    cl_error(err, "Failed to create kernel for GPU 0");
    
    gpu1_ctx.kernel = clCreateKernel(gpu1_ctx.program, "gaussian_filter", &err);
    cl_error(err, "Failed to create kernel for GPU 1");
    
    std::cout << "\n=== GPU Contexts Initialized on Berlin Server ===\n" << std::endl;

    // 5. Load image ONCE
    cimg_library::CImg<unsigned char> img("dataset/cat_1000x600.jpg");

    int width  = img.width();
    int height = img.height();
    size_t num_pixels = (size_t)width * height;

    std::vector<unsigned char> rgba_in(num_pixels * 4);
    std::vector<unsigned char> rgba_out(width * height * 4);
    cimg_forXY(img, x, y) {
        int idx = 4 * (y * width + x);
        rgba_in[idx + 0] = img(x,y,0);
        rgba_in[idx + 1] = img(x,y,1);
        rgba_in[idx + 2] = img(x,y,2);
        rgba_in[idx + 3] = 255;
    }

    // 6. Create Gaussian mask ONCE
    int maskSize;
    float* mask = createMask(sigma, &maskSize);
    int side = 2 * maskSize + 1;

    // 7. OpenCL image description
    cl_image_format img_format;
    img_format.image_channel_order = CL_RGBA;
    img_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc img_desc{};
    img_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    img_desc.image_width = width;
    img_desc.image_height = height;

    size_t origin[3] = {0,0,0};
    size_t region[3] = {(size_t)width, (size_t)height, 1};

    // 8. Create buffers ONCE per GPU
    auto setupGPU = [&](GPUContext& ctx,
                        cl_mem& img_in,
                        cl_mem& img_out,
                        cl_mem& mask_buf) {

        int err;
        img_in = clCreateImage(ctx.context, CL_MEM_READ_ONLY,
                               &img_format, &img_desc, nullptr, &err);
        cl_error(err, "clCreateImage input");

        img_out = clCreateImage(ctx.context, CL_MEM_WRITE_ONLY,
                                &img_format, &img_desc, nullptr, &err);
        cl_error(err, "clCreateImage output");

        mask_buf = clCreateBuffer(ctx.context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  sizeof(float) * side * side,
                                  mask, &err);
        cl_error(err, "clCreateBuffer mask");

        clSetKernelArg(ctx.kernel, 0, sizeof(cl_mem), &img_in);
        clSetKernelArg(ctx.kernel, 1, sizeof(cl_mem), &img_out);
        clSetKernelArg(ctx.kernel, 2, sizeof(cl_mem), &mask_buf);
        clSetKernelArg(ctx.kernel, 3, sizeof(int), &maskSize);
        clSetKernelArg(ctx.kernel, 4, sizeof(int), &width);
        clSetKernelArg(ctx.kernel, 5, sizeof(int), &height);
    };

    cl_mem img0_in, img0_out, mask0;
    cl_mem img1_in, img1_out, mask1;

    setupGPU(gpu0_ctx, img0_in, img0_out, mask0);
    setupGPU(gpu1_ctx, img1_in, img1_out, mask1);

    // 9. Execute FULL PIPELINE 5000 times
    const int N = 5000;
    size_t global_size[2] = {(size_t)width, (size_t)height};
    Metrics m[2];
    std::vector<cl_event> h2d_ev[2], ker_ev[2], d2h_ev[2];
    size_t bytes_per_image = (size_t)width * height * 4;
    std::vector<unsigned char> out_gpu[2] = {
        std::vector<unsigned char>(bytes_per_image),
        std::vector<unsigned char>(bytes_per_image)
    };

    auto t_start = high_resolution_clock::now();

    for (int i = 0; i < N; ++i) {
        int g = (i % 2);
        GPUContext& ctx = (g == 0) ? gpu0_ctx : gpu1_ctx;
        cl_mem& img_in  = (g == 0) ? img0_in  : img1_in;
        cl_mem& img_out = (g == 0) ? img0_out : img1_out;

        cl_event evt_h2d, evt_kernel, evt_d2h;

        clEnqueueWriteImage(ctx.command_queue,
                            img_in, CL_FALSE,
                            origin, region,
                            0, 0, rgba_in.data(),
                            0, nullptr, &evt_h2d);

        clEnqueueNDRangeKernel(ctx.command_queue,
                               ctx.kernel,
                               2, nullptr,
                               global_size, nullptr,
                               1, &evt_h2d, &evt_kernel);

        // std::vector<unsigned char> tmp(width * height * 4);
        clEnqueueReadImage(ctx.command_queue,
                           img_out, CL_FALSE,
                           origin, region,
                           0, 0, out_gpu[g].data(),
                           1, &evt_kernel, &evt_d2h);
        
        h2d_ev[g].push_back(evt_h2d);
        ker_ev[g].push_back(evt_kernel);
        d2h_ev[g].push_back(evt_d2h);
        
        m[g].iters++;
    }

    clFlush(gpu0_ctx.command_queue);
    clFlush(gpu1_ctx.command_queue);

    clFinish(gpu0_ctx.command_queue);
    clFinish(gpu1_ctx.command_queue);

    auto t_end = high_resolution_clock::now();
    double total_ms = duration<double, std::milli>(t_end - t_start).count();


    for (int g = 0; g < 2; ++g) {
        for (auto e : h2d_ev[g]) m[g].h2d_ms += eventDurationMs(e);
        for (auto e : ker_ev[g]) m[g].kernel_ms += eventDurationMs(e);
        for (auto e : d2h_ev[g]) m[g].d2h_ms += eventDurationMs(e);

        for (auto e : h2d_ev[g]) clReleaseEvent(e);
        for (auto e : ker_ev[g]) clReleaseEvent(e);
        for (auto e : d2h_ev[g]) clReleaseEvent(e);
    }

    // 10. Print meaningful metrics
    std::cout << "\n==============================\n";
    std::cout << "Total iterations:     " << N << "\n";
    std::cout << "Total time:           " << total_ms << " ms\n";

    double avg_h2d_ms[2];
    double avg_kernel_ms[2];
    double avg_d2h_ms[2];

    for (int g = 0; g < 2; ++g) {
        avg_h2d_ms[g]    = m[g].h2d_ms    / m[g].iters;
        avg_kernel_ms[g] = m[g].kernel_ms / m[g].iters;
        avg_d2h_ms[g]    = m[g].d2h_ms    / m[g].iters;
    }

    std::cout << "\n=== Average times per iteration ===\n";
    std::cout << "GPU 0: H2D = " << avg_h2d_ms[0]
            << " ms | Kernel = " << avg_kernel_ms[0]
            << " ms | D2H = " << avg_d2h_ms[0] << " ms\n"
            << "iters = " << m[0].iters;

    std::cout << "GPU 1: H2D = " << avg_h2d_ms[1]
            << " ms | Kernel = " << avg_kernel_ms[1]
            << " ms | D2H = " << avg_d2h_ms[1] << " ms\n"
            << "iters = " << m[1].iters;

    double h2d_bw_avg0 = (bytes_per_image / (1024.0*1024.0)) / (avg_h2d_ms[0] / 1000.0);
    double d2h_bw_avg0 = (bytes_per_image / (1024.0*1024.0)) / (avg_d2h_ms[0] / 1000.0);

    double h2d_bw_avg1 = (bytes_per_image / (1024.0*1024.0)) / (avg_h2d_ms[1] / 1000.0);
    double d2h_bw_avg1 = (bytes_per_image / (1024.0*1024.0)) / (avg_d2h_ms[1] / 1000.0);

    std::cout << "\n=== Average Bandwidth per GPU ===\n";
    std::cout << "GPU 0: H2D = " << h2d_bw_avg0
            << " MB/s | D2H = " << d2h_bw_avg0 << " MB/s\n";
    std::cout << "GPU 1: H2D = " << h2d_bw_avg1
            << " MB/s | D2H = " << d2h_bw_avg1 << " MB/s\n";

    double imbalance_iters =
    std::abs(m[0].iters - m[1].iters) / double(N);

    std::cout << "\nWorkload imbalance (iterations): "
          << imbalance_iters * 100.0 << " %\n";

    double kernel_imbalance_ratio =
    std::max(m[0].kernel_ms, m[1].kernel_ms) /
    std::min(m[0].kernel_ms, m[1].kernel_ms);
    std::cout << "Kernel time imbalance ratio (max/min): "
          << kernel_imbalance_ratio << "\n";

    double comm_ms_gpu0 = m[0].h2d_ms + m[0].d2h_ms;
    double comm_ms_gpu1 = m[1].h2d_ms + m[1].d2h_ms;

    double comp_frac_gpu0 = m[0].kernel_ms / (m[0].kernel_ms + comm_ms_gpu0);
    double comm_frac_gpu0 = comm_ms_gpu0     / (m[0].kernel_ms + comm_ms_gpu0);

    double comp_frac_gpu1 = m[1].kernel_ms / (m[1].kernel_ms + comm_ms_gpu1);
    double comm_frac_gpu1 = comm_ms_gpu1     / (m[1].kernel_ms + comm_ms_gpu1);

    std::cout << "\n=== Bottleneck analysis ===\n";
    std::cout << "GPU 0: Computation = " << comp_frac_gpu0 * 100.0
            << "% | Communication = " << comm_frac_gpu0 * 100.0 << "%\n";
    std::cout << "GPU 1: Computation = " << comp_frac_gpu1 * 100.0
            << "% | Communication = " << comm_frac_gpu1 * 100.0 << "%\n";


    double total_kernel_ms = m[0].kernel_ms + m[1].kernel_ms;
    double total_comm_ms   = m[0].h2d_ms + m[0].d2h_ms
                        + m[1].h2d_ms + m[1].d2h_ms;

    double global_comp_frac = total_kernel_ms / (total_kernel_ms + total_comm_ms);
    double global_comm_frac = total_comm_ms   / (total_kernel_ms + total_comm_ms);


    std::cout << "\nGlobal bottleneck:\n";
    std::cout << "Computation = " << global_comp_frac * 100.0
            << "% | Communication = " << global_comm_frac * 100.0 << "%\n";


    // 11. Read FINAL result and save
    cimg_library::CImg<unsigned char> outImg(width, height, 1, 3);
    cimg_forXY(outImg, x, y) {
        int i = 4*(y*width + x);
        outImg(x,y,0) = out_gpu[0][i];
        outImg(x,y,1) = out_gpu[0][i+1];
        outImg(x,y,2) = out_gpu[0][i+2];
    }

    std::string result = "final_result.jpg";
    outImg.save(result.c_str());
    std::cout << "Gaussian filter saved\n"<<std::endl;

    // 12. Cleanup
    clReleaseMemObject(img0_in);
    clReleaseMemObject(img0_out);
    clReleaseMemObject(mask0);
    clReleaseMemObject(img1_in);
    clReleaseMemObject(img1_out);
    clReleaseMemObject(mask1);
    clReleaseKernel(gpu0_ctx.kernel);
    clReleaseKernel(gpu1_ctx.kernel);
    clReleaseProgram(gpu0_ctx.program);
    clReleaseProgram(gpu1_ctx.program);
    clReleaseCommandQueue(gpu0_ctx.command_queue);
    clReleaseCommandQueue(gpu1_ctx.command_queue);
    clReleaseContext(gpu0_ctx.context);
    clReleaseContext(gpu1_ctx.context);
    delete[] mask;

    return 0;
}