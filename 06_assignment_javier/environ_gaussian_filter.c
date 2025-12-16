#include <chrono>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>
#include <tuple>
#include <vector>
#include <thread>
#include <mutex>
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
std::mutex console_mutex;

// check error, in such a case, it exits
void cl_error(cl_int code, const char *string){
    if (code != CL_SUCCESS){
        std::lock_guard<std::mutex> lock(console_mutex);
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
    auto t_start = high_resolution_clock::now();
    
    int err;
    
    // Extract filename from path
    std::string filename = fs::path(image_path).filename().string();
    
    {
        std::lock_guard<std::mutex> lock(console_mutex);
        //std::cout << "[GPU " << gpu_ctx.gpu_id << "] Processing: " << filename << std::endl;
    }
    
    // Load image
    cimg_library::CImg<unsigned char> img(image_path.c_str());
    int width = img.width();
    int height = img.height();
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
    
    // Write input image
    cl_event write_event;
    size_t origin[3] = {0,0,0};
    size_t region[3] = {(size_t)width, (size_t)height, 1};
    err = clEnqueueWriteImage(gpu_ctx.command_queue, clImage_In, CL_TRUE, origin, region, 0, 0, rgba_in.data(), 0, NULL, &write_event);
    cl_error(err, "Failed to write input image to device");
    
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
    
    // Launch kernel
    cl_event kernel_event;
    size_t global_size[2] = { (size_t)width, (size_t)height };
    err = clEnqueueNDRangeKernel(gpu_ctx.command_queue, gpu_ctx.kernel, 2, NULL, global_size, NULL, 0, NULL, &kernel_event);
    cl_error(err, "Failed to launch kernel to the device");
    
    // Read output image
    std::vector<unsigned char> outRGBA(width*height*4);
    cl_event read_event;
    err = clEnqueueReadImage(gpu_ctx.command_queue, clImage_Out, CL_TRUE, origin, region, 0, 0, outRGBA.data(), 0, NULL, &read_event);
    cl_error(err, "Failed to read output");
    
    // Save output
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
    
    std::string result = results_dir + "/result_gpu" + std::to_string(gpu_ctx.gpu_id) + "_" + filename;
    outImg.save(result.c_str());
    
    // Calculate metrics
    auto t_end = high_resolution_clock::now();
    double total_ms = duration<double, std::milli>(t_end - t_start).count();
    
    cl_ulong start_ns = 0, end_ns = 0;
    clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(start_ns), &start_ns, NULL);
    clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(end_ns), &end_ns, NULL);
    double kernel_ms = (end_ns - start_ns) * 1e-6;
    
    {
        std::lock_guard<std::mutex> lock(console_mutex);
        //std::cout << "[GPU " << gpu_ctx.gpu_id << "] Completed: " << filename 
        //          << " | Total: " << total_ms << " ms | Kernel: " << kernel_ms << " ms" << std::endl;
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

int main(int argc, const char *argv[]){
    
    // Check arguments
    if (argc != 2) {
        std::cerr << "Invalid syntax: environ_gaussian_filter <sigma>" << std::endl;
        std::exit(1);
    }
    
    float sigma = std::stof(argv[1]);
    
    auto t_program_start = high_resolution_clock::now();
    
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
    
    std::cout << "\n=== GPU Contexts Initialized ===\n" << std::endl;
    
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
    
    // 6. Process images with 50/50 load balancing
    std::vector<std::thread> threads;
    
    for (size_t i = 0; i < image_paths.size(); i++) {
        // Alternate between GPU 0 and GPU 1
        GPUContext& gpu_ctx = (i % 2 == 0) ? gpu0_ctx : gpu1_ctx;
        
        // Launch thread for processing
        threads.emplace_back(processImage, std::ref(gpu_ctx), image_paths[i], sigma);
        
        // Process in pairs to keep both GPUs busy
        // Wait for pair to complete before launching next pair
        if (threads.size() == 2 || i == image_paths.size() - 1) {
            for (auto& t : threads) {
                t.join();
            }
            threads.clear();
        }
    }
    
    // 7. Cleanup
    clReleaseKernel(gpu0_ctx.kernel);
    clReleaseKernel(gpu1_ctx.kernel);
    clReleaseProgram(gpu0_ctx.program);
    clReleaseProgram(gpu1_ctx.program);
    clReleaseCommandQueue(gpu0_ctx.command_queue);
    clReleaseCommandQueue(gpu1_ctx.command_queue);
    clReleaseContext(gpu0_ctx.context);
    clReleaseContext(gpu1_ctx.context);
    
    auto t_program_end = high_resolution_clock::now();
    double program_ms = duration<double, std::milli>(t_program_end - t_program_start).count();
    
    std::cout << "\n=== Processing Complete ===" << std::endl;
    std::cout << "Total images processed: " << image_paths.size() << std::endl;
    std::cout << "Total program time: " << program_ms << " ms" << std::endl;
    std::cout << "Average time per image: " << program_ms / image_paths.size() << " ms" << std::endl;
    
    return 0;
}