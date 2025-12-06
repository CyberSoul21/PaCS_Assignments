
#include <chrono>
#include <filesystem>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>
#include "CImg/CImg.h"
#define CL_TARGET_OPENCL_VERSION 220
#ifdef __APPLE__
	  #include <OpenCL/opencl.h>
#else
	#include <CL/cl.h>
#endif
	  
using namespace std::chrono;

// check error, in such a case, it exits

void cl_error(cl_int code, const char *string){
	if (code != CL_SUCCESS){
		printf("%d - %s\n", code, string);
		exit(-1);
	}
}

// Creat gaussian mask
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

std::pair<size_t, std::string>
usage(int argc, const char *argv[]) {
    // read the number of steps from the command line
    if (argc != 3) {
        std::cerr << "Invalid syntax: environ_gaussian_filter <sigma> <image>" << std::endl;
        exit(1);
    }

    size_t sigma = std::stoll(argv[1]);
    std::string image = argv[2];

    return std::make_pair(sigma, image);
}


int main(int argc, const char *argv[]){

	//Set arguments
	auto ret_pair = usage(argc, argv);
    float sigma = (float)ret_pair.first;
    std::string image = ret_pair.second;

	// Complete program time
	auto t_program_start = high_resolution_clock::now();

	int err;                    // error code returned from api calls
	size_t t_buf = 50;			// size of str_buffer
	char str_buffer[t_buf];		// auxiliary buffer	
	size_t e_buf;				// effective size of str_buffer in use
		
	const cl_uint num_platforms_ids = 10;				// max of allocatable platforms
	cl_platform_id platforms_ids[num_platforms_ids];		// array of platforms
	cl_uint n_platforms;						// effective number of platforms in use
	const cl_uint num_devices_ids = 10;				// max of allocatable devices
	cl_device_id devices_ids[num_platforms_ids][num_devices_ids];	// array of devices
	cl_uint n_devices[num_platforms_ids];				// effective number of devices in use for each platform
		
	cl_device_id device_id;             				// compute device id 
	cl_context context;                 				// compute context
	cl_command_queue command_queue;     				// compute command queue
	cl_program program;                 				// compute program
	cl_kernel kernel;                   				// compute kernel
	
	
	// 1. Scan the available platforms:
	err = clGetPlatformIDs (num_platforms_ids, platforms_ids, &n_platforms);
	cl_error(err, "Error: Failed to Scan for Platforms IDs");
	printf("Number of available platforms: %d\n\n", n_platforms);

	for (int i = 0; i < n_platforms; i++ ){
		err= clGetPlatformInfo(platforms_ids[i], CL_PLATFORM_NAME, t_buf, str_buffer, &e_buf);
		cl_error (err, "Error: Failed to get info of the platform\n");
		printf( "\t[%d]-Platform Name: %s\n", i, str_buffer);
	}
	printf("\n");
		
	// 2. Scan for devices in each platform
	for (int i = 0; i < n_platforms; i++ ){
		err = clGetDeviceIDs(platforms_ids[i], CL_DEVICE_TYPE_ALL, num_devices_ids, devices_ids[i], &(n_devices[i]));
		cl_error(err, "Error: Failed to Scan for Devices IDs");
		printf("\t[%d]-Platform. Number of available devices: %d\n", i, n_devices[i]);

		for(int j = 0; j < n_devices[i]; j++){
		err = clGetDeviceInfo(devices_ids[i][j], CL_DEVICE_NAME, sizeof(str_buffer), &str_buffer, NULL);
		cl_error(err, "clGetDeviceInfo: Getting device name");
		printf("\t\t [%d]-Platform [%d]-Device CL_DEVICE_NAME: %s\n", i, j,str_buffer);

		cl_uint max_compute_units_available;
		err = clGetDeviceInfo(devices_ids[i][j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(max_compute_units_available), &max_compute_units_available, NULL);
		cl_error(err, "clGetDeviceInfo: Getting device max compute units available");
		printf("\t\t [%d]-Platform [%d]-Device CL_DEVICE_MAX_COMPUTE_UNITS: %d\n\n", i, j, max_compute_units_available);
		
		size_t max_work_size;
		err = clGetDeviceInfo(devices_ids[i][j], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_size), &max_work_size, NULL);
		cl_error(err, "clGetDeviceInfo: Getting device max work-group size available");
		printf("\t\t [%d]-Platform [%d]-Device CL_DEVICE_MAX_WORK_GROUP_SIZE: %d\n\n", i, j, max_work_size);
		}
	}	

	// 3. Create a context, with a device
	cl_context_properties properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platforms_ids[0], 0};
	context = clCreateContext(properties, 1, devices_ids[0], NULL, NULL, &err);
	cl_error(err, "Failed to create a compute context\n");

	// 4. Create a command queue
	cl_command_queue_properties proprt[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
	command_queue = clCreateCommandQueueWithProperties(context, devices_ids[0][0], proprt, &err);
	cl_error(err, "Failed to create a command queue\n");

	// 5. Read an OpenCL program from the file kernel.cl
	// Calculate size of the file
	FILE *fileHandler = fopen("./kernel_gaussian_filter.cl", "r");
	fseek(fileHandler, 0, SEEK_END);
	size_t fileSize = ftell(fileHandler);
	rewind(fileHandler);

	// read kernel source into buffer
	char * sourceCode = (char*) malloc(fileSize + 1);
	sourceCode[fileSize] = '\0';
	fread(sourceCode, sizeof(char), fileSize, fileHandler);
	fclose(fileHandler);

	// create program from buffer
	program = clCreateProgramWithSource(context, 1, (const char**) &sourceCode, &fileSize, &err);
	cl_error(err, "Failed to create program with source\n");
	free(sourceCode);

	// read kernel source back in from program to check
	size_t kernelSourceSize;
	clGetProgramInfo(program, CL_PROGRAM_SOURCE, 0, NULL, &kernelSourceSize);
	char *kernelSource = (char*) malloc(kernelSourceSize);
	clGetProgramInfo(program, CL_PROGRAM_SOURCE, kernelSourceSize, kernelSource, NULL);
	free(kernelSource);

	// Build the executable and check errors
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS){
		size_t len;
		char buffer[2048];

		printf("Error: Some error at building process.\n");
		clGetProgramBuildInfo(program, devices_ids[0][0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		printf("%s\n", buffer);
		exit(-1);
	}

	// Create a compute kernel with the program we want to run
	kernel = clCreateKernel(program, "gaussian_filter", &err);
	cl_error(err, "Failed to create kernel from the program\n");

	// 6. Get image
	std::string abs_image = std::filesystem::absolute(image).string();
	// cimg_library::CImg<unsigned char> img(image.c_str());
	cimg_library::CImg<unsigned char> img(abs_image.c_str());
	int width = img.width();
	int height = img.height();
	double num_pixels = (double)width * (double)height;

	// Convert CImg planar RGB → interleaved RGB (OpenCL format)
    std::vector<unsigned char> rgba_in(num_pixels*4);
    cimg_forXY(img, x, y){
        int idx = 4*(y*width + x);
        rgba_in[idx+0] = img(x,y,0);
        rgba_in[idx+1] = img(x,y,1);
        rgba_in[idx+2] = img(x,y,2);
        rgba_in[idx+3] = 255;
    }

	// 7. Create Gaussian mask
    int maskSize;
    float * mask = createMask(sigma, &maskSize);
	int side = 2*maskSize + 1;

	// Create OpenCL buffer visible to the OpenCl runtime
	cl_image_format img_format;
	img_format.image_channel_order = CL_RGBA;
	img_format.image_channel_data_type = CL_UNORM_INT8;

	cl_image_desc img_desc;
	memset(&img_desc, 0, sizeof(img_desc));   // MUY IMPORTANTE

	img_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
	img_desc.image_width = width;
	img_desc.image_height = height;
	img_desc.image_depth = 1;
	img_desc.image_array_size = 1;

	// 8. Write input image
	cl_mem clImage_In = clCreateImage(context, CL_MEM_READ_ONLY, &img_format, &img_desc,NULL, &err);
	cl_error(err, "Failed to create input image at device\n");
    cl_event write_event;
	size_t origin[3] = {0,0,0};
	size_t region[3] = {(size_t)width, (size_t)height, 1};
	err = clEnqueueWriteImage(command_queue,clImage_In,CL_TRUE,origin,region,0, 0,rgba_in.data(),0, NULL,&write_event);
	cl_error(err, "Failed to write input image to device\n");

	// 9. Create read buffer
	cl_mem clImage_Out = clCreateImage(context, CL_MEM_WRITE_ONLY, &img_format, &img_desc, NULL, &err);
	cl_error(err, "Failed to create output image at device\n");
    
    // 10. Create buffer for mask and transfer it to the device
    cl_mem clMask = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*side*side, mask, &err);
	cl_error(err, "Failed to create mask buffer at device\n");

	// 11. Set the arguments to the kernel
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &clImage_In);
	cl_error(err, "Failed to set argument 0\n");
	err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &clImage_Out);
	cl_error(err, "Failed to set argument 1\n");
	err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &clMask);
	cl_error(err, "Failed to set argument 2\n");
	err = clSetKernelArg(kernel, 3, sizeof(int), &maskSize);
	cl_error(err, "Failed to set argument 3\n");
	err = clSetKernelArg(kernel, 4, sizeof(int), &width);
	cl_error(err, "Failed to set argument 4\n");
	err = clSetKernelArg(kernel, 5, sizeof(int), &height);
	cl_error(err, "Failed to set argument 5\n");

	// 12. Launch Kernel
	cl_event kernel_event;
	size_t local_size[2] = {16, 16};
	size_t global_size[2] = { (size_t)width, (size_t)height };
	err = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_size, NULL, 0, NULL, &kernel_event);
	cl_error(err, "Failed to launch kernel to the device\n");
	//we are not calling clWaitForEvents(1, &kernel_event); because in readImage we have CL_TRUE,
	//that blocks until the kernel has finished, avoiding calling the event before

    // 13. Read output image 
    std::vector<unsigned char> outRGBA(width*height*4);
	cl_event read_event;
    err = clEnqueueReadImage(command_queue, clImage_Out, CL_TRUE,origin, region,0, 0,outRGBA.data(),0, NULL, &read_event);
    cl_error(err,"Failed to read output");

	// 14. Save output
	cimg_library::CImg<unsigned char> outImg(width, height, 1, 3);
    cimg_forXY(outImg, x,y) {
        int i = 4*(y*width + x);
        outImg(x,y,0) = outRGBA[i+0];
        outImg(x,y,1) = outRGBA[i+1];
        outImg(x,y,2) = outRGBA[i+2];
    }
	
	std::string result = "result_" + std::to_string(sigma) + "_" + image;
    outImg.save(result.c_str());
    std::cout << "Gaussian filter saved\n"<<std::endl;

	// 15. Released cl objects
	clReleaseMemObject(clMask);
	clReleaseMemObject(clImage_Out);
	clReleaseMemObject(clImage_In);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
	
	// 16. Metrics
	// Complete program time
	auto t_program_end = high_resolution_clock::now();
    double program_ms = duration<double, std::milli>(t_program_end - t_program_start).count();
    std::cout << "sigma: " << sigma << ", image: " <<image<<std::endl;
	std::cout << "matrix dimention: " << side <<std::endl;
	std::cout << "Program time: " << program_ms << " ms" << std::endl;

	// Kernel time
	cl_ulong start_ns = 0, end_ns = 0;
	clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(start_ns), &start_ns, NULL);
	clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END,	sizeof(end_ns), &end_ns, NULL);
	clReleaseEvent(kernel_event);
	double kernel_ms = (end_ns - start_ns) * 1e-6;
	std::cout << "Kernel time: " << kernel_ms << " ms" << std::endl;

	// Measure input image time and bandwidth
	cl_ulong w_start = 0, w_end = 0;
	clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_START,sizeof(w_start), &w_start, NULL);
	clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_END,sizeof(w_end), &w_end, NULL);
	clReleaseEvent(write_event);
	double write_ms = (w_end - w_start) * 1e-6;
	size_t bytes_h2d = num_pixels * 4; // 4 bytes per pixel (RGBA)
	double bandwidth_h2d_GBps = (bytes_h2d / (1024.0*1024.0*1024.0)) / (write_ms / 1000.0);
	std::cout << "Host->Device: " << write_ms << " ms, "<< bandwidth_h2d_GBps << " GB/s" << std::endl;

	// Measure output image time and bandwidth
	cl_ulong r_start = 0, r_end = 0;
	clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START,sizeof(r_start), &r_start, NULL);
	clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END,sizeof(r_end), &r_end, NULL);
	clReleaseEvent(read_event);
	double read_ms = (r_end - r_start) * 1e-6;
	size_t bytes_d2h = num_pixels * 4;
	double bandwidth_d2h_GBps = (bytes_d2h / (1024.0*1024.0*1024.0)) / (read_ms / 1000.0);
	std::cout << "Device->Host: " << read_ms << " ms, "<< bandwidth_d2h_GBps << " GB/s" << std::endl;

	//Kernel troughput
	double kernel_s = kernel_ms / 1000.0;
	double pixels_per_sec = num_pixels / kernel_s; //each pixel corresponds to a work-item, kernel time is what it takes from the first pixel to the last 
	std::cout << "Throughput (pixels/s): " << pixels_per_sec << std::endl;
	// This could be GFLOP/s if we assume an operation to be 1 mul + 1 add, so 2 FLOPs per color (we compute 3 colors)
	double flops_per_pixel = 2.0 * (double)side * (double)side * 3.0;
	double total_flops = num_pixels * flops_per_pixel; //We are assuming all pixels do the same operations, it would be less considering that pixels on the edges have less operations
	double gflops = (total_flops / kernel_s) / 1e9;
	std::cout << "Kernel throughput: " << gflops << " GFLOP/s" << std::endl;

	// Kernel bandwithd
	double bytes_per_read = 16.0;  // Read_imagef in kernel returns a float4 (4 components of 4 bytes) in private memory
	double bytes_per_write = 16.0; // write_imagef too
	double total_bytes = num_pixels * (bytes_per_read + bytes_per_write);
 	double kernel_bandwidth_GBps = (total_bytes / (1024.0 * 1024.0 * 1024.0)) / kernel_s;
	std::cout << "Kernel global-memory bandwidth (1 read + 1 write): "<< kernel_bandwidth_GBps << " GB/s" << std::endl;// Memory footprint
	
	// Footprints
	size_t host_mem = num_pixels*4 + num_pixels*4 + side*side*sizeof(float); // rgba_in + outRGBA + mask
	size_t device_mem = num_pixels*4 + num_pixels*4 + side*side*sizeof(float); // clImage_In + clImage_Out + clMask
	size_t kernel_mem = 2*4 + 16 + 16 + 5*4; // pos(int2) + acc(float3) + pix(float4) + a,b,masksize,width,height(int)
	std::cout << "Host memory footprint:   " << host_mem / (1024.0*1024.0)<< " MB" << std::endl;
	std::cout << "Device memory footprint in global data: " << device_mem / (1024.0*1024.0)<< " MB" << std::endl;
	std::cout << "Kernel memory footprint in private data per workitem: " << kernel_mem << " Bytes" << std::endl;

	return 0;
}

