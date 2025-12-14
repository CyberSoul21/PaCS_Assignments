
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

// std::pair<size_t, std::string, size_t>
struct Args {
    float sigma;
    std::string image;
    float gpu_share;
};
Args usage(int argc, const char *argv[]) {
    // read the number of steps from the command line
    if (argc != 4) {
        std::cerr << "Invalid syntax: environ_gaussian_filter <sigma> <image> <gpu_share>" << std::endl;
        exit(1);
    }
    // size_t sigma = std::stoll(argv[1]);
    // std::string image = argv[2];
	// size_t gpu_share = std::stoll(argv[3]);

    // return std::make_pair(sigma, image, gpu_share);
	Args a;
    a.sigma = std::stof(argv[1]);
    a.image = argv[2];
    a.gpu_share = std::stof(argv[3]);

	if (a.gpu_share < 0.0f) a.gpu_share = 0.0f;
	if (a.gpu_share > 1.0f) a.gpu_share = 1.0f;
    return a;
}


int main(int argc, const char *argv[]){

	//Set arguments
	auto args = usage(argc, argv);
	float sigma = args.sigma;
	std::string image = args.image;
	float gpu_share = args.gpu_share;
	// auto ret_pair = usage(argc, argv);
    // float sigma = (float)ret_pair.first;
    // std::string image = ret_pair.second;
	// float gpu_share = (float)ret_pair.third;

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
	// cl_command_queue command_queue;     				// compute command queue
	cl_program program;                 				// compute program
	cl_kernel kernel_gpu;                   				// compute kernel
	cl_kernel kernel_cpu;                   				// compute kernel
	cl_device_id gpu_device = nullptr;
	cl_device_id cpu_device = nullptr;
	
	
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

			cl_device_type dtype;
        	clGetDeviceInfo(devices_ids[i][j], CL_DEVICE_TYPE, sizeof(dtype), &dtype, NULL);
			cl_error(err, "clGetDeviceInfo: Getting device type");

			// if ((dtype & CL_DEVICE_TYPE_GPU) && gpu_device == nullptr)
            // 	gpu_device = devices_ids[i][j];

			// if ((dtype & CL_DEVICE_TYPE_CPU) && cpu_device == nullptr)
			// 	cpu_device = devices_ids[i][j];
		}
	}	

	// Berlin case: both gpus
	gpu_device = devices_ids[0][0];
	cpu_device = devices_ids[0][1];

	// 3. Create a context, with a device
	cl_device_id context_devices[2] = { gpu_device, cpu_device };
	cl_context_properties properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platforms_ids[0], 0};
	// context = clCreateContext(properties, 1, devices_ids[0], NULL, NULL, &err);
	context = clCreateContext(properties, 2, context_devices, NULL, NULL, &err);
	cl_error(err, "Failed to create a compute context\n");

	// 4. Create a command queue
	cl_command_queue_properties proprt[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
	// command_queue = clCreateCommandQueueWithProperties(context, devices_ids[0][0], proprt, &err);
	cl_command_queue queue_gpu;
	cl_command_queue queue_cpu;
	queue_gpu = clCreateCommandQueueWithProperties(context, gpu_device, proprt, &err);
	cl_error(err, "Failed to create a command queue\n");
	queue_cpu = clCreateCommandQueueWithProperties(context, cpu_device, proprt, &err);
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
	kernel_gpu = clCreateKernel(program, "gaussian_filter", &err);
	kernel_cpu = clCreateKernel(program, "gaussian_filter", &err);
	cl_error(err, "Failed to create kernel from the program\n");

	// 6. Get image
	cimg_library::CImg<unsigned char> img(image.c_str());
	int width = img.width();
	int height = img.height();
	double num_pixels = (double)width * (double)height;
	std::cout<<"image"<<width<<std::endl;

	// Convert CImg planar RGB → interleaved RGB (OpenCL format)
    std::vector<unsigned char> rgba_in(num_pixels*4);
    cimg_forXY(img, x, y){
        int idx = 4*(y*width + x);
        rgba_in[idx+0] = img(x,y,0);
        rgba_in[idx+1] = img(x,y,1);
        rgba_in[idx+2] = img(x,y,2);
        rgba_in[idx+3] = 255;
    }

	// Replicating image 5000 times
	const size_t N_IMAGES = 5000; 
	const size_t image_bytes = width * height * 4;
	size_t gpu_images = (size_t)(gpu_share * N_IMAGES);
	size_t cpu_images = N_IMAGES - gpu_images;

	std::vector<unsigned char> stream_in(N_IMAGES * image_bytes);
	std::vector<unsigned char> stream_out(N_IMAGES * image_bytes);
	for (size_t i = 0; i < N_IMAGES; ++i) {
		memcpy(stream_in.data() + i * image_bytes,
			rgba_in.data(),
			image_bytes);
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
	cl_mem clImage_In_gpu = clCreateImage(context, CL_MEM_READ_ONLY, &img_format, &img_desc,NULL, &err);
	cl_mem clImage_In_cpu = clCreateImage(context, CL_MEM_READ_ONLY, &img_format, &img_desc,NULL, &err);
	cl_error(err, "Failed to create input image at device\n");
    // cl_event write_event;
	size_t origin[3] = {0,0,0};
	size_t region[3] = {(size_t)width, (size_t)height, 1};
	// err = clEnqueueWriteImage(command_queue,clImage_In,CL_TRUE,origin,region,0, 0,rgba_in.data(),0, NULL,&write_event);
	// cl_error(err, "Failed to write input image to device\n");

	// 9. Create read buffer
	cl_mem clImage_Out_gpu = clCreateImage(context, CL_MEM_WRITE_ONLY, &img_format, &img_desc, NULL, &err);
	cl_mem clImage_Out_cpu = clCreateImage(context, CL_MEM_WRITE_ONLY, &img_format, &img_desc, NULL, &err);
	cl_error(err, "Failed to create output image at device\n");
    
    // 10. Create buffer for mask and transfer it to the device
    cl_mem clMask = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*side*side, mask, &err);
	cl_error(err, "Failed to create mask buffer at device\n");

	// 12. Launch Kernel
	// cl_event kernel_event;
	// size_t local_size[2] = {16, 16};
	size_t global_size[2] = { (size_t)width, (size_t)height };
	// err = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_size, NULL, 0, NULL, &kernel_event);
	// cl_error(err, "Failed to launch kernel to the device\n");

	// 11. Set the arguments to the kernel
	err = clSetKernelArg(kernel_gpu, 0, sizeof(cl_mem), &clImage_In_gpu);
	cl_error(err, "Failed to set argument 0 gpu\n");
	err = clSetKernelArg(kernel_gpu, 1, sizeof(cl_mem), &clImage_Out_gpu);
	cl_error(err, "Failed to set argument 1 gpu\n");
	err = clSetKernelArg(kernel_gpu, 2, sizeof(cl_mem), &clMask);
	cl_error(err, "Failed to set argument 2 gpu\n");
	err = clSetKernelArg(kernel_gpu, 3, sizeof(int), &maskSize);
	cl_error(err, "Failed to set argument 3 gpu\n");
	err = clSetKernelArg(kernel_gpu, 4, sizeof(int), &width);
	cl_error(err, "Failed to set argument 4 gpu\n");
	err = clSetKernelArg(kernel_gpu, 5, sizeof(int), &height);
	cl_error(err, "Failed to set argument 5 gpu\n");

	// Enqueue GPU images
	double gpu_kernel_ms = 0.0, cpu_kernel_ms = 0.0;
	double gpu_h2d_ms = 0.0, cpu_h2d_ms = 0.0;
	double gpu_d2h_ms = 0.0, cpu_d2h_ms = 0.0;
	for (size_t i = 0; i < gpu_images; i++) {
		unsigned char* in_ptr  = stream_in.data()  + i * image_bytes;
		unsigned char* out_ptr = stream_out.data() + i * image_bytes;

		cl_event ev_write_gpu, ev_kernel_gpu, ev_read_gpu;

		clEnqueueWriteImage(queue_gpu, clImage_In_gpu, CL_FALSE,
							origin, region, 0, 0,
							in_ptr, 0, NULL, &ev_write_gpu);

		clEnqueueNDRangeKernel(queue_gpu, kernel_gpu, 2,
							NULL, global_size, NULL,
							1, &ev_write_gpu, &ev_kernel_gpu);

		clEnqueueReadImage(queue_gpu, clImage_Out_gpu, CL_FALSE,
						origin, region, 0, 0,
						out_ptr, 0, &ev_kernel_gpu, &ev_read_gpu);
		
		// This is necessary for the metrics, it serializes per image.
		clWaitForEvents(1, &ev_read_gpu);

		// Profiling
		cl_ulong s, e;

		clGetEventProfilingInfo(ev_write_gpu, CL_PROFILING_COMMAND_START, sizeof(s), &s, NULL);
		clGetEventProfilingInfo(ev_write_gpu, CL_PROFILING_COMMAND_END,   sizeof(e), &e, NULL);
		gpu_h2d_ms += (e - s) * 1e-6;

		clGetEventProfilingInfo(ev_kernel_gpu, CL_PROFILING_COMMAND_START, sizeof(s), &s, NULL);
		clGetEventProfilingInfo(ev_kernel_gpu, CL_PROFILING_COMMAND_END,   sizeof(e), &e, NULL);
		gpu_kernel_ms += (e - s) * 1e-6;

		clGetEventProfilingInfo(ev_read_gpu, CL_PROFILING_COMMAND_START, sizeof(s), &s, NULL);
		clGetEventProfilingInfo(ev_read_gpu, CL_PROFILING_COMMAND_END,   sizeof(e), &e, NULL);
		gpu_d2h_ms += (e - s) * 1e-6;

		clReleaseEvent(ev_write_gpu);
		clReleaseEvent(ev_kernel_gpu);
		clReleaseEvent(ev_read_gpu);
	}

	//Enqueue CPU images
	err = clSetKernelArg(kernel_cpu, 0, sizeof(cl_mem), &clImage_In_cpu);
	cl_error(err, "Failed to set argument 0 gpu\n");
	err = clSetKernelArg(kernel_cpu, 1, sizeof(cl_mem), &clImage_Out_cpu);
	cl_error(err, "Failed to set argument 1 gpu\n");
	err = clSetKernelArg(kernel_cpu, 2, sizeof(cl_mem), &clMask);
	cl_error(err, "Failed to set argument 2 gpu\n");
	err = clSetKernelArg(kernel_cpu, 3, sizeof(int), &maskSize);
	cl_error(err, "Failed to set argument 3 gpu\n");
	err = clSetKernelArg(kernel_cpu, 4, sizeof(int), &width);
	cl_error(err, "Failed to set argument 4 gpu\n");
	err = clSetKernelArg(kernel_cpu, 5, sizeof(int), &height);
	cl_error(err, "Failed to set argument 5 gpu\n");

	for (size_t i = gpu_images; i < N_IMAGES; i++) {
		unsigned char* in_ptr  = stream_in.data()  + i * image_bytes;
		unsigned char* out_ptr = stream_out.data() + i * image_bytes;

		cl_event ev_write_cpu, ev_kernel_cpu, ev_read_cpu;

		clEnqueueWriteImage(queue_cpu, clImage_In_cpu, CL_FALSE,
							origin, region, 0, 0,
							in_ptr, 0, NULL, &ev_write_cpu);

		clEnqueueNDRangeKernel(queue_cpu, kernel_cpu, 2,
							NULL, global_size, NULL,
							1, &ev_write_cpu, &ev_kernel_cpu);

		clEnqueueReadImage(queue_cpu, clImage_Out_cpu, CL_FALSE,
						origin, region, 0, 0,
						out_ptr, 0, &ev_kernel_cpu, &ev_read_cpu);
		
		// clWaitForEvents(1, &ev_read_cpu);

		// Profiling
		cl_ulong s, e;

		clGetEventProfilingInfo(ev_write_cpu, CL_PROFILING_COMMAND_START, sizeof(s), &s, NULL);
		clGetEventProfilingInfo(ev_write_cpu, CL_PROFILING_COMMAND_END,   sizeof(e), &e, NULL);
		cpu_h2d_ms += (e - s) * 1e-6;

		clGetEventProfilingInfo(ev_kernel_cpu, CL_PROFILING_COMMAND_START, sizeof(s), &s, NULL);
		clGetEventProfilingInfo(ev_kernel_cpu, CL_PROFILING_COMMAND_END,   sizeof(e), &e, NULL);
		cpu_kernel_ms += (e - s) * 1e-6;

		clGetEventProfilingInfo(ev_read_cpu, CL_PROFILING_COMMAND_START, sizeof(s), &s, NULL);
		clGetEventProfilingInfo(ev_read_cpu, CL_PROFILING_COMMAND_END,   sizeof(e), &e, NULL);
		cpu_d2h_ms += (e - s) * 1e-6;

		clReleaseEvent(ev_write_cpu);
		clReleaseEvent(ev_kernel_cpu);
		clReleaseEvent(ev_read_cpu);

	}

	clFinish(queue_gpu);
	clFinish(queue_cpu);

    // 13. Read output image 
    // std::vector<unsigned char> outRGBA(width*height*4);
	// cl_event read_event;
    // err = clEnqueueReadImage(command_queue, clImage_Out, CL_TRUE,origin, region,0, 0,outRGBA.data(),0, NULL, &read_event);
    // cl_error(err,"Failed to read output");

	// 14. Save output
	// cimg_library::CImg<unsigned char> outImg(width, height, 1, 3);
    // cimg_forXY(outImg, x,y) {
    //     int i = 4*(y*width + x);
    //     outImg(x,y,0) = outRGBA[i+0];
    //     outImg(x,y,1) = outRGBA[i+1];
    //     outImg(x,y,2) = outRGBA[i+2];
    // }
	
	// std::string result = "result_" + std::to_string(sigma) + "_" + image;
    // outImg.save(result.c_str());
	unsigned char* out0 = stream_out.data();
	cimg_library::CImg<unsigned char> outImg(width, height, 1, 3);
	cimg_forXY(outImg, x,y) {
		int i = 4*(y*width + x);
		outImg(x,y,0) = out0[i+0];
		outImg(x,y,1) = out0[i+1];
		outImg(x,y,2) = out0[i+2];
	}
	outImg.save("result_stream.jpg");
    std::cout << "Gaussian filter saved\n"<<std::endl;

	// 15. Released cl objects
	clReleaseMemObject(clMask);
	clReleaseMemObject(clImage_Out_gpu);
	clReleaseMemObject(clImage_In_gpu);
	clReleaseMemObject(clImage_Out_cpu);
	clReleaseMemObject(clImage_In_cpu);
	clReleaseProgram(program);
	clReleaseKernel(kernel_gpu);
	clReleaseKernel(kernel_cpu);
	clReleaseCommandQueue(queue_gpu);
	clReleaseCommandQueue(queue_cpu);
	clReleaseContext(context);
	
	// 16. Metrics
	// Complete program time
	auto t_program_end = high_resolution_clock::now();
    double program_ms = duration<double, std::milli>(t_program_end - t_program_start).count();
    std::cout << "sigma: " << sigma << ", image: " <<image<<std::endl;
	std::cout << "matrix dimention: " << side <<std::endl;
	std::cout << "Program time: " << program_ms << " ms" << std::endl;


	std::cout << "\n=== GPU timings ===\n";
	std::cout << "H2D:    " << gpu_h2d_ms << " ms\n";
	std::cout << "Kernel: " << gpu_kernel_ms << " ms\n";
	std::cout << "D2H:    " << gpu_d2h_ms << " ms\n";
	std::cout << "Total:  " << (gpu_h2d_ms + gpu_kernel_ms + gpu_d2h_ms) << " ms\n";

	std::cout << "\n=== CPU timings ===\n";
	std::cout << "H2D:    " << cpu_h2d_ms << " ms\n";
	std::cout << "Kernel: " << cpu_kernel_ms << " ms\n";
	std::cout << "D2H:    " << cpu_d2h_ms << " ms\n";
	std::cout << "Total:  " << (cpu_h2d_ms + cpu_kernel_ms + cpu_d2h_ms) << " ms\n";
	// // Kernel time
	// cl_ulong start_ns = 0, end_ns = 0;
	// clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(start_ns), &start_ns, NULL);
	// clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END,	sizeof(end_ns), &end_ns, NULL);
	// clReleaseEvent(kernel_event);
	// double kernel_ms = (end_ns - start_ns) * 1e-6;
	// std::cout << "Kernel time: " << kernel_ms << " ms" << std::endl;

	// // Measure input image time and bandwidth
	// cl_ulong w_start = 0, w_end = 0;
	// clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_START,sizeof(w_start), &w_start, NULL);
	// clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_END,sizeof(w_end), &w_end, NULL);
	// clReleaseEvent(write_event);
	// double write_ms = (w_end - w_start) * 1e-6;
	// size_t bytes_h2d = num_pixels * 4; // 4 bytes per pixel (RGBA)
	// double bandwidth_h2d_GBps = (bytes_h2d / (1024.0*1024.0*1024.0)) / (write_ms / 1000.0);
	// std::cout << "Host->Device: " << write_ms << " ms, "<< bandwidth_h2d_GBps << " GB/s" << std::endl;

	// // Measure output image time and bandwidth
	// cl_ulong r_start = 0, r_end = 0;
	// clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START,sizeof(r_start), &r_start, NULL);
	// clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END,sizeof(r_end), &r_end, NULL);
	// clReleaseEvent(read_event);
	// double read_ms = (r_end - r_start) * 1e-6;
	// size_t bytes_d2h = num_pixels * 4;
	// double bandwidth_d2h_GBps = (bytes_d2h / (1024.0*1024.0*1024.0)) / (read_ms / 1000.0);
	// std::cout << "Device->Host: " << read_ms << " ms, "<< bandwidth_d2h_GBps << " GB/s" << std::endl;

	// //Kernel troughput
	// double kernel_s = kernel_ms / 1000.0;
	// double pixels_per_sec = num_pixels / kernel_s; //each pixel corresponds to a work-item, kernel time is what it takes from the first pixel to the last 
	// std::cout << "Throughput (pixels/s): " << pixels_per_sec << std::endl;
	// // This could be GFLOP/s if we assume an operation to be 1 mul + 1 add, so 2 FLOPs per color (we compute 3 colors)
	// double flops_per_pixel = 2.0 * (double)side * (double)side * 3.0;
	// double total_flops = num_pixels * flops_per_pixel; //We are assuming all pixels do the same operations, it would be less considering that pixels on the edges have less operations
	// double gflops = (total_flops / kernel_s) / 1e9;
	// std::cout << "Kernel throughput: " << gflops << " GFLOP/s" << std::endl;

	// // Kernel bandwithd
	// double bytes_per_read = 16.0;  // Read_imagef in kernel returns a float4 (4 components of 4 bytes) in private memory
	// double bytes_per_write = 16.0; // write_imagef too
	// double total_bytes = num_pixels * (bytes_per_read + bytes_per_write);
 	// double kernel_bandwidth_GBps = (total_bytes / (1024.0 * 1024.0 * 1024.0)) / kernel_s;
	// std::cout << "Kernel global-memory bandwidth (1 read + 1 write): "<< kernel_bandwidth_GBps << " GB/s" << std::endl;// Memory footprint
	
	// // Footprints
	// size_t host_mem = num_pixels*4 + num_pixels*4 + side*side*sizeof(float); // rgba_in + outRGBA + mask
	// size_t device_mem = num_pixels*4 + num_pixels*4 + side*side*sizeof(float); // clImage_In + clImage_Out + clMask
	// size_t kernel_mem = 2*4 + 16 + 16 + 5*4; // pos(int2) + acc(float3) + pix(float4) + a,b,masksize,width,height(int)
	// std::cout << "Host memory footprint:   " << host_mem / (1024.0*1024.0)<< " MB" << std::endl;
	// std::cout << "Device memory footprint in global data: " << device_mem / (1024.0*1024.0)<< " MB" << std::endl;
	// std::cout << "Kernel memory footprint in private data per workitem: " << kernel_mem << " Bytes" << std::endl;

	return 0;
}

