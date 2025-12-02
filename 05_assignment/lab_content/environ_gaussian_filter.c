

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
#ifdef __APPLE__
	  #include <OpenCL/opencl.h>
#else
	  #include <CL/cl.h>
#endif
	  
// check error, in such a case, it exits

void cl_error(cl_int code, const char *string){
	if (code != CL_SUCCESS){
		printf("%d - %s\n", code, string);
		exit(-1);
	}
}
////////////////////////////////////////////////////////////////////////////////

float * createMask(float sigma, int& masksize, int& radius){
	// int maskSize = (int)ceil(sigma); // If we want a dynamic size
	masksize = 3;
	radius = 1;
	float * mask = new float[masksize * masksize];
	float sum = 0.0f;
	float denom = (2*sigma*sigma);
	int idx = 0;
	for(int x = -radius; x<=radius;x++){
		for(int y = -radius; y<=radius;y++){
			float temp = exp(-((float)(x*x+y*y) / denom));
			sum += temp;
			mask[idx++] = temp;
		}
	}

	// Normalization
	for(int i = 0;i<masksize*masksize;i++){
		mask[i] = mask[i] / sum;
	}
	
	// Return Matrix dimension, in case we compute it from sigma
	return mask;

}


int main(int argc, char** argv)
{
	int err;                            	// error code returned from api calls
	size_t t_buf = 50;			// size of str_buffer
	char str_buffer[t_buf];		// auxiliary buffer	
	size_t e_buf;				// effective size of str_buffer in use
	size_t program_Size;			// size of the opencl program
		
	// size_t global_size;                      	// global domain size for our calculation
	// size_t local_size;                       	// local domain size for our calculation

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
	FILE *fileHandler = fopen("kernel_gaussian_filter.cl", "r");
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
	printf("nKernel source:\n\n%s\n", kernelSource);
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

	// Get image
	cimg_library::CImg<unsigned char> img("cats.jpg");
	int width = img.width();
	int height = img.height();

	// Convert CImg planar RGB → interleaved RGB (OpenCL format)
    std::vector<unsigned char> rgb_in(width*height*3);
    cimg_forXY(img, x, y){
        int idx = 3*(y*width + x);
        rgb_in[idx+0] = img(x,y,0);
        rgb_in[idx+1] = img(x,y,1);
        rgb_in[idx+2] = img(x,y,2);
    }

	// Create Gaussian mask
    int maskSize, radius;
    float * mask = createMask(1.f, maskSize, radius);

	// Create OpenCL buffer visible to the OpenCl runtime
	cl_image_format img_format;
	img_format.image_channel_order = CL_RGB;
	img_format.image_channel_data_type = CL_UNORM_INT8;

	cl_image_desc img_desc;
	memset(&img_desc, 0, sizeof(img_desc));   // MUY IMPORTANTE

	img_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
	img_desc.image_width = width;
	img_desc.image_height = height;
	img_desc.image_depth = 1;
	img_desc.image_array_size = 1;

	// cl_mem clImage_In = clCreateImage2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, img_format, width, height, 0, img.data(), &err);
	cl_mem clImage_In = clCreateImage(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &img_format, &img_desc,rgb_in.data(), &err);
	cl_error(err, "Failed to create input image at device\n");
    // cl_mem clImage_Out = clCreateImage2D(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, img_format, width, height, 0, NULL, &err);
    cl_mem clImage_Out = clCreateImage(context, CL_MEM_WRITE_ONLY, &img_format, &img_desc, NULL, &err);
	cl_error(err, "Failed to create output image at device\n");
    
    // Create buffer for mask and transfer it to the device
    cl_mem clMask = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*maskSize*maskSize, mask, &err);
	cl_error(err, "Failed to create mask buffer at device\n");
	// err = clEnqueueWriteBuffer(command_queue, clMask, CL_TRUE, 0, sizeof(float)*maskSize*maskSize, mask, 0, NULL, NULL);
	// cl_error(err, "Failed to enqueue a write command\n");

	// Set the arguments to the kernel
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &clImage_In);
	cl_error(err, "Failed to set argument 0\n");
	err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &clImage_Out);
	cl_error(err, "Failed to set argument 1\n");
	err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &clMask);
	cl_error(err, "Failed to set argument 2\n");
	err = clSetKernelArg(kernel, 3, sizeof(int), &radius);
	cl_error(err, "Failed to set argument 3\n");


	// Launch Kernel
	size_t global_size[2] = { (size_t)width, (size_t)height};
	// err = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
	err = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_size, NULL, 0, NULL, NULL);
	cl_error(err, "Failed to launch kernel to the device\n");

    // Read output image
    size_t origin[3] = {0,0,0};
    size_t region[3] = {(size_t)width, (size_t)height, 1};

    std::vector<unsigned char> outRGB(width*height*3);

    err = clEnqueueReadImage(command_queue, clImage_Out, CL_TRUE,
                    origin, region,
                    0, 0,
                    outRGB.data(),
                    0, NULL, NULL);
    cl_error(err,"Failed to read output");

	// Save output
	cimg_library::CImg<unsigned char> outImg(width, height, 1, 3);
    cimg_forXY(outImg, x,y) {
        int i = 3*(y*width + x);
        outImg(x,y,0) = outRGB[i+0];
        outImg(x,y,1) = outRGB[i+1];
        outImg(x,y,2) = outRGB[i+2];
    }

    outImg.save("result.jpg");
    std::cout << "Gaussian filter saved to result.jpg\n"<<std::endl;

	// Read data form device memory back to host memory
	// err = clEnqueueReadBuffer(command_queue, out_device_object, CL_TRUE, 0, sizeof(float) * count, out_host_object, 0, NULL, NULL);
	// cl_error(err, "Failed to enqueue a read command\n");

	clReleaseMemObject(clMask);
	clReleaseMemObject(clImage_Out);
	clReleaseMemObject(clImage_In);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
	return 0;
	}

