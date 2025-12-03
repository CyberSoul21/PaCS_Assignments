////////////////////////////////////////////////////////////////////
//File: gaussian_filter.cpp
//
//Description: Gaussian filter implementation using OpenCL and CImg
//
// Compile: g++ gaussian_filter.cpp -o gaussian_filter -I[PATH_TO_CIMG] -lOpenCL -lm -lpthread -lX11 -ljpeg
// Example: g++ gaussian_filter.cpp -o gaussian_filter -I /home/javier/Documents/PaCS/labs/CImg -lOpenCL -lm -lpthread -lX11 -ljpeg
////////////////////////////////////////////////////////////////////

#define cimg_use_jpeg

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <iostream>
#include "CImg.h"

#ifdef __APPLE__
  #include <OpenCL/opencl.h>
#else
  #include <CL/cl.h>
#endif

using namespace cimg_library;

// check error, in such a case, it exits
void cl_error(cl_int code, const char *string){
  if (code != CL_SUCCESS){
    printf("%d - %s\n", code, string);
    exit(-1);
  }
}

// Generate a Gaussian kernel with given size and sigma
void generate_gaussian_kernel(float *kernel, int kernel_size, float sigma){
  float sum = 0.0f;
  int half_kernel = kernel_size / 2;
  
  // Calculate Gaussian values
  for(int y = -half_kernel; y <= half_kernel; y++){
    for(int x = -half_kernel; x <= half_kernel; x++){
      float value = (1.0f / (2.0f * M_PI * sigma * sigma)) * 
                    exp(-(x*x + y*y) / (2.0f * sigma * sigma));
      int idx = (y + half_kernel) * kernel_size + (x + half_kernel);
      kernel[idx] = value;
      sum += value;
    }
  }
  
  // Normalize the kernel
  for(int i = 0; i < kernel_size * kernel_size; i++){
    kernel[i] /= sum;
  }
  
  // Print kernel for verification
  printf("Gaussian Kernel (%dx%d, sigma=%.2f):\n", kernel_size, kernel_size, sigma);
  for(int y = 0; y < kernel_size; y++){
    printf("[ ");
    for(int x = 0; x < kernel_size; x++){
      printf("%.4f ", kernel[y * kernel_size + x]);
    }
    printf("]\n");
  }
  printf("\n");
}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv)
{
  int err;
  size_t t_buf = 50;
  char str_buffer[t_buf];
  size_t e_buf;
  
  size_t global_size[2];
  size_t local_size[2];

  const cl_uint num_platforms_ids = 10;
  cl_platform_id platforms_ids[num_platforms_ids];
  cl_uint n_platforms;
  const cl_uint num_devices_ids = 10;
  cl_device_id devices_ids[num_platforms_ids][num_devices_ids];
  cl_uint n_devices[num_platforms_ids];
  
  cl_device_id device_id;
  cl_context context;
  cl_command_queue command_queue;
  cl_program program;
  cl_kernel kernel_r, kernel_g, kernel_b;
  
  // Load image using CImg
  printf("Loading image...\n");
  CImg<unsigned char> img("image.jpg");
  
  int width = img.width();
  int height = img.height();
  int channels = img.spectrum();
  int image_size = width * height;
  
  printf("Image loaded successfully!\n");
  printf("  Width: %d\n", width);
  printf("  Height: %d\n", height);
  printf("  Channels: %d\n", channels);
  printf("  Depth: %d\n\n", img.depth());
  
  // Display original image
  img.display("Original Image");
  
  // Gaussian kernel parameters
  int kernel_size = 5; // Use 5x5 kernel for better smoothing (can be 3, 5, 7, etc.)
  float sigma = 1.5f;
  int gaussian_kernel_elements = kernel_size * kernel_size;
  
  // Allocate memory for each channel
  unsigned char *input_r = (unsigned char*)malloc(image_size * sizeof(unsigned char));
  unsigned char *input_g = (unsigned char*)malloc(image_size * sizeof(unsigned char));
  unsigned char *input_b = (unsigned char*)malloc(image_size * sizeof(unsigned char));
  unsigned char *output_r = (unsigned char*)malloc(image_size * sizeof(unsigned char));
  unsigned char *output_g = (unsigned char*)malloc(image_size * sizeof(unsigned char));
  unsigned char *output_b = (unsigned char*)malloc(image_size * sizeof(unsigned char));
  float *gaussian_kernel = (float*)malloc(gaussian_kernel_elements * sizeof(float));
  
  // Extract RGB channels from CImg (CImg stores data in planar format)
  for(int j = 0; j < height; j++){
    for(int i = 0; i < width; i++){
      int idx = j * width + i;
      input_r[idx] = img(i, j, 0, 0); // Red channel
      input_g[idx] = img(i, j, 0, 1); // Green channel
      input_b[idx] = img(i, j, 0, 2); // Blue channel
    }
  }
  
  printf("Generating Gaussian kernel...\n");
  generate_gaussian_kernel(gaussian_kernel, kernel_size, sigma);

  // 1. Scan the available platforms
  err = clGetPlatformIDs(num_platforms_ids, platforms_ids, &n_platforms);
  cl_error(err, "Error: Failed to Scan for Platforms IDs");
  printf("Number of available platforms: %d\n\n", n_platforms);

  for(int i = 0; i < n_platforms; i++){
    err = clGetPlatformInfo(platforms_ids[i], CL_PLATFORM_NAME, t_buf, str_buffer, &e_buf);
    cl_error(err, "Error: Failed to get info of the platform\n");
    printf("\t[%d]-Platform Name: %s\n", i, str_buffer);
  }
  printf("\n");

  // 2. Scan for devices in each platform
  for(int i = 0; i < n_platforms; i++){
    err = clGetDeviceIDs(platforms_ids[i], CL_DEVICE_TYPE_ALL, num_devices_ids, devices_ids[i], &(n_devices[i]));
    cl_error(err, "Error: Failed to Scan for Devices IDs");
    printf("\t[%d]-Platform. Number of available devices: %d\n", i, n_devices[i]);

    for(int j = 0; j < n_devices[i]; j++){
      err = clGetDeviceInfo(devices_ids[i][j], CL_DEVICE_NAME, sizeof(str_buffer), &str_buffer, NULL);
      cl_error(err, "clGetDeviceInfo: Getting device name");
      printf("\t\t [%d]-Platform [%d]-Device CL_DEVICE_NAME: %s\n", i, j, str_buffer);

      cl_uint max_compute_units_available;
      err = clGetDeviceInfo(devices_ids[i][j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(max_compute_units_available), &max_compute_units_available, NULL);
      cl_error(err, "clGetDeviceInfo: Getting device max compute units available");
      printf("\t\t [%d]-Platform [%d]-Device CL_DEVICE_MAX_COMPUTE_UNITS: %d\n\n", i, j, max_compute_units_available);
    }
  }

  // 3. Create a context, with a device
  cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platforms_ids[0], 0};
  context = clCreateContext(properties, 1, devices_ids[0], NULL, NULL, &err);
  cl_error(err, "Failed to create a compute context\n");

  // 4. Create a command queue
  cl_command_queue_properties proprt[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
  command_queue = clCreateCommandQueueWithProperties(context, devices_ids[0][0], proprt, &err);
  cl_error(err, "Failed to create a command queue\n");

  // 5. Read an OpenCL program from the file gaussian_kernel.cl
  FILE *fileHandler = fopen("gaussian_kernel.cl", "r");
  if(!fileHandler){
    printf("Error: Could not open kernel file gaussian_kernel.cl\n");
    exit(-1);
  }
  fseek(fileHandler, 0, SEEK_END);
  size_t fileSize = ftell(fileHandler);
  rewind(fileHandler);

  char *sourceCode = (char*)malloc(fileSize + 1);
  sourceCode[fileSize] = '\0';
  fread(sourceCode, sizeof(char), fileSize, fileHandler);
  fclose(fileHandler);

  program = clCreateProgramWithSource(context, 1, (const char**)&sourceCode, &fileSize, &err);
  cl_error(err, "Failed to create program with source\n");
  free(sourceCode);

  // Build the executable
  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if(err != CL_SUCCESS){
    size_t len;
    char buffer[2048];
    printf("Error: Some error at building process.\n");
    clGetProgramBuildInfo(program, devices_ids[0][0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
    printf("%s\n", buffer);
    exit(-1);
  }

  // Create kernel (we'll use the same kernel for all channels)
  kernel_r = clCreateKernel(program, "gaussian_filter", &err);
  cl_error(err, "Failed to create kernel from the program\n");

  // Create OpenCL buffers for Red channel
  cl_mem input_device_r = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(unsigned char) * image_size, NULL, &err);
  cl_error(err, "Failed to create input memory buffer at device\n");
  cl_mem output_device_r = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(unsigned char) * image_size, NULL, &err);
  cl_error(err, "Failed to create output memory buffer at device\n");
  
  // Create OpenCL buffers for Green channel
  cl_mem input_device_g = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(unsigned char) * image_size, NULL, &err);
  cl_error(err, "Failed to create input memory buffer at device\n");
  cl_mem output_device_g = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(unsigned char) * image_size, NULL, &err);
  cl_error(err, "Failed to create output memory buffer at device\n");
  
  // Create OpenCL buffers for Blue channel
  cl_mem input_device_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(unsigned char) * image_size, NULL, &err);
  cl_error(err, "Failed to create input memory buffer at device\n");
  cl_mem output_device_b = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(unsigned char) * image_size, NULL, &err);
  cl_error(err, "Failed to create output memory buffer at device\n");
  
  // Create buffer for Gaussian kernel
  cl_mem kernel_device = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * gaussian_kernel_elements, NULL, &err);
  cl_error(err, "Failed to create kernel memory buffer at device\n");

  // Write gaussian kernel to device (shared by all channels)
  err = clEnqueueWriteBuffer(command_queue, kernel_device, CL_TRUE, 0, sizeof(float) * gaussian_kernel_elements, gaussian_kernel, 0, NULL, NULL);
  cl_error(err, "Failed to enqueue write command for gaussian kernel\n");

  // Launch kernel for each color channel
  local_size[0] = 16;
  local_size[1] = 16;
  global_size[0] = width;
  global_size[1] = height;
  
  printf("Processing Red channel...\n");
  // Write red channel input
  err = clEnqueueWriteBuffer(command_queue, input_device_r, CL_TRUE, 0, sizeof(unsigned char) * image_size, input_r, 0, NULL, NULL);
  cl_error(err, "Failed to enqueue write command for red channel\n");
  
  // Set kernel arguments for red channel
  err = clSetKernelArg(kernel_r, 0, sizeof(cl_mem), &input_device_r);
  cl_error(err, "Failed to set argument 0\n");
  err = clSetKernelArg(kernel_r, 1, sizeof(cl_mem), &output_device_r);
  cl_error(err, "Failed to set argument 1\n");
  err = clSetKernelArg(kernel_r, 2, sizeof(unsigned int), &width);
  cl_error(err, "Failed to set argument 2\n");
  err = clSetKernelArg(kernel_r, 3, sizeof(unsigned int), &height);
  cl_error(err, "Failed to set argument 3\n");
  err = clSetKernelArg(kernel_r, 4, sizeof(cl_mem), &kernel_device);
  cl_error(err, "Failed to set argument 4\n");
  err = clSetKernelArg(kernel_r, 5, sizeof(unsigned int), &kernel_size);
  cl_error(err, "Failed to set argument 5\n");
  
  // Launch kernel for red channel
  err = clEnqueueNDRangeKernel(command_queue, kernel_r, 2, NULL, global_size, local_size, 0, NULL, NULL);
  cl_error(err, "Failed to launch kernel for red channel\n");
  
  // Read red channel output
  err = clEnqueueReadBuffer(command_queue, output_device_r, CL_TRUE, 0, sizeof(unsigned char) * image_size, output_r, 0, NULL, NULL);
  cl_error(err, "Failed to read red channel output\n");
  
  printf("Processing Green channel...\n");
  // Write green channel input
  err = clEnqueueWriteBuffer(command_queue, input_device_g, CL_TRUE, 0, sizeof(unsigned char) * image_size, input_g, 0, NULL, NULL);
  cl_error(err, "Failed to enqueue write command for green channel\n");
  
  // Set kernel arguments for green channel
  err = clSetKernelArg(kernel_r, 0, sizeof(cl_mem), &input_device_g);
  err = clSetKernelArg(kernel_r, 1, sizeof(cl_mem), &output_device_g);
  
  // Launch kernel for green channel
  err = clEnqueueNDRangeKernel(command_queue, kernel_r, 2, NULL, global_size, local_size, 0, NULL, NULL);
  cl_error(err, "Failed to launch kernel for green channel\n");
  
  // Read green channel output
  err = clEnqueueReadBuffer(command_queue, output_device_g, CL_TRUE, 0, sizeof(unsigned char) * image_size, output_g, 0, NULL, NULL);
  cl_error(err, "Failed to read green channel output\n");
  
  printf("Processing Blue channel...\n");
  // Write blue channel input
  err = clEnqueueWriteBuffer(command_queue, input_device_b, CL_TRUE, 0, sizeof(unsigned char) * image_size, input_b, 0, NULL, NULL);
  cl_error(err, "Failed to enqueue write command for blue channel\n");
  
  // Set kernel arguments for blue channel
  err = clSetKernelArg(kernel_r, 0, sizeof(cl_mem), &input_device_b);
  err = clSetKernelArg(kernel_r, 1, sizeof(cl_mem), &output_device_b);
  
  // Launch kernel for blue channel
  err = clEnqueueNDRangeKernel(command_queue, kernel_r, 2, NULL, global_size, local_size, 0, NULL, NULL);
  cl_error(err, "Failed to launch kernel for blue channel\n");
  
  // Read blue channel output
  err = clEnqueueReadBuffer(command_queue, output_device_b, CL_TRUE, 0, sizeof(unsigned char) * image_size, output_b, 0, NULL, NULL);
  cl_error(err, "Failed to read blue channel output\n");

  printf("Gaussian filter applied successfully!\n\n");
  
  // Create output image and copy filtered channels
  CImg<unsigned char> output_img(width, height, 1, 3);
  for(int j = 0; j < height; j++){
    for(int i = 0; i < width; i++){
      int idx = j * width + i;
      output_img(i, j, 0, 0) = output_r[idx]; // Red channel
      output_img(i, j, 0, 1) = output_g[idx]; // Green channel
      output_img(i, j, 0, 2) = output_b[idx]; // Blue channel
    }
  }
  
  // Save output image
  output_img.save("output_image.jpg");
  printf("Output image saved as 'output_image.jpg'\n\n");
  
  // Display output image
  output_img.display("Filtered Image");
  
  // Sample pixel values
  int sample_x = width / 2;
  int sample_y = height / 2;
  printf("Sample pixel values at position (%d, %d):\n", sample_x, sample_y);
  printf("  Input RGB:  (%d, %d, %d)\n", 
         img(sample_x, sample_y, 0, 0), 
         img(sample_x, sample_y, 0, 1), 
         img(sample_x, sample_y, 0, 2));
  printf("  Output RGB: (%d, %d, %d)\n\n", 
         output_img(sample_x, sample_y, 0, 0), 
         output_img(sample_x, sample_y, 0, 1), 
         output_img(sample_x, sample_y, 0, 2));

  // Cleanup
  clReleaseMemObject(input_device_r);
  clReleaseMemObject(output_device_r);
  clReleaseMemObject(input_device_g);
  clReleaseMemObject(output_device_g);
  clReleaseMemObject(input_device_b);
  clReleaseMemObject(output_device_b);
  clReleaseMemObject(kernel_device);
  clReleaseProgram(program);
  clReleaseKernel(kernel_r);
  clReleaseCommandQueue(command_queue);
  clReleaseContext(context);
  
  free(input_r);
  free(input_g);
  free(input_b);
  free(output_r);
  free(output_g);
  free(output_b);
  free(gaussian_kernel);

  printf("Done!\n");
  
  return 0;
}