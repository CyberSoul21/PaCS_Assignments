__kernel void gaussian_filter(
  __global unsigned char *input_image,
  __global unsigned char *output_image,
  const unsigned int width,
  const unsigned int height,
  __constant float *gaussian_kernel,
  const unsigned int kernel_size){

  int x = get_global_id(0);
  int y = get_global_id(1);

  if(x < width && y < height){
    int half_kernel = kernel_size / 2;
    float sum = 0.0f;
    
    // Apply convolution
    for(int ky = -half_kernel; ky <= half_kernel; ky++){
      for(int kx = -half_kernel; kx <= half_kernel; kx++){
        int image_x = x + kx;
        int image_y = y + ky;
        
        // Handle border pixels (clamp to edge)
        if(image_x < 0) image_x = 0;
        if(image_x >= width) image_x = width - 1;
        if(image_y < 0) image_y = 0;
        if(image_y >= height) image_y = height - 1;
        
        int image_idx = image_y * width + image_x;
        int kernel_idx = (ky + half_kernel) * kernel_size + (kx + half_kernel);
        
        sum += input_image[image_idx] * gaussian_kernel[kernel_idx];
      }
    }
    
    int output_idx = y * width + x;
    output_image[output_idx] = (unsigned char)(sum);
  }
}