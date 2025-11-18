#define cimg_use_jpeg

#include <iostream>
//#include "/home/javier/Documents/PaCS/labs/CImg/CImg.h"//Javier Path
#include "CImg.h"//Javier Path
using namespace cimg_library;

/*

g++ example.cpp -o example -I[PATH_TO_CIMG] -lm -lpthread -lX11 -ljpeg
g++ example.cpp -o example -I /home/javier/Documents/PaCS/labs/CImg -lm -lpthread -lX11 -ljpeg

Fix: 
/home/javier/Documents/PaCS/labs/CImg/CImg.h:500:10: fatal error: jpeglib.h: No such file or directory
  500 | #include "jpeglib.h"

sudo apt install libjpeg-dev


*/

int main(){
  CImg<unsigned char> img("image.jpg");  // Load image file "image.jpg" at object img

  std::cout << "Image width: " << img.width() << "Image height: " << img.height() << "Number of slices: " << img.depth() << "Number of channels: " << img.spectrum() << std::endl;  //dump some characteristics of the loaded image

  int i = 0;//XXX;
  int j = 0;//XXX;
  std::cout << std::hex << (int) img(i, j, 0, 0) << std::endl;  //print pixel value for channel 0 (red) 
  std::cout << std::hex << (int) img(i, j, 0, 1) << std::endl;  //print pixel value for channel 1 (green) 
  std::cout << std::hex << (int) img(i, j, 0, 2) << std::endl;  //print pixel value for channel 2 (blue) 
  
  img.display("My first CImg code");             // Display the image in a display window

  return 0;

}
