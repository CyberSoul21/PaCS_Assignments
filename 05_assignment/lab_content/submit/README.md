In order to execute the file:

CImg library needs to be on the same folder as the .c
You can create a folder build, execute cmake .. -G "Unix Makefiles" and make on that folder
the executabl ehas 2 arguments: gaussian_filter <sigma> <image.jpg>
Apparently there is an issue with CImg that is not able to read images from other folder. 
you can execute gaussian_filter from the parent folder like ./build/gaussian_filter 1 cat_250x334.jpg 
but from build like ./gaussian_filter 1 ../cat_250x334.jpg it gives a coredump