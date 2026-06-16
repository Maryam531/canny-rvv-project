#ifndef IMAGE_IO_H
#define IMAGE_IO_H

#include "image.h"

Image load_image(const char* path, int width, int height);
void save_image(const char* path, const Image& img);
void free_image(Image& img);

#endif

