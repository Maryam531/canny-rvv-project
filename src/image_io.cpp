#include "image.h"
#include <cstdlib>
#include <cstdio>

Image load_image(const char* path, int width, int height) {
    Image img;
    img.width  = width;
    img.height = height;
    img.data   = (uint8_t*)aligned_alloc(64, width * height);

    FILE* f = fopen(path, "rb");
    if (!f) {
        printf("Error: cannot open %s\n", path);
        img.data = nullptr;
        return img;
    }
size_t bytes_read = fread(img.data, 1, width * height, f);
(void)bytes_read; 
    fclose(f);
    return img;
}

void save_image(const char* path, const Image& img) {
    FILE* f = fopen(path, "wb");
    if (!f) {
        printf("Error: cannot save %s\n", path);
        return;
    }
    fwrite(img.data, 1, img.width * img.height, f);
    fclose(f);
}

void free_image(Image& img) {
    free(img.data);
    img.data = nullptr;
}
