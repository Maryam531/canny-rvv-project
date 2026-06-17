#include "image.h"
#include <cstdlib>
#include <cstdio>
#include <cstdint>

Image load_image(const char* path, int width, int height) {
    Image img;
    img.width  = width;
    img.height = height;
    
    // Enforce size to be a safe, clean multiple of 64 bytes
    size_t total_bytes = static_cast<size_t>(width) * height;
    size_t aligned_size = (total_bytes + 63) & ~size_t(63);
    
    img.data = (uint8_t*)aligned_alloc(64, aligned_size);
    if (!img.data) {
        printf("Error: Memory allocation failed in load_image\n");
        return img;
    }

    FILE* f = fopen(path, "rb");
    if (!f) {
        printf("Error: cannot open %s\n", path);
        free(img.data); // Clean up allocated memory on failure
        img.data = nullptr;
        return img;
    }
    
    size_t bytes_read = fread(img.data, 1, total_bytes, f);
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
    if (img.data) {
        free(img.data);
        img.data = nullptr;
    }
}
