#include "image.h"
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <cerrno>
#include <cstring>

Image load_image(const char* path, int width, int height)
{
    Image img;

    img.width  = width;
    img.height = height;

    size_t total_bytes  = static_cast<size_t>(width) * height;
    size_t aligned_size = (total_bytes + 63) & ~size_t(63);

    img.data = (uint8_t*)aligned_alloc(64, aligned_size);

    if (!img.data)
    {
        printf("Error: Memory allocation failed.\n");
        return img;
    }

    FILE* f = fopen(path, "rb");

    if (!f)
    {
        printf("\n=====================================\n");
        printf("           fopen() FAILED\n");
        printf("=====================================\n");
        printf("Path  : %s\n", path);
        printf("errno : %d\n", errno);
        printf("Error : %s\n", strerror(errno));
        printf("=====================================\n");

        free(img.data);
        img.data = nullptr;

        return img;
    }

    size_t bytes_read = fread(img.data, 1, total_bytes, f);

    if (bytes_read != total_bytes)
    {
        printf("Warning: expected %zu bytes but read %zu bytes.\n",
               total_bytes,
               bytes_read);
    }

    fclose(f);

    return img;
}

void save_image(const char* path, const Image& img)
{
    FILE* f = fopen(path, "wb");

    if (!f)
    {
        printf("Error: cannot save %s\n", path);
        return;
    }

    fwrite(img.data, 1, img.width * img.height, f);

    fclose(f);
}

void free_image(Image& img)
{
    if (img.data)
    {
        free(img.data);
        img.data = nullptr;
    }
}
