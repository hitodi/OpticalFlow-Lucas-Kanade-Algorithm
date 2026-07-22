#include <CoreFoundation/CoreFoundation.h>
#include <CoreGraphics/CoreGraphics.h>
#include <ImageIO/ImageIO.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int kWindowSize = 5;
constexpr int kStepSize = 10;
constexpr float kMinEigenvalue = 1.0e-4f;

struct Image {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> rgba;
};

std::string fallbackPath(const std::string& path)
{
    FILE* file = std::fopen(path.c_str(), "rb");
    if (file != nullptr) {
        std::fclose(file);
        return path;
    }

    return "test/" + path;
}

Image loadPng(const std::string& input_path)
{
    const std::string path = fallbackPath(input_path);
    CFURLRef url = CFURLCreateFromFileSystemRepresentation(
        kCFAllocatorDefault,
        reinterpret_cast<const UInt8*>(path.c_str()),
        path.size(),
        false);
    if (url == nullptr) {
        return {};
    }

    CGImageSourceRef source = CGImageSourceCreateWithURL(url, nullptr);
    CFRelease(url);
    if (source == nullptr) {
        return {};
    }

    CGImageRef source_image = CGImageSourceCreateImageAtIndex(source, 0, nullptr);
    CFRelease(source);
    if (source_image == nullptr) {
        return {};
    }

    Image image;
    image.width = static_cast<int>(CGImageGetWidth(source_image));
    image.height = static_cast<int>(CGImageGetHeight(source_image));
    image.rgba.resize(static_cast<std::size_t>(image.width) * image.height * 4);

    CGColorSpaceRef color_space = CGColorSpaceCreateDeviceRGB();
    CGContextRef context = CGBitmapContextCreate(
        image.rgba.data(),
        image.width,
        image.height,
        8,
        static_cast<std::size_t>(image.width) * 4,
        color_space,
        kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
    CGColorSpaceRelease(color_space);

    if (context == nullptr) {
        CGImageRelease(source_image);
        return {};
    }

    CGContextDrawImage(context, CGRectMake(0, 0, image.width, image.height), source_image);
    CGContextRelease(context);
    CGImageRelease(source_image);

    return image;
}

bool savePng(const Image& image, const std::string& path)
{
    if (image.width <= 0 || image.height <= 0 || image.rgba.empty()) {
        return false;
    }

    CFURLRef url = CFURLCreateFromFileSystemRepresentation(
        kCFAllocatorDefault,
        reinterpret_cast<const UInt8*>(path.c_str()),
        path.size(),
        false);
    if (url == nullptr) {
        return false;
    }

    CGDataProviderRef provider = CGDataProviderCreateWithData(
        nullptr,
        image.rgba.data(),
        image.rgba.size(),
        nullptr);
    if (provider == nullptr) {
        CFRelease(url);
        return false;
    }

    CGColorSpaceRef color_space = CGColorSpaceCreateDeviceRGB();
    CGImageRef cg_image = CGImageCreate(
        image.width,
        image.height,
        8,
        32,
        static_cast<std::size_t>(image.width) * 4,
        color_space,
        kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big,
        provider,
        nullptr,
        false,
        kCGRenderingIntentDefault);
    CGColorSpaceRelease(color_space);
    CGDataProviderRelease(provider);

    if (cg_image == nullptr) {
        CFRelease(url);
        return false;
    }

    CGImageDestinationRef destination = CGImageDestinationCreateWithURL(url, CFSTR("public.png"), 1, nullptr);
    CFRelease(url);
    if (destination == nullptr) {
        CGImageRelease(cg_image);
        return false;
    }

    CGImageDestinationAddImage(destination, cg_image, nullptr);
    const bool success = CGImageDestinationFinalize(destination);
    CFRelease(destination);
    CGImageRelease(cg_image);
    return success;
}

std::vector<float> toGray(const Image& image)
{
    std::vector<float> gray(static_cast<std::size_t>(image.width) * image.height);
    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            const std::size_t offset = (static_cast<std::size_t>(y) * image.width + x) * 4;
            const float r = image.rgba[offset + 0];
            const float g = image.rgba[offset + 1];
            const float b = image.rgba[offset + 2];
            gray[static_cast<std::size_t>(y) * image.width + x] = 0.299f * r + 0.587f * g + 0.114f * b;
        }
    }

    return gray;
}

void setPixel(Image& image, int x, int y, std::uint8_t r, std::uint8_t g, std::uint8_t b)
{
    if (x < 0 || y < 0 || x >= image.width || y >= image.height) {
        return;
    }

    const std::size_t offset = (static_cast<std::size_t>(y) * image.width + x) * 4;
    image.rgba[offset + 0] = r;
    image.rgba[offset + 1] = g;
    image.rgba[offset + 2] = b;
    image.rgba[offset + 3] = 255;
}

void drawLine(Image& image, int x0, int y0, int x1, int y1)
{
    const int dx = std::abs(x1 - x0);
    const int sx = x0 < x1 ? 1 : -1;
    const int dy = -std::abs(y1 - y0);
    const int sy = y0 < y1 ? 1 : -1;
    int error = dx + dy;

    while (true) {
        setPixel(image, x0, y0, 255, 0, 0);
        if (x0 == x1 && y0 == y1) {
            break;
        }

        const int twice_error = 2 * error;
        if (twice_error >= dy) {
            error += dy;
            x0 += sx;
        }
        if (twice_error <= dx) {
            error += dx;
            y0 += sy;
        }
    }
}

void drawArrow(Image& image, int x0, int y0, int x1, int y1)
{
    drawLine(image, x0, y0, x1, y1);

    const float angle = std::atan2(static_cast<float>(y1 - y0), static_cast<float>(x1 - x0));
    const float head_length = 6.0f;
    const float head_angle = 0.55f;

    const int left_x = static_cast<int>(std::round(x1 - head_length * std::cos(angle - head_angle)));
    const int left_y = static_cast<int>(std::round(y1 - head_length * std::sin(angle - head_angle)));
    const int right_x = static_cast<int>(std::round(x1 - head_length * std::cos(angle + head_angle)));
    const int right_y = static_cast<int>(std::round(y1 - head_length * std::sin(angle + head_angle)));

    drawLine(image, x1, y1, left_x, left_y);
    drawLine(image, x1, y1, right_x, right_y);
}

float minEigenvalue2x2(float a, float b, float c)
{
    const float trace = a + c;
    const float determinant_part = std::sqrt((a - c) * (a - c) + 4.0f * b * b);
    return 0.5f * (trace - determinant_part);
}

}  // namespace

int main()
{
    Image current = loadPng("data/img/scene00140.png");
    Image next = loadPng("data/img/scene00141.png");

    if (current.rgba.empty() || next.rgba.empty()) {
        std::cerr << "Failed to load input images." << std::endl;
        return 1;
    }

    if (current.width != next.width || current.height != next.height) {
        std::cerr << "Input images must have the same size." << std::endl;
        return 1;
    }

    const int width = current.width;
    const int height = current.height;
    const std::vector<float> current_gray = toGray(current);
    const std::vector<float> next_gray = toGray(next);

    std::vector<float> df_dx(static_cast<std::size_t>(width) * height, 0.0f);
    std::vector<float> df_dy(static_cast<std::size_t>(width) * height, 0.0f);
    std::vector<float> df_dt(static_cast<std::size_t>(width) * height, 0.0f);

    for (int y = 0; y < height - 1; ++y) {
        for (int x = 0; x < width - 1; ++x) {
            const std::size_t index = static_cast<std::size_t>(y) * width + x;
            df_dx[index] = current_gray[index + 1] - current_gray[index];
            df_dy[index] = current_gray[index + width] - current_gray[index];
            df_dt[index] = next_gray[index] - current_gray[index];
        }
    }

    const int half_window = kWindowSize / 2;
    for (int y = half_window; y < height - half_window; y += kStepSize) {
        for (int x = half_window; x < width - half_window; x += kStepSize) {
            float sum_ix2 = 0.0f;
            float sum_ixiy = 0.0f;
            float sum_iy2 = 0.0f;
            float sum_ixb = 0.0f;
            float sum_iyb = 0.0f;

            for (int window_y = y - half_window; window_y <= y + half_window; ++window_y) {
                for (int window_x = x - half_window; window_x <= x + half_window; ++window_x) {
                    const std::size_t index = static_cast<std::size_t>(window_y) * width + window_x;
                    const float ix = df_dx[index];
                    const float iy = df_dy[index];
                    const float b = -df_dt[index];

                    sum_ix2 += ix * ix;
                    sum_ixiy += ix * iy;
                    sum_iy2 += iy * iy;
                    sum_ixb += ix * b;
                    sum_iyb += iy * b;
                }
            }

            if (minEigenvalue2x2(sum_ix2, sum_ixiy, sum_iy2) < kMinEigenvalue) {
                continue;
            }

            const float determinant = sum_ix2 * sum_iy2 - sum_ixiy * sum_ixiy;
            if (std::fabs(determinant) < 1.0e-6f) {
                continue;
            }

            const float u = (sum_ixb * sum_iy2 - sum_ixiy * sum_iyb) / determinant;
            const float v = (sum_ix2 * sum_iyb - sum_ixiy * sum_ixb) / determinant;

            const int end_x = static_cast<int>(std::round(x + u));
            const int end_y = static_cast<int>(std::round(y + v));
            if (end_x != x || end_y != y) {
                drawArrow(current, x, y, end_x, end_y);
            }
        }
    }

    const std::string output_path = "output_mac.png";
    if (!savePng(current, output_path)) {
        std::cerr << "Failed to save " << output_path << "." << std::endl;
        return 1;
    }

    std::cout << "Saved " << output_path << std::endl;
    return 0;
}
