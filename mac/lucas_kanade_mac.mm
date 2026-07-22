#include <CoreFoundation/CoreFoundation.h>
#include <CoreGraphics/CoreGraphics.h>
#include <ImageIO/ImageIO.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int kWindowSize = 21;
constexpr int kStepSize = 30;
constexpr float kMinEigenvalue = 0.05f;
constexpr float kMinEigenRatio = 0.03f;
constexpr float kMinMotion = 0.15f;
constexpr float kMaxMotion = 40.0f;
constexpr float kDisplayScale = 2.0f;
constexpr int kPyramidLevels = 4;
constexpr int kIterationsPerLevel = 5;
constexpr float kConvergenceEpsilon = 0.01f;
constexpr float kForwardBackwardTolerance = 2.0f;
constexpr float kCornerMinEigenvalue = 0.04f;
constexpr float kCornerMinEigenRatio = 0.03f;

struct Image {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> rgba;
};

struct GrayImage {
    int width = 0;
    int height = 0;
    std::vector<float> pixels;
};

struct Flow {
    float u = 0.0f;
    float v = 0.0f;
    bool valid = false;
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

GrayImage toGray(const Image& image)
{
    GrayImage gray;
    gray.width = image.width;
    gray.height = image.height;
    gray.pixels.resize(static_cast<std::size_t>(image.width) * image.height);
    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            const std::size_t offset = (static_cast<std::size_t>(y) * image.width + x) * 4;
            const float r = image.rgba[offset + 0] / 255.0f;
            const float g = image.rgba[offset + 1] / 255.0f;
            const float b = image.rgba[offset + 2] / 255.0f;
            gray.pixels[static_cast<std::size_t>(y) * image.width + x] = 0.299f * r + 0.587f * g + 0.114f * b;
        }
    }

    return gray;
}

GrayImage downsampleHalf(const GrayImage& image)
{
    GrayImage result;
    result.width = image.width / 2;
    result.height = image.height / 2;
    result.pixels.resize(static_cast<std::size_t>(result.width) * result.height);

    for (int y = 0; y < result.height; ++y) {
        for (int x = 0; x < result.width; ++x) {
            const int source_x = x * 2;
            const int source_y = y * 2;
            const std::size_t i00 = static_cast<std::size_t>(source_y) * image.width + source_x;
            const std::size_t i01 = i00 + 1;
            const std::size_t i10 = i00 + image.width;
            const std::size_t i11 = i10 + 1;
            result.pixels[static_cast<std::size_t>(y) * result.width + x] =
                0.25f * (image.pixels[i00] + image.pixels[i01] + image.pixels[i10] + image.pixels[i11]);
        }
    }

    return result;
}

std::vector<GrayImage> buildPyramid(const GrayImage& base)
{
    std::vector<GrayImage> pyramid;
    pyramid.push_back(base);
    for (int level = 1; level < kPyramidLevels; ++level) {
        const GrayImage& previous = pyramid.back();
        if (previous.width < 64 || previous.height < 64) {
            break;
        }

        pyramid.push_back(downsampleHalf(previous));
    }

    return pyramid;
}

float sampleBilinear(const GrayImage& image, float x, float y)
{
    if (x < 0.0f || y < 0.0f || x >= image.width - 1 || y >= image.height - 1) {
        return 0.0f;
    }

    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const float tx = x - x0;
    const float ty = y - y0;
    const std::size_t index = static_cast<std::size_t>(y0) * image.width + x0;

    const float top = (1.0f - tx) * image.pixels[index] + tx * image.pixels[index + 1];
    const float bottom = (1.0f - tx) * image.pixels[index + image.width] + tx * image.pixels[index + image.width + 1];
    return (1.0f - ty) * top + ty * bottom;
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

float maxEigenvalue2x2(float a, float b, float c)
{
    const float trace = a + c;
    const float determinant_part = std::sqrt((a - c) * (a - c) + 4.0f * b * b);
    return 0.5f * (trace + determinant_part);
}

bool refineFlowAtLevel(
    const GrayImage& current,
    const GrayImage& next,
    float x,
    float y,
    float& u,
    float& v)
{
    const int half_window = kWindowSize / 2;

    for (int iteration = 0; iteration < kIterationsPerLevel; ++iteration) {
        float sum_ix2 = 0.0f;
        float sum_ixiy = 0.0f;
        float sum_iy2 = 0.0f;
        float sum_ixb = 0.0f;
        float sum_iyb = 0.0f;

        for (int offset_y = -half_window; offset_y <= half_window; ++offset_y) {
            for (int offset_x = -half_window; offset_x <= half_window; ++offset_x) {
                const float current_x = x + offset_x;
                const float current_y = y + offset_y;
                const float next_x = current_x + u;
                const float next_y = current_y + v;

                if (current_x < 1.0f || current_y < 1.0f ||
                    current_x >= current.width - 2 || current_y >= current.height - 2 ||
                    next_x < 1.0f || next_y < 1.0f ||
                    next_x >= next.width - 2 || next_y >= next.height - 2) {
                    continue;
                }

                const float current_dx =
                    sampleBilinear(current, current_x + 1.0f, current_y) -
                    sampleBilinear(current, current_x - 1.0f, current_y);
                const float next_dx =
                    sampleBilinear(next, next_x + 1.0f, next_y) -
                    sampleBilinear(next, next_x - 1.0f, next_y);
                const float current_dy =
                    sampleBilinear(current, current_x, current_y + 1.0f) -
                    sampleBilinear(current, current_x, current_y - 1.0f);
                const float next_dy =
                    sampleBilinear(next, next_x, next_y + 1.0f) -
                    sampleBilinear(next, next_x, next_y - 1.0f);

                const float ix = 0.25f * (current_dx + next_dx);
                const float iy = 0.25f * (current_dy + next_dy);
                const float it =
                    sampleBilinear(next, next_x, next_y) -
                    sampleBilinear(current, current_x, current_y);
                const float b = -it;

                sum_ix2 += ix * ix;
                sum_ixiy += ix * iy;
                sum_iy2 += iy * iy;
                sum_ixb += ix * b;
                sum_iyb += iy * b;
            }
        }

        const float min_eigenvalue = minEigenvalue2x2(sum_ix2, sum_ixiy, sum_iy2);
        const float max_eigenvalue = maxEigenvalue2x2(sum_ix2, sum_ixiy, sum_iy2);
        if (min_eigenvalue < kMinEigenvalue ||
            max_eigenvalue <= 0.0f ||
            min_eigenvalue / max_eigenvalue < kMinEigenRatio) {
            return false;
        }

        const float determinant = sum_ix2 * sum_iy2 - sum_ixiy * sum_ixiy;
        if (std::fabs(determinant) < 1.0e-6f) {
            return false;
        }

        const float delta_u = (sum_ixb * sum_iy2 - sum_ixiy * sum_iyb) / determinant;
        const float delta_v = (sum_ix2 * sum_iyb - sum_ixiy * sum_ixb) / determinant;
        u += delta_u;
        v += delta_v;

        if (delta_u * delta_u + delta_v * delta_v < kConvergenceEpsilon * kConvergenceEpsilon) {
            break;
        }
    }

    return true;
}

Flow estimatePyramidFlow(
    const std::vector<GrayImage>& current_pyramid,
    const std::vector<GrayImage>& next_pyramid,
    float x,
    float y)
{
    float u = 0.0f;
    float v = 0.0f;
    bool valid = false;

    for (int level = static_cast<int>(current_pyramid.size()) - 1; level >= 0; --level) {
        if (level != static_cast<int>(current_pyramid.size()) - 1) {
            u *= 2.0f;
            v *= 2.0f;
        }

        const float scale = 1.0f / static_cast<float>(1 << level);
        const float level_x = x * scale;
        const float level_y = y * scale;

        valid = refineFlowAtLevel(current_pyramid[level], next_pyramid[level], level_x, level_y, u, v);
        if (!valid) {
            return {};
        }
    }

    const float motion_squared = u * u + v * v;
    if (motion_squared < kMinMotion * kMinMotion ||
        motion_squared > kMaxMotion * kMaxMotion) {
        return {};
    }

    return {u, v, true};
}

bool isTrackableCorner(const GrayImage& image, int x, int y)
{
    const int half_window = kWindowSize / 2;
    float sum_ix2 = 0.0f;
    float sum_ixiy = 0.0f;
    float sum_iy2 = 0.0f;

    if (x < half_window + 1 || y < half_window + 1 ||
        x >= image.width - half_window - 1 ||
        y >= image.height - half_window - 1) {
        return false;
    }

    for (int offset_y = -half_window; offset_y <= half_window; ++offset_y) {
        for (int offset_x = -half_window; offset_x <= half_window; ++offset_x) {
            const int px = x + offset_x;
            const int py = y + offset_y;
            const std::size_t index = static_cast<std::size_t>(py) * image.width + px;
            const float ix = 0.5f * (image.pixels[index + 1] - image.pixels[index - 1]);
            const float iy = 0.5f * (image.pixels[index + image.width] - image.pixels[index - image.width]);

            sum_ix2 += ix * ix;
            sum_ixiy += ix * iy;
            sum_iy2 += iy * iy;
        }
    }

    const float min_eigenvalue = minEigenvalue2x2(sum_ix2, sum_ixiy, sum_iy2);
    const float max_eigenvalue = maxEigenvalue2x2(sum_ix2, sum_ixiy, sum_iy2);
    return min_eigenvalue >= kCornerMinEigenvalue &&
        max_eigenvalue > 0.0f &&
        min_eigenvalue / max_eigenvalue >= kCornerMinEigenRatio;
}

bool passesForwardBackwardCheck(
    const std::vector<GrayImage>& current_pyramid,
    const std::vector<GrayImage>& next_pyramid,
    int x,
    int y,
    const Flow& forward_flow)
{
    const float next_x = x + forward_flow.u;
    const float next_y = y + forward_flow.v;
    const GrayImage& base_next = next_pyramid.front();
    if (next_x < 0.0f || next_y < 0.0f ||
        next_x >= base_next.width || next_y >= base_next.height) {
        return false;
    }

    const Flow backward_flow = estimatePyramidFlow(next_pyramid, current_pyramid, next_x, next_y);
    if (!backward_flow.valid) {
        return false;
    }

    const float error_u = forward_flow.u + backward_flow.u;
    const float error_v = forward_flow.v + backward_flow.v;
    return error_u * error_u + error_v * error_v <=
        kForwardBackwardTolerance * kForwardBackwardTolerance;
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
    const GrayImage current_gray = toGray(current);
    const std::vector<GrayImage> current_pyramid = buildPyramid(current_gray);
    const std::vector<GrayImage> next_pyramid = buildPyramid(toGray(next));

    const int half_window = kWindowSize / 2;
    for (int y = half_window + 1; y < height - half_window - 1; y += kStepSize) {
        for (int x = half_window + 1; x < width - half_window - 1; x += kStepSize) {
            if (!isTrackableCorner(current_gray, x, y)) {
                continue;
            }

            const Flow flow = estimatePyramidFlow(current_pyramid, next_pyramid, x, y);
            if (!flow.valid) {
                continue;
            }

            if (!passesForwardBackwardCheck(current_pyramid, next_pyramid, x, y, flow)) {
                continue;
            }

            const int end_x = static_cast<int>(std::round(x + kDisplayScale * flow.u));
            const int end_y = static_cast<int>(std::round(y + kDisplayScale * flow.v));
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
