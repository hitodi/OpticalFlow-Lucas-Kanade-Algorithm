# Optical Flow - Lucas-Kanade Algorithm

This repository contains a small C++/OpenCV implementation of sparse optical flow using the Lucas-Kanade method.

The program compares two consecutive sample images:

- `test/data/img/scene00140.png`
- `test/data/img/scene00141.png`

It estimates local motion vectors and draws them on the first frame.

## Requirements For The macOS Build

- C++17 compiler
- Xcode Command Line Tools

The macOS build does not require OpenCV, CMake, or Homebrew.

## Build On macOS Without OpenCV Or CMake

The repository includes a macOS-native implementation that uses Apple's built-in ImageIO/CoreGraphics frameworks for PNG input/output. It does not require Homebrew, OpenCV, or CMake.

Build:

```sh
make mac
```

Run:

```sh
make run-mac
```

The result is saved as:

```text
output_mac.png
```

## Build With CMake

This path is optional and requires OpenCV 4.x plus CMake 3.16 or newer.

From the repository root:

```sh
cmake -S . -B build
cmake --build build
```

Run:

```sh
./build/lucas_kanade_optical_flow
```

On Windows, the executable may be under a configuration directory:

```powershell
.\build\Debug\lucas_kanade_optical_flow.exe
```

## Build With Visual Studio

Open `Opencv_test.sln` in Visual Studio and build the `x64` configuration.

The Visual Studio project expects OpenCV at:

```text
C:\opencv\build
```

If OpenCV is installed somewhere else, define the `OpenCVRoot` environment variable before opening Visual Studio:

```powershell
setx OpenCVRoot "C:\path\to\opencv\build"
```

The project currently links against OpenCV 4.7.0 world libraries:

- Debug: `opencv_world470d.lib`
- Release: `opencv_world470.lib`

If your OpenCV version is different, update the library names in `test/test.vcxproj`.

## Notes

The implementation follows the Lucas-Kanade least-squares form:

```text
[Ix Iy] [u v]^T = -It
```

Low-texture windows are skipped using a minimum eigenvalue threshold to avoid unstable motion vectors.
