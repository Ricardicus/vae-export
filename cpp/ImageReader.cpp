#include "ImageReader.hpp"
#include <fstream>
#include <iostream>
#include <png.h>
#include <stdio.h>
ImageReader::ImageReader() : width(0), height(0), channels(0) {}
ImageReader::~ImageReader() {}
bool ImageReader::readPNG(const std::string &filename) {
  std::cout << filename << std::endl;
  FILE *file = fopen(filename.c_str(), "rb");
  if (!file) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return false;
  }
  png_structp png =
      png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  if (!png) {
    std::cerr << "Failed to create PNG read struct" << std::endl;
    fclose(file);
    return false;
  }
  png_infop info = png_create_info_struct(png);
  if (!info) {
    std::cerr << "Failed to create PNG info struct" << std::endl;
    png_destroy_read_struct(&png, nullptr, nullptr);
    fclose(file);
    return false;
  }
  if (setjmp(png_jmpbuf(png))) {
    std::cerr << "Failed to setjmp" << std::endl;
    png_destroy_read_struct(&png, &info, nullptr);
    fclose(file);
    return false;
  }
  png_init_io(png, file);
  png_read_info(png, info);
  width = png_get_image_width(png, info);
  height = png_get_image_height(png, info);
  channels = png_get_channels(png, info);
  png_bytep row_pointers[height];
  for (int y = 0; y < height; ++y) {
    row_pointers[y] = new png_byte[png_get_rowbytes(png, info)];
  }
  png_read_image(png, row_pointers);
  data.clear();
  data.reserve(width * height * channels);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width * channels; ++x) {
      data.push_back(row_pointers[y][x]);
    }
  }
  for (int y = 0; y < height; ++y) {
    delete[] row_pointers[y];
  }
  png_destroy_read_struct(&png, &info, nullptr);
  fclose(file);
  return true;
}
bool ImageReader::writePNG(const std::string &filename) {
  FILE *file = fopen(filename.c_str(), "wb");
  if (!file) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return false;
  }
  png_structp png =
      png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  if (!png) {
    std::cerr << "Failed to create PNG write struct" << std::endl;
    fclose(file);
    return false;
  }
  png_infop info = png_create_info_struct(png);
  if (!info) {
    std::cerr << "Failed to create PNG info struct" << std::endl;
    png_destroy_write_struct(&png, nullptr);
    fclose(file);
    return false;
  }
  if (setjmp(png_jmpbuf(png))) {
    std::cerr << "Failed to setjmp" << std::endl;
    png_destroy_write_struct(&png, &info);
    fclose(file);
    return false;
  }
  png_init_io(png, file);
  png_set_IHDR(png, info, width, height, 8, PNG_COLOR_TYPE_RGBA,
               PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
               PNG_FILTER_TYPE_DEFAULT);
  png_write_info(png, info);
  png_bytep row_pointers[height];
  for (int y = 0; y < height; ++y) {
    row_pointers[y] = new png_byte[width * channels];
    for (int x = 0; x < width * channels; ++x) {
      row_pointers[y][x] = data[y * width * channels + x];
    }
  }
  png_write_image(png, row_pointers);
  png_write_end(png, nullptr);
  for (int y = 0; y < height; ++y) {
    delete[] row_pointers[y];
  }
  png_destroy_write_struct(&png, &info);
  fclose(file);
  return true;
}
int ImageReader::getWidth() const { return width; }
int ImageReader::getHeight() const { return height; }
int ImageReader::getChannels() const { return channels; }

void ImageReader::setWidth(int newWidth) { width = newWidth; }
void ImageReader::setHeight(int newHeight) { height = newHeight; }
void ImageReader::setChannels(int newChannels) { channels = newChannels; }
std::vector<unsigned char> &ImageReader::getData() { return data; }
void ImageReader::print() { printf("%dx%d c:%d\n", width, height, channels); }
