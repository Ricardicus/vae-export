#ifndef IMAGEREADER_HPP
#define IMAGEREADER_HPP
#include <string>
#include <vector>
class ImageReader {
public:
  ImageReader();
  ~ImageReader();
  bool readPNG(const std::string &filename);
  bool writePNG(const std::string &filename);
  int getWidth() const;
  int getHeight() const;
  int getChannels() const;

  void setWidth(int width);
  void setHeight(int height);
  void setChannels(int channels);
  void print();
  std::vector<unsigned char> &getData();

private:
  int width;
  int height;
  int channels;
  std::vector<unsigned char> data;
};
#endif // IMAGEREADER_HPP
