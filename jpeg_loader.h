#ifndef JPEG_LOADER_H
#define JPEG_LOADER_H

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <cstdio>
#include <cstdlib>

class JpegLoader {

public:

  explicit JpegLoader() : width_(0), height_(0), output_components_(0) { }

  JpegLoader(const std::vector<unsigned char>& data, int height, int width, int output_components)
      : data(data), height_(height), width_(width), output_components_(output_components) { }
  
  bool is_gray() const
  {
      return (output_components_ == 1);
  }

  bool is_rgb() const
  {
      return (output_components_ == 3);
  }

  bool is_rgba() const
  {
      return (output_components_ == 4);
  }

  // decode jpeg image from memory
  // https://gist.github.com/PhirePhly/3080633
  void ReadImage(const std::string& content);

  template<typename T> void GetImage(T &t_) const
{

  dlib::image_view<T> t(t_);
  t.set_size(height_, width_);
  for (unsigned n = 0; n < height_; n++)
  {
      const unsigned char* v = get_row(n);
      for (unsigned m = 0; m < width_;m++)
      {
          if (is_gray())
          {
              unsigned char p = v[m];
              dlib::assign_pixel( t[n][m], p );
          }
          else if (is_rgba()) {
              dlib::rgb_alpha_pixel p;
              p.red = v[m*4];
              p.green = v[m*4+1];
              p.blue = v[m*4+2];
              p.alpha = v[m*4+3];
              dlib::assign_pixel( t[n][m], p );
          }
          else // if ( is_rgb() )
          {
              dlib::rgb_pixel p;
              p.red = v[m*3];
              p.green = v[m*3+1];
              p.blue = v[m*3+2];
              dlib::assign_pixel( t[n][m], p );
          }
      }
  }
}

private:


  int width_;

  int height_;

  int output_components_;

  std::vector<unsigned char> data;

  const unsigned char* get_row(unsigned long i) const
  {
      return &data[i*width_*output_components_];
  }
};

#endif
