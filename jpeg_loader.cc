#include <iostream>
#include <memory>
#include <string>
#include <cstdio>
#include <cstdlib>

#include <dlib/config.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <jpeglib.h>
#include "jpeg_loader.h"

// decode jpeg image from memory
// https://gist.github.com/PhirePhly/3080633
void JpegLoader::ReadImage(const std::string& content) {
  const char * chrs = content.c_str();

  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);

  auto uchrs = reinterpret_cast<unsigned char*>(const_cast<char*>(chrs));
  auto jpg_size = content.length();
  unsigned char *jpg_buffer = uchrs;
  jpeg_mem_src(&cinfo, jpg_buffer, jpg_size);

  int rc = jpeg_read_header(&cinfo, TRUE);
  if (rc != 1) {
    std::cerr << "jpeg: decode failed" << std::endl;
    throw std::runtime_error("jpeg: decode failed");
    // return grpc::Status(grpc::INVALID_ARGUMENT, "jpeg decode failed");
  }
  jpeg_start_decompress(&cinfo);

  width_ = cinfo.output_width;
  height_ = cinfo.output_height;
  output_components_ = cinfo.output_components;

  if (output_components_ != 1 && 
      output_components_ != 3 &&
      output_components_ != 4)
  {
    std::cerr << "jpeg: unsupported number of colors: " << output_components_ << std::endl;
    jpeg_destroy_decompress(&cinfo);
    throw std::runtime_error("jpeg: unsupported number of colors");
    // return grpc::Status(grpc::INVALID_ARGUMENT, "jpeg: unsupported number of colors");
    // std::ostringstream sout;
    // sout << "jpeg_loader: Unsupported number of colors (" << output_components_ << ") in file " << filename;
    // throw image_load_error(sout.str());
  }

  std::vector<unsigned char*> rows;
  rows.resize(height_);

  // size the image buffer
  data.resize(height_ * width_ * output_components_);

  // setup pointers to each row
  for (unsigned long i = 0; i < rows.size(); ++i)
      rows[i] = &data[i * width_ * output_components_];

  // read the data into the buffer
  while (cinfo.output_scanline < cinfo.output_height)
  {
      jpeg_read_scanlines(&cinfo, &rows[cinfo.output_scanline], 100);
  }

  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
}
