#include <chrono>

#include <iostream>
#include <fstream>
#include <sstream>

#include <memory>
#include <random>
#include <string>
#include <thread>

#include <grpc/grpc.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>

#include "serving.pb.h"
#include "serving.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::ClientReader;
using grpc::ClientReaderWriter;
using grpc::ClientWriter;
using grpc::Status;

using serving::ObjectDetection;
using serving::DetectionRequest;
using serving::DetectionResponse;
using serving::Object;
using serving::Point;

class ObjectDetectionClient {

  public:
    ObjectDetectionClient(std::shared_ptr<Channel> channel)
      : stub_(serving::ObjectDetection::NewStub(channel)) { }

  bool DetectImageFile(const std::string& image_file) {
    std::ifstream file(image_file, std::ios::binary);
    std::ostringstream ostrm;
    ostrm << file.rdbuf();
    std::string bytes = ostrm.str();
    return DetectImage(bytes);
  }

  bool DetectImage(const std::string& image) {
    DetectionRequest request;
    request.set_allocated_image(new std::string(image));
    request.clear_region();

    DetectionResponse response;

    ClientContext context;
    Status status = stub_->Detect(&context, request, &response);
    if (status.ok()) {
      std::cerr << "Detect rpc succeeded." << std::endl;
    } else {
      std::cerr << "Detect rpc failed." << std::endl;
      std::cout << "error_code:" << status.error_code() << std::endl;
      std::cout << "error_message:" << status.error_message() << std::endl;
      std::cout << "error_details:" << status.error_details() << std::endl;
      return false;
    }

    std::cerr << "type: " << response.type() << std::endl;
    for (int i = 0; i < response.objects_size() ; i++) {
      const Object obj = response.objects(i);
      std::cerr 
        << "Found face box at: (" << obj.box().x() << "," << obj.box().y() << ")"
        << " size: " << obj.box().width() << "x" << obj.box().height();

      std::cerr << std::endl;
      std::cerr.flush();
    }

    return true;
  }



  bool DetectImageStream(const std::string& image) {
    DetectionRequest request;
    request.set_allocated_image(new std::string(image));
    request.clear_region();

    ClientContext context;
    std::unique_ptr<ClientReader<Object> > reader(
        stub_->DetectStream(&context, request));

    Object obj;
    while (reader->Read(&obj)) {
      std::cerr 
        << "Found rect at: (" << obj.box().x() << "," << obj.box().y() << ")"
        << " size: " << obj.box().width() << "x" << obj.box().height()
        << std::endl;
      std::cerr.flush();
    }

    Status status = reader->Finish();
    if (status.ok()) {
      std::cerr << "Detect rpc succeeded." << std::endl;
    } else {
      std::cerr << "Detect rpc failed:" << status.error_message() << std::endl;
      return false;
    }
    std::cout << status.error_details() << std::endl;
    std::cout << status.error_message() << std::endl;

    return true;
  }

  private:
  std::unique_ptr<ObjectDetection::Stub> stub_;
};

int main(int argc, char** argv) {
    auto channel = grpc::CreateChannel("localhost:50051", grpc::InsecureChannelCredentials());
    auto client = ObjectDetectionClient(channel);

    auto image_file = "dlib/examples/faces/2007_007763.jpg";

    std::ifstream file(image_file, std::ios::binary);
    std::ostringstream ostrm;
    ostrm << file.rdbuf();
    std::string bytes = ostrm.str();
    client.DetectImage(bytes);
    // client.DetectImageStream(bytes);
  return 0;
}
