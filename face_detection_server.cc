#include <iostream>
#include <memory>
#include <string>

#include <cstdio>
#include <cstdlib>

#include <grpc/grpc.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/security/server_credentials.h>

#include "inference.pb.h"
#include "inference.grpc.pb.h"

#include "face_detection_service.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;

using std::chrono::system_clock;

void RunServer() {
  std::string server_address("0.0.0.0:50051"); 

  DlibFaceDetectionService service;

  ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);

  std::unique_ptr<Server> server(builder.BuildAndStart());
  std::cout << "Server listening on " << server_address << std::endl;
  server->Wait();
}

int main(int argc, char** argv) {
  // const char * env_model_file = std::getenv("MODEL_FILE");
  RunServer();
  return 0;
}
