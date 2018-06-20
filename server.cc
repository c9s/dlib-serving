/*
 *
 * Copyright 2015 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */
#include <chrono>
#include <iostream>
#include <memory>
#include <string>

#include <cstdio>
#include <cstdlib>

#include <grpc/grpc.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>
#include <grpcpp/security/server_credentials.h>

#include <dlib/config.h>
#include <dlib/image_processing.h>

#include "helper.h"
#include "inference.pb.h"
#include "inference.grpc.pb.h"

#include "jpeg_loader.h"
#include "face_detection_service.h"
#include "shape_detection_service.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;

using inference::ObjectDetection;
using inference::ShapeDetection;
using inference::ShapeDetectionResponse;

using std::chrono::system_clock;

void RunServer(const std::string& db_path) {
  std::string server_address("0.0.0.0:50051");

  DlibShapeDetectionService service("shape_predictor_68_face_landmarks.dat");
  DlibFaceDetectionService service2;

  ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  builder.RegisterService(&service2);
  std::unique_ptr<Server> server(builder.BuildAndStart());
  std::cout << "Server listening on " << server_address << std::endl;
  server->Wait();
}

int main(int argc, char** argv) {
  // Expect only arg: --db_path=path/to/route_guide_db.json.
  // std::string db = inference::GetDbFileContent(argc, argv);
  std::string db = "test";
  RunServer(db);
  return 0;
}
