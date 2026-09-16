#!/usr/bin/env bash
# Regenerates the gRPC Python stubs from protos/ogm_inference.proto.
# Requires grpcio-tools (dev-only; not needed to run the server afterward):
#   pip install grpcio-tools
set -euo pipefail
cd "$(dirname "$0")"

python -m grpc_tools.protoc \
  -I../../protos \
  --python_out=generated \
  --grpc_python_out=generated \
  ../../protos/ogm_inference.proto

# grpc_tools generates an absolute-style import ("import ogm_inference_pb2 as ...")
# in the _grpc.py file; rewrite it to a package-relative import so it works when
# imported as `serving.generated.ogm_inference_pb2_grpc`.
sed -i 's/^import ogm_inference_pb2 as/from . import ogm_inference_pb2 as/' generated/ogm_inference_pb2_grpc.py

echo "Generated src/serving/generated/ogm_inference_pb2.py and ogm_inference_pb2_grpc.py"
