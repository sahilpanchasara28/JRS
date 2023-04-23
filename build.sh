#!/bin/bash
PROTOC_VERSION=3.19.0
PROTOC_URL=https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOC_VERSION}/protoc-${PROTOC_VERSION}-linux-x86_64.zip

echo "Installing protobuf compiler version $PROTOC_VERSION..."
curl -OL $PROTOC_URL
unzip protoc-${PROTOC_VERSION}-linux-x86_64.zip -d /usr/local


pip install -r requirements.txt