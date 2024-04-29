#!/bin/bash
set -e

envoy_name=$1

fx envoy start -n $envoy_name --disable-tls --envoy-config-path envoy_config3.yaml -dh worker-1 -dp 50051