"""Benchmark model constructors."""

from netflip.benchmarks.cifar_resnet20 import (
    CIFAR10_CLASSES,
    CIFAR_RESNET20_BENCHMARK_ID,
    ResNet20Config,
    build_cifar_resnet20,
)

__all__ = [
    "CIFAR10_CLASSES",
    "CIFAR_RESNET20_BENCHMARK_ID",
    "ResNet20Config",
    "build_cifar_resnet20",
]
