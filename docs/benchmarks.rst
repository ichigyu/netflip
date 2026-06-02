Benchmarks
==========

CIFAR-10 ResNet-20
------------------

The MVP Benchmark target is CIFAR-10 classification with a CIFAR-compatible
ResNet-20 model and BFA-compatible int8 quantization metadata. The benchmark
constructor lives at ``netflip.benchmarks.build_cifar_resnet20`` and imports
PyTorch lazily, so the core package can still validate Experiment Specs without
installing benchmark runtime dependencies.

Install the optional benchmark runtime dependencies when running CIFAR-10 data
loading or model evaluation locally:

.. code-block:: bash

   uv sync --extra benchmark

Example YAML Experiment Specs are provided in ``examples/cifar10_resnet20``:

``random_soft_error.yaml``
   A Soft Error Scenario using a Uniform Eligible-Bit Fault Model and a
   One-Bit Step Schedule.

``bfa_pbs.yaml``
   An Attack Scenario skeleton for BFA/PBS with a configurable selection batch
   size and Flip Count budget.

Dataset And Checkpoint Paths
----------------------------

Both example specs expose the paths that vary across local machines:

``dataset.root``
   Root directory containing CIFAR-10 data. NetFlip does not download CIFAR-10
   automatically unless the loader is called with ``download=True``.

``dataset.selection_split`` / ``dataset.evaluation_split``
   Explicit Selection Dataset and Evaluation Dataset split names. The CIFAR-10
   benchmark loader supports ``train`` and ``test``.

``dataset.selection_sample_limit`` / ``dataset.evaluation_sample_limit``
   Optional small-validation sample limits for MBP-friendly checks before a
   larger CUDA-server reproduction run.

``model.checkpoint.path``
   Path to the prepared ResNet-20 checkpoint.

``model.quantization.scale_path``
   Path to per-tensor scale metadata used with the Signed Integer Two's
   Complement Codec.

``device``
   PyTorch runtime device request. ``auto`` selects CUDA when available, then
   MPS when available, and otherwise records CPU as the Run Manifest device.
   Explicit ``cuda`` and ``mps`` requests fail when that backend is unavailable.

Checkpoint Preparation
----------------------

The benchmark expects a CIFAR-10 ResNet-20 checkpoint whose persistent model
state uses BFA-compatible int8 quantization:

1. Train or obtain a CIFAR-10 ResNet-20 model outside NetFlip.
2. Quantize weights to signed int8 two's-complement values with per-tensor
   scale metadata.
3. Save the model state dictionary at ``model.checkpoint.path``.
4. Save quantization scale metadata at ``model.quantization.scale_path``.
5. Point ``dataset.root`` at the CIFAR-10 data used for selection and
   evaluation splits.

The production training pipeline and external model zoo integration are outside
the MVP Benchmark scope.

Evaluation Helpers
------------------

The benchmark module exposes ``build_cifar10_dataloaders`` for constructing
Selection Dataset and Evaluation Dataset loaders with standard CIFAR-10 tensor
normalization. It also exposes ``compute_top1_accuracy`` and
``compute_cross_entropy_loss`` for classification evaluation on a model and
DataLoader pair.
