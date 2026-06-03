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
   size and Flip Count budget. CIFAR-10 PyTorch artifacts use gradient-ranked
   Progressive Bit Search in the style of the upstream BFA implementation,
   evaluating the best Conv/Linear candidate plan per layer instead of
   exhaustively forwarding every eligible bit.

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
   larger CUDA server reproduction run.

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

.. code-block:: bash

   uv run netflip prepare-cifar10-resnet20 --download

The command uses the benchmark's ``build_cifar_resnet20`` constructor, trains
the FP32 ResNet-20 model, writes an intermediate FP32 checkpoint, quantizes
perturbable ``*.weight`` tensors with BFA-Compatible Int8 Quantization, writes
Per-Tensor Quantization Scale metadata, and validates that the emitted int8
artifact can be loaded by ``load_cifar_resnet20_quantized_artifact``.
When ``--download`` is used, NetFlip labels the dataset preparation phase and
torchvision may show its own 0-100% download progress. Training emits one line
per epoch with learning rate, training loss, and top-1 accuracy, and keeps a
batch-level progress bar visible while each epoch is running.

The default training configuration mirrors the CIFAR-10 ResNet-20 setup from
the upstream BFA training script: 160 epochs, SGD, batch size 128, learning rate
0.1, momentum 0.9, weight decay 0.0003, and learning rate decays at epochs 80
and 120 with gamma 0.1 at each milestone.

Default outputs match the example Experiment Specs:

``checkpoints/cifar10/resnet20-fp32.pt``
   Intermediate FP32 model state produced before quantization.

``checkpoints/cifar10/resnet20-int8.pt``
   BFA-compatible checkpoint whose perturbable weight tensors use signed int8
   two's-complement values.

``checkpoints/cifar10/resnet20-int8-scales.json``
   Per-tensor scale metadata for each perturbable weight tensor.

The local artifact directories ``data/``, ``checkpoints/``, ``runs/``, and
``save/`` are ignored by git so downloaded datasets, trained checkpoints, and
Run outputs are not uploaded accidentally.

CIFAR-10 is downloaded only when ``--download`` is provided. Without that flag,
``--dataset-root`` must point at an existing CIFAR-10 root:

.. code-block:: bash

   uv run netflip prepare-cifar10-resnet20 \
      --dataset-root data/cifar10 \
      --output-dir checkpoints/cifar10

For a quick local smoke run, reduce the training and evaluation sample counts
and use zero epochs to validate artifact writing without spending time on a full
training pass:

.. code-block:: bash

   uv run netflip prepare-cifar10-resnet20 \
      --download \
      --epochs 0 \
      --train-sample-limit 128 \
      --evaluation-sample-limit 128

``--device auto`` selects CUDA when available, then MPS, then CPU. Explicit
``--device cuda`` and ``--device mps`` fail when the requested backend is not
available. Increase ``--epochs`` for a useful Clean Baseline before launching a
larger BFA/PBS Run.

After the dataset and prepared artifacts exist, run the Attack Scenario:

.. code-block:: bash

   uv run netflip run examples/cifar10_resnet20/bfa_pbs.yaml

For this benchmark path, each BFA/PBS step runs a selection-batch backward pass,
evaluates the strongest Conv/Linear candidate plan per layer, and commits the
plan whose forward pass maximizes cross entropy. If a one-bit plan does not
increase the objective, the PyTorch scorer progressively tries larger per-layer
plans up to the remaining Flip Count budget.

Distributed training, external model zoo integration, and Per-Channel
Quantization Scale support remain outside the MVP Benchmark scope.

Evaluation Helpers
------------------

The benchmark module exposes ``build_cifar10_dataloaders`` for constructing
Selection Dataset and Evaluation Dataset loaders with standard CIFAR-10 tensor
normalization. It also exposes ``compute_top1_accuracy`` and
``compute_cross_entropy_loss`` for classification evaluation on a model and
DataLoader pair.
