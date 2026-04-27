/***************************************************************************************************
 * Copyright (c) 2011-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are not permit-
 * ted.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "FmhaOptions.h"

namespace fmha {

///////////////////////////////////////////////////////////////////////////////////////////////////

// Calculate the numCtasX, numCtasY and numCtasZ.
std::tuple<int32_t, int32_t, int32_t> computeNumCtas(FmhaOptions& options,
                                                     int32_t multiProcessorCount,
                                                     bool enablesLogging = true);

///////////////////////////////////////////////////////////////////////////////////////////////////

// AutoTuner to select the kernel based on the heuristics.
// We might also use performance tests to select the best one from the candidates in the future.
// Ensure that multiProcessorCount is obtained from the current device's properties;
// if it does not match the current device, FmhaAutoTuner may fail to choose the optimal Fmha
// configuration.
class FmhaAutoTuner {

public:
  // The constructor.
  FmhaAutoTuner(FmhaOptions const& options,
                FmhaOptionsFromArgs const& optionsFromArgs,
                int32_t multiProcessorCount)
    : mMultiProcessorCount(multiProcessorCount)
    , mOptions(options)
    , mOptionsFromArgs(optionsFromArgs) {}

public:
  // Get the mmaOpsPerClk.
  static int32_t getMmaOpsPerClk(FmhaOptions const& options,
                                 KernelTraits const& kernelTraits,
                                 bool isBmm1 = true);

  // Select the GQA generation kernel.
  void selectGqaGenerationKernel();

  // Select the kernel for tree-based speculative decoding (Eagle3 dynamic tree, MTP tree).
  // Uses numTokensHeadsQ = numHeadsQPerKv * specDecodingTargetMaxGenLen as a config-time
  // deterministic heuristic to choose tileSizeQ + kernelType.
  void selectSpecDecTreeKernel();

  // Select the kernel and update the options.
  std::tuple<FmhaOptions, FmhaOptionsFromArgs, int32_t> selectKernel();

  // Select the MLA generation kernel.
  void selectMlaGenerationKernel();

private:
  // Enables the cgaReduction if all clusters can be launched in one wave.
  void enableCgaReduction(int32_t numCtasX, int32_t numCtasY, int32_t numCtasZ);

  // Get the cluster size.
  int32_t getClusterSize();

  // Get the swapsMmaAbTileSizeQ.
  int32_t getSwapsMmaAbTileSizeQ() const;

  // Get the maximum number of active clusters for a given cluster size which considers the
  // floorsweeping configurations.
  int32_t getMaxNumActiveClusters(int32_t clusterSize);

  // Selects the tileSizeQ for GQA generation kernels.
  void selectTileSizeQForGqaGeneration();

  // Set ctaDim.
  void setCtaDim();

  // Set mHeadDimPerStageKv.
  void setHeadDimPerStageKv();

  // Sets the kernel type and tileSizeQ for GQA generation kernels.
  void setGqaKernelTypeAndTileSizeQ();

  // Set headDimPerCtaV for context and GQA generation kernels. MLA sets separately.
  void setHeadDimPerCtaV(FmhaOptions& options);

  // Set the numInstsQ and numInstsKv.
  void setNumInstsQAndKv(FmhaOptions& options, bool forceSet = false, bool updateSetFlags = true);

  // Set mNumKPartitionsMmaPv and mNumKPartitionsTileP.
  void setNumKPartitionsMmaPvAndTileP();

  // Set MMA order, interleavesMufuAndSums, and usesOrderedSequence.
  void setMmaOrder();

  // Select the sparse MLA generation kernel.
  void selectSparseMlaGenerationKernel();

  // Set softmax configs.
  void setSoftmaxConfigs();

private:
  // The ctaDim.
  int mCtaDim{512};
  // The multiProcessorCount.
  int mMultiProcessorCount;
  // The FmhaOptions.
  FmhaOptions mOptions;
  // The FmhaOptionsFromArgs.
  FmhaOptionsFromArgs mOptionsFromArgs;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fmha