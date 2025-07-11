import React, { useState, useEffect } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "./components/ui/card";
import { Button } from "./components/ui/button";
import { Badge } from "./components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./components/ui/tabs";
import { Slider } from "./components/ui/slider";
import { Alert, AlertDescription } from "./components/ui/alert";
import { Separator } from "./components/ui/separator";

interface CudaIndexingConfig {
  gridX: number;
  gridY: number;
  gridZ: number;
  blockX: number;
  blockY: number;
  blockZ: number;
}

interface ThreadInfo {
  blockIdx: { x: number; y: number; z: number };
  threadIdx: { x: number; y: number; z: number };
  blockId: number;
  threadOffset: number;
  globalId: number;
}

const CudaIndexingTutorial: React.FC = () => {
  const [config, setConfig] = useState<CudaIndexingConfig>(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("cuda-indexing-config");
      if (saved) {
        try {
          return JSON.parse(saved);
        } catch {
          // Fall back to default if parsing fails
        }
      }
    }
    return {
      gridX: 2,
      gridY: 3,
      gridZ: 4,
      blockX: 4,
      blockY: 4,
      blockZ: 4,
    };
  });

  const [selectedBlock, setSelectedBlock] = useState<{
    x: number;
    y: number;
    z: number;
  } | null>(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("cuda-indexing-selected-block");
      if (saved) {
        try {
          return JSON.parse(saved);
        } catch {
          // Fall back to null if parsing fails
        }
      }
    }
    return null;
  });

  const [selectedThread, setSelectedThread] = useState<{
    x: number;
    y: number;
    z: number;
  } | null>(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("cuda-indexing-selected-thread");
      if (saved) {
        try {
          return JSON.parse(saved);
        } catch {
          // Fall back to null if parsing fails
        }
      }
    }
    return null;
  });

  const [currentView, setCurrentView] = useState<"grid" | "block">(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("cuda-indexing-current-view");
      if (saved === "block" || saved === "grid") {
        return saved;
      }
    }
    return "grid";
  });

  // Save state to localStorage when it changes
  useEffect(() => {
    if (typeof window !== "undefined") {
      localStorage.setItem("cuda-indexing-config", JSON.stringify(config));
    }
  }, [config]);

  useEffect(() => {
    if (typeof window !== "undefined") {
      localStorage.setItem(
        "cuda-indexing-selected-block",
        JSON.stringify(selectedBlock)
      );
    }
  }, [selectedBlock]);

  useEffect(() => {
    if (typeof window !== "undefined") {
      localStorage.setItem(
        "cuda-indexing-selected-thread",
        JSON.stringify(selectedThread)
      );
    }
  }, [selectedThread]);

  useEffect(() => {
    if (typeof window !== "undefined") {
      localStorage.setItem("cuda-indexing-current-view", currentView);
    }
  }, [currentView]);

  // Calculate totals
  const blocksPerGrid = config.gridX * config.gridY * config.gridZ;
  const threadsPerBlock = config.blockX * config.blockY * config.blockZ;
  const totalThreads = blocksPerGrid * threadsPerBlock;

  // Calculate linear block ID from 3D coordinates
  const calculateBlockId = (x: number, y: number, z: number) => {
    return x + y * config.gridX + z * config.gridX * config.gridY;
  };

  // Calculate linear thread ID within block
  const calculateThreadOffset = (x: number, y: number, z: number) => {
    return x + y * config.blockX + z * config.blockX * config.blockY;
  };

  // Calculate global thread ID
  const calculateGlobalId = (blockId: number, threadOffset: number) => {
    return blockId * threadsPerBlock + threadOffset;
  };

  // Get thread info for selected thread
  const getSelectedThreadInfo = (): ThreadInfo | null => {
    if (!selectedBlock || !selectedThread) return null;

    const blockId = calculateBlockId(
      selectedBlock.x,
      selectedBlock.y,
      selectedBlock.z
    );
    const threadOffset = calculateThreadOffset(
      selectedThread.x,
      selectedThread.y,
      selectedThread.z
    );
    const globalId = calculateGlobalId(blockId, threadOffset);

    return {
      blockIdx: selectedBlock,
      threadIdx: selectedThread,
      blockId,
      threadOffset,
      globalId,
    };
  };

  // CUDA code examples
  const cudaKernelCode = `__global__ void whoami() {
    //---------------------------------------------------------------------
    // 1. Derive a *linear* block ID from the 3-D block coordinates.
    //    Think of the grid as a city                     (z-dimension)
    //    made of buildings (y-dimension)
    //    with floors      (x-dimension).
    //---------------------------------------------------------------------
    int block_id =
        blockIdx.x +                       // apartment number on this floor
        blockIdx.y * gridDim.x +           // floor number within the building
        blockIdx.z * gridDim.x * gridDim.y;// building number within the city

    //---------------------------------------------------------------------
    // 2. Convert that block ID into a *global* offset for its first thread.
    //---------------------------------------------------------------------
    int threads_per_block = blockDim.x * blockDim.y * blockDim.z; // people / apt.
    int block_offset      = block_id * threads_per_block;         // first resident

    //---------------------------------------------------------------------
    // 3. Derive a *linear* thread index within its own block.
    //---------------------------------------------------------------------
    int thread_offset =
        threadIdx.x +                       // position within a row
        threadIdx.y * blockDim.x +          // row within a column
        threadIdx.z * blockDim.x * blockDim.y; // layer within the stack

    //---------------------------------------------------------------------
    // 4. Combine the two to obtain a unique global thread ID.
    //---------------------------------------------------------------------
    int global_id = block_offset + thread_offset;

    //---------------------------------------------------------------------
    // 5. Report the mapping for this thread.
    //---------------------------------------------------------------------
    printf("%04d | Block(%d %d %d) = %3d | Thread(%d %d %d) = %3d\\n",
        global_id,
        blockIdx.x, blockIdx.y, blockIdx.z, block_id,
        threadIdx.x, threadIdx.y, threadIdx.z, thread_offset);
}`;

  const hostCode = `int main() {
    //---------------------------------------------------------------------
    // 1. Define grid (block) dimensions.
    //    Grid:  ${config.gridX} × ${config.gridY} × ${config.gridZ}  = ${blocksPerGrid} blocks
    //    Block: ${config.blockX} × ${config.blockY} × ${config.blockZ}  = ${threadsPerBlock} threads
    //---------------------------------------------------------------------
    const int b_x = ${config.gridX}, b_y = ${config.gridY}, b_z = ${config.gridZ};         // gridDim.{x,y,z}
    const int t_x = ${config.blockX}, t_y = ${config.blockY}, t_z = ${config.blockZ};         // blockDim.{x,y,z}

    printf("%d blocks/grid\\n",   ${blocksPerGrid});
    printf("%d threads/block\\n", ${threadsPerBlock});
    printf("%d total threads\\n", ${totalThreads});

    //---------------------------------------------------------------------
    // 2. Launch the kernel.
    //---------------------------------------------------------------------
    dim3 blocksPerGrid(b_x, b_y, b_z);   // <<<grid>>>
    dim3 threadsPerBlock(t_x, t_y, t_z); // <<<block>>>

    whoami<<<blocksPerGrid, threadsPerBlock>>>();
    
    cudaDeviceSynchronize();
}`;

  // Grid visualization - shows blocks in 3D grid
  const GridVisualization: React.FC = () => {
    const renderBlocks = () => {
      const blocks: React.ReactElement[] = [];

      for (let z = 0; z < config.gridZ; z++) {
        blocks.push(
          <div key={`layer-${z}`} className="mb-4">
            <h4 className="text-sm font-semibold mb-2 text-gray-700">
              Building Layer {z} (blockIdx.z = {z})
            </h4>
            <div className="border-2 border-gray-300 p-2 bg-gray-50 rounded">
              <div
                className="grid gap-1"
                style={{ gridTemplateColumns: `repeat(${config.gridX}, 1fr)` }}
              >
                {Array.from({ length: config.gridY }).map((_, y) =>
                  Array.from({ length: config.gridX }).map((_, x) => {
                    const blockId = calculateBlockId(x, y, z);
                    const isSelected =
                      selectedBlock &&
                      selectedBlock.x === x &&
                      selectedBlock.y === y &&
                      selectedBlock.z === z;

                    return (
                      <div
                        key={`block-${x}-${y}-${z}`}
                        className={`w-12 h-12 border-2 flex flex-col items-center justify-center text-xs cursor-pointer transition-colors ${
                          isSelected
                            ? "bg-blue-200 border-blue-500 text-blue-900"
                            : "bg-white border-gray-400 text-gray-700 hover:bg-gray-100"
                        }`}
                        onClick={() => {
                          setSelectedBlock({ x, y, z });
                          setCurrentView("block");
                        }}
                        title={`Block (${x}, ${y}, ${z}) - ID: ${blockId}`}
                      >
                        <div className="font-bold">{blockId}</div>
                        <div className="text-xs">
                          ({x},{y},{z})
                        </div>
                      </div>
                    );
                  })
                )}
              </div>
              <div className="text-xs text-gray-600 mt-1">
                Y-axis (floors): 0 to {config.gridY - 1} →
              </div>
            </div>
            <div className="text-xs text-gray-600 mt-1">
              X-axis (apartments): 0 to {config.gridX - 1} →
            </div>
          </div>
        );
      }

      return blocks;
    };

    return (
      <Card>
        <CardHeader>
          <CardTitle>3D Grid Visualization - "The City"</CardTitle>
          <CardDescription>
            {blocksPerGrid} blocks arranged in a {config.gridX}×{config.gridY}×
            {config.gridZ} grid
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {renderBlocks()}
            <Alert>
              <AlertDescription>
                <strong>Apartment Complex Analogy:</strong>
                <br />• <strong>Z-dimension</strong>: Buildings in the city
                <br />• <strong>Y-dimension</strong>: Floors in each building
                <br />• <strong>X-dimension</strong>: Apartments on each floor
                <br />
                Click any block to see its threads!
              </AlertDescription>
            </Alert>
          </div>
        </CardContent>
      </Card>
    );
  };

  // Block visualization - shows threads within selected block
  const BlockVisualization: React.FC = () => {
    if (!selectedBlock) {
      return (
        <Card>
          <CardHeader>
            <CardTitle>Block Visualization - "Inside the Building"</CardTitle>
            <CardDescription>
              Select a block from the grid view to see its threads
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-center py-8 text-gray-500">
              Click on a block in the grid view to explore its threads
            </div>
          </CardContent>
        </Card>
      );
    }

    const blockId = calculateBlockId(
      selectedBlock.x,
      selectedBlock.y,
      selectedBlock.z
    );

    const renderThreads = () => {
      const threads: React.ReactElement[] = [];

      for (let z = 0; z < config.blockZ; z++) {
        threads.push(
          <div key={`thread-layer-${z}`} className="mb-4">
            <h4 className="text-sm font-semibold mb-2 text-gray-700">
              Thread Layer {z} (threadIdx.z = {z})
            </h4>
            <div className="border-2 border-blue-300 p-2 bg-blue-50 rounded">
              <div
                className="grid gap-1"
                style={{ gridTemplateColumns: `repeat(${config.blockX}, 1fr)` }}
              >
                {Array.from({ length: config.blockY }).map((_, y) =>
                  Array.from({ length: config.blockX }).map((_, x) => {
                    const threadOffset = calculateThreadOffset(x, y, z);
                    const globalId = calculateGlobalId(blockId, threadOffset);
                    const isSelected =
                      selectedThread &&
                      selectedThread.x === x &&
                      selectedThread.y === y &&
                      selectedThread.z === z;

                    return (
                      <div
                        key={`thread-${x}-${y}-${z}`}
                        className={`w-8 h-8 border flex flex-col items-center justify-center text-xs cursor-pointer transition-colors ${
                          isSelected
                            ? "bg-orange-200 border-orange-500 text-orange-900"
                            : "bg-white border-blue-400 text-blue-700 hover:bg-blue-100"
                        }`}
                        onClick={() => setSelectedThread({ x, y, z })}
                        title={`Thread (${x}, ${y}, ${z}) - Offset: ${threadOffset}, Global ID: ${globalId}`}
                      >
                        <div className="font-bold text-xs">{threadOffset}</div>
                      </div>
                    );
                  })
                )}
              </div>
            </div>
          </div>
        );
      }

      return threads;
    };

    return (
      <Card>
        <CardHeader>
          <CardTitle>
            Block {blockId} - "Inside Building ({selectedBlock.x},{" "}
            {selectedBlock.y}, {selectedBlock.z})"
          </CardTitle>
          <CardDescription>
            {threadsPerBlock} threads arranged in a {config.blockX}×
            {config.blockY}×{config.blockZ} block
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="bg-blue-50 p-3 rounded border border-blue-200">
              <h4 className="font-semibold text-blue-800 mb-1">Block Info:</h4>
              <div className="text-sm text-blue-700">
                <div>
                  Block Coordinates: ({selectedBlock.x}, {selectedBlock.y},{" "}
                  {selectedBlock.z})
                </div>
                <div>Linear Block ID: {blockId}</div>
                <div>First Thread Global ID: {blockId * threadsPerBlock}</div>
                <div>
                  Last Thread Global ID: {(blockId + 1) * threadsPerBlock - 1}
                </div>
              </div>
            </div>

            {renderThreads()}

            <Button
              onClick={() => setCurrentView("grid")}
              variant="outline"
              className="w-full"
            >
              ← Back to Grid View
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  };

  // Thread calculation explanation
  const ThreadCalculationExplanation: React.FC = () => {
    const threadInfo = getSelectedThreadInfo();

    if (!threadInfo) {
      return (
        <Card>
          <CardHeader>
            <CardTitle>Thread ID Calculation</CardTitle>
            <CardDescription>
              Select a block and thread to see the indexing calculation
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-center py-8 text-gray-500">
              Select a block from the grid, then select a thread within that
              block
            </div>
          </CardContent>
        </Card>
      );
    }

    return (
      <Card>
        <CardHeader>
          <CardTitle>Thread ID Calculation Step-by-Step</CardTitle>
          <CardDescription>
            How thread ({threadInfo.threadIdx.x}, {threadInfo.threadIdx.y},{" "}
            {threadInfo.threadIdx.z}) in block ({threadInfo.blockIdx.x},{" "}
            {threadInfo.blockIdx.y}, {threadInfo.blockIdx.z}) gets global ID{" "}
            {threadInfo.globalId}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold mb-2 text-gray-800">
                Step 1: Linear Block ID
              </h4>
              <div className="bg-gray-50 p-3 rounded text-sm font-mono space-y-1">
                <div>
                  block_id = blockIdx.x + blockIdx.y × gridDim.x + blockIdx.z ×
                  gridDim.x × gridDim.y
                </div>
                <div>
                  block_id = {threadInfo.blockIdx.x} + {threadInfo.blockIdx.y} ×{" "}
                  {config.gridX} + {threadInfo.blockIdx.z} × {config.gridX} ×{" "}
                  {config.gridY}
                </div>
                <div>
                  block_id = {threadInfo.blockIdx.x} +{" "}
                  {threadInfo.blockIdx.y * config.gridX} +{" "}
                  {threadInfo.blockIdx.z * config.gridX * config.gridY}
                </div>
                <div className="font-bold text-blue-600">
                  block_id = {threadInfo.blockId}
                </div>
              </div>
            </div>

            <div>
              <h4 className="font-semibold mb-2 text-gray-800">
                Step 2: Block Offset
              </h4>
              <div className="bg-gray-50 p-3 rounded text-sm font-mono space-y-1">
                <div>
                  threads_per_block = {config.blockX} × {config.blockY} ×{" "}
                  {config.blockZ} = {threadsPerBlock}
                </div>
                <div>block_offset = block_id × threads_per_block</div>
                <div>
                  block_offset = {threadInfo.blockId} × {threadsPerBlock}
                </div>
                <div className="font-bold text-blue-600">
                  block_offset = {threadInfo.blockId * threadsPerBlock}
                </div>
              </div>
            </div>

            <div>
              <h4 className="font-semibold mb-2 text-gray-800">
                Step 3: Thread Offset Within Block
              </h4>
              <div className="bg-gray-50 p-3 rounded text-sm font-mono space-y-1">
                <div>
                  thread_offset = threadIdx.x + threadIdx.y × blockDim.x +
                  threadIdx.z × blockDim.x × blockDim.y
                </div>
                <div>
                  thread_offset = {threadInfo.threadIdx.x} +{" "}
                  {threadInfo.threadIdx.y} × {config.blockX} +{" "}
                  {threadInfo.threadIdx.z} × {config.blockX} × {config.blockY}
                </div>
                <div>
                  thread_offset = {threadInfo.threadIdx.x} +{" "}
                  {threadInfo.threadIdx.y * config.blockX} +{" "}
                  {threadInfo.threadIdx.z * config.blockX * config.blockY}
                </div>
                <div className="font-bold text-purple-600">
                  thread_offset = {threadInfo.threadOffset}
                </div>
              </div>
            </div>

            <div>
              <h4 className="font-semibold mb-2 text-gray-800">
                Step 4: Global Thread ID
              </h4>
              <div className="bg-gray-50 p-3 rounded text-sm font-mono space-y-1">
                <div>global_id = block_offset + thread_offset</div>
                <div>
                  global_id = {threadInfo.blockId * threadsPerBlock} +{" "}
                  {threadInfo.threadOffset}
                </div>
                <div className="font-bold text-green-600 text-lg">
                  global_id = {threadInfo.globalId}
                </div>
              </div>
            </div>
          </div>

          <Alert>
            <AlertDescription>
              <strong>Key Insight:</strong> Each thread gets a unique global ID
              by combining its block's starting position with its position
              within that block. This allows CUDA to give every thread in the
              entire grid a unique identifier!
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    );
  };

  return (
    <div className="w-full max-w-7xl mx-auto p-6 space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="text-2xl">
            CUDA 3D Indexing Tutorial - "whoami.cu"
          </CardTitle>
          <CardDescription>
            Learn how CUDA maps 3D thread coordinates to linear IDs using the
            apartment complex analogy
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-6">
            <div>
              <label className="block text-sm font-medium mb-2">
                Grid X: {config.gridX}
              </label>
              <Slider
                value={[config.gridX]}
                onValueChange={(value) =>
                  setConfig((prev) => ({ ...prev, gridX: value[0] }))
                }
                min={1}
                max={4}
                step={1}
                className="w-full"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">
                Grid Y: {config.gridY}
              </label>
              <Slider
                value={[config.gridY]}
                onValueChange={(value) =>
                  setConfig((prev) => ({ ...prev, gridY: value[0] }))
                }
                min={1}
                max={4}
                step={1}
                className="w-full"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">
                Grid Z: {config.gridZ}
              </label>
              <Slider
                value={[config.gridZ]}
                onValueChange={(value) =>
                  setConfig((prev) => ({ ...prev, gridZ: value[0] }))
                }
                min={1}
                max={4}
                step={1}
                className="w-full"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">
                Block X: {config.blockX}
              </label>
              <Slider
                value={[config.blockX]}
                onValueChange={(value) =>
                  setConfig((prev) => ({ ...prev, blockX: value[0] }))
                }
                min={1}
                max={4}
                step={1}
                className="w-full"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">
                Block Y: {config.blockY}
              </label>
              <Slider
                value={[config.blockY]}
                onValueChange={(value) =>
                  setConfig((prev) => ({ ...prev, blockY: value[0] }))
                }
                min={1}
                max={4}
                step={1}
                className="w-full"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">
                Block Z: {config.blockZ}
              </label>
              <Slider
                value={[config.blockZ]}
                onValueChange={(value) =>
                  setConfig((prev) => ({ ...prev, blockZ: value[0] }))
                }
                min={1}
                max={4}
                step={1}
                className="w-full"
              />
            </div>
          </div>

          <div className="grid grid-cols-3 gap-4 text-center">
            <Badge variant="outline" className="text-sm p-2">
              {blocksPerGrid} Blocks Total
            </Badge>
            <Badge variant="outline" className="text-sm p-2">
              {threadsPerBlock} Threads/Block
            </Badge>
            <Badge variant="outline" className="text-sm p-2">
              {totalThreads} Total Threads
            </Badge>
          </div>
        </CardContent>
      </Card>

      <Tabs defaultValue="visualization" className="w-full">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="visualization">3D Visualization</TabsTrigger>
          <TabsTrigger value="calculation">ID Calculation</TabsTrigger>
          <TabsTrigger value="walkthrough">Code Walkthrough</TabsTrigger>
          <TabsTrigger value="code">CUDA Code</TabsTrigger>
          <TabsTrigger value="explanation">Concepts</TabsTrigger>
        </TabsList>

        <TabsContent value="visualization" className="space-y-6">
          <div className="flex gap-4 mb-4">
            <Button
              onClick={() => setCurrentView("grid")}
              variant={currentView === "grid" ? "default" : "outline"}
            >
              🏙️ Grid View (City)
            </Button>
            <Button
              onClick={() => setCurrentView("block")}
              variant={currentView === "block" ? "default" : "outline"}
              disabled={!selectedBlock}
            >
              🏢 Block View (Building)
            </Button>
          </div>

          {currentView === "grid" ? (
            <GridVisualization />
          ) : (
            <BlockVisualization />
          )}
        </TabsContent>

        <TabsContent value="calculation" className="space-y-6">
          <ThreadCalculationExplanation />
        </TabsContent>

        <TabsContent value="walkthrough" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>whoami.cu - Block by Block Code Explanation</CardTitle>
              <CardDescription>
                Detailed walkthrough of each section of the CUDA indexing code
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="border-l-4 border-blue-500 pl-4">
                <h3 className="text-lg font-semibold mb-2 text-blue-700">
                  🔧 Block 1: Kernel Function Declaration
                </h3>
                <div className="bg-gray-900 text-green-400 p-4 rounded-lg mb-3 overflow-x-auto">
                  <pre className="whitespace-pre text-sm">
                    <code>__global__ void whoami() &#123;</code>
                  </pre>
                </div>
                <div className="space-y-2 text-sm text-gray-700">
                  <p>
                    <strong>__global__</strong>: This qualifier tells the
                    compiler this function runs on the GPU
                  </p>
                  <p>
                    <strong>void</strong>: Returns nothing (GPU kernels can't
                    return values directly)
                  </p>
                  <p>
                    <strong>whoami</strong>: Function name - each thread will
                    execute this function
                  </p>
                  <p>
                    <strong>Key Point</strong>: Every single thread in your grid
                    executes this function simultaneously!
                  </p>
                </div>
              </div>

              <div className="border-l-4 border-green-500 pl-4">
                <h3 className="text-lg font-semibold mb-2 text-green-700">
                  🏗️ Block 2: Linear Block ID Calculation
                </h3>
                <div className="bg-gray-900 text-green-400 p-4 rounded-lg mb-3 overflow-x-auto">
                  <pre className="whitespace-pre text-sm">
                    <code>
                      {`// 1. Derive a linear block ID from the 3-D block coordinates.
//    Think of the grid as a city                     (z-dimension)
//    made of buildings (y-dimension)
//    with floors      (x-dimension).
int block_id =
    blockIdx.x +                       // apartment number on this floor
    blockIdx.y * gridDim.x +           // floor number within the building
    blockIdx.z * gridDim.x * gridDim.y;// building number within the city`}
                    </code>
                  </pre>
                </div>

                {/* Interactive Example */}
                <div className="bg-green-50 border border-green-200 p-4 rounded-lg mb-3">
                  <h4 className="font-semibold text-green-800 mb-2">
                    🧮 Interactive Example:
                  </h4>
                  <div className="space-y-3">
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <p className="text-sm font-medium text-green-700">
                          Grid Dimensions:
                        </p>
                        <div className="text-sm font-mono bg-white p-2 rounded text-gray-800">
                          gridDim.x = {config.gridX}
                          <br />
                          gridDim.y = {config.gridY}
                          <br />
                          gridDim.z = {config.gridZ}
                        </div>
                      </div>
                      <div>
                        <p className="text-sm font-medium text-green-700">
                          Example Block Coordinates:
                        </p>
                        <div className="text-sm font-mono bg-white p-2 rounded text-gray-800">
                          blockIdx.x = {selectedBlock ? selectedBlock.x : 1}
                          <br />
                          blockIdx.y = {selectedBlock ? selectedBlock.y : 1}
                          <br />
                          blockIdx.z = {selectedBlock ? selectedBlock.z : 1}
                        </div>
                      </div>
                    </div>

                    <div className="bg-white p-3 rounded border">
                      <p className="font-medium text-green-800 mb-2">
                        Step-by-step calculation:
                      </p>
                      <div className="text-sm font-mono space-y-1 text-gray-800">
                        <div>
                          block_id = blockIdx.x + blockIdx.y × gridDim.x +
                          blockIdx.z × gridDim.x × gridDim.y
                        </div>
                        <div>
                          block_id = {selectedBlock ? selectedBlock.x : 1} +{" "}
                          {selectedBlock ? selectedBlock.y : 1} × {config.gridX}{" "}
                          + {selectedBlock ? selectedBlock.z : 1} ×{" "}
                          {config.gridX} × {config.gridY}
                        </div>
                        <div>
                          block_id = {selectedBlock ? selectedBlock.x : 1} +{" "}
                          {selectedBlock
                            ? selectedBlock.y * config.gridX
                            : config.gridX}{" "}
                          +{" "}
                          {selectedBlock
                            ? selectedBlock.z * config.gridX * config.gridY
                            : config.gridX * config.gridY}
                        </div>
                        <div className="font-bold text-green-600">
                          block_id ={" "}
                          {selectedBlock
                            ? calculateBlockId(
                                selectedBlock.x,
                                selectedBlock.y,
                                selectedBlock.z
                              )
                            : 1 + config.gridX + config.gridX * config.gridY}
                        </div>
                      </div>
                    </div>

                    <div className="text-xs text-green-600">
                      💡{" "}
                      {selectedBlock
                        ? `Block (${selectedBlock.x}, ${selectedBlock.y}, ${selectedBlock.z})`
                        : "Select a block from the visualization"}{" "}
                      to see live calculations!
                    </div>
                  </div>
                </div>

                <div className="space-y-2 text-sm text-gray-700">
                  <p>
                    <strong>blockIdx.x, blockIdx.y, blockIdx.z</strong>:
                    Built-in CUDA variables giving this block's 3D coordinates
                  </p>
                  <p>
                    <strong>gridDim.x, gridDim.y</strong>: Built-in variables
                    giving the grid dimensions
                  </p>
                  <p>
                    <strong>Formula Logic</strong>: Converts 3D coordinates to a
                    single unique number (like street addresses)
                  </p>
                  <p>
                    <strong>Apartment Analogy</strong>: X = apartment on floor,
                    Y = floor in building, Z = building in city
                  </p>
                  <p>
                    <strong>Math</strong>: Same as array indexing:{" "}
                    <code>index = z×(width×height) + y×width + x</code>
                  </p>
                </div>
              </div>

              <div className="border-l-4 border-purple-500 pl-4">
                <h3 className="text-lg font-semibold mb-2 text-purple-700">
                  📍 Block 3: Global Thread Offset Calculation
                </h3>
                <div className="bg-gray-900 text-green-400 p-4 rounded-lg mb-3 overflow-x-auto">
                  <pre className="whitespace-pre text-sm">
                    <code>
                      {`// 2. Convert that block ID into a global offset for its first thread.
int threads_per_block = blockDim.x * blockDim.y * blockDim.z; // people / apt.
int block_offset      = block_id * threads_per_block;         // first resident`}
                    </code>
                  </pre>
                </div>

                {/* Interactive Example */}
                <div className="bg-purple-50 border border-purple-200 p-4 rounded-lg mb-3">
                  <h4 className="font-semibold text-purple-800 mb-2">
                    🧮 Interactive Example:
                  </h4>
                  <div className="space-y-3">
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <p className="text-sm font-medium text-purple-700">
                          Block Dimensions:
                        </p>
                        <div className="text-sm font-mono bg-white p-2 rounded text-gray-800">
                          blockDim.x = {config.blockX}
                          <br />
                          blockDim.y = {config.blockY}
                          <br />
                          blockDim.z = {config.blockZ}
                        </div>
                      </div>
                      <div>
                        <p className="text-sm font-medium text-purple-700">
                          Selected Block:
                        </p>
                        <div className="text-sm font-mono bg-white p-2 rounded text-gray-800">
                          block_id ={" "}
                          {selectedBlock
                            ? calculateBlockId(
                                selectedBlock.x,
                                selectedBlock.y,
                                selectedBlock.z
                              )
                            : "N/A"}
                          <br />
                          coords = ({selectedBlock
                            ? selectedBlock.x
                            : "N/A"}, {selectedBlock ? selectedBlock.y : "N/A"},{" "}
                          {selectedBlock ? selectedBlock.z : "N/A"})
                        </div>
                      </div>
                    </div>

                    <div className="bg-white p-3 rounded border">
                      <p className="font-medium text-purple-800 mb-2">
                        Step-by-step calculation:
                      </p>
                      <div className="text-sm font-mono space-y-1 text-gray-800">
                        <div>
                          threads_per_block = blockDim.x × blockDim.y ×
                          blockDim.z
                        </div>
                        <div>
                          threads_per_block = {config.blockX} × {config.blockY}{" "}
                          × {config.blockZ}
                        </div>
                        <div className="font-bold text-purple-600">
                          threads_per_block = {threadsPerBlock}
                        </div>
                        <div className="mt-2">
                          block_offset = block_id × threads_per_block
                        </div>
                        <div>
                          block_offset ={" "}
                          {selectedBlock
                            ? calculateBlockId(
                                selectedBlock.x,
                                selectedBlock.y,
                                selectedBlock.z
                              )
                            : "N/A"}{" "}
                          × {threadsPerBlock}
                        </div>
                        <div className="font-bold text-purple-600">
                          block_offset ={" "}
                          {selectedBlock
                            ? calculateBlockId(
                                selectedBlock.x,
                                selectedBlock.y,
                                selectedBlock.z
                              ) * threadsPerBlock
                            : "N/A"}
                        </div>
                      </div>
                    </div>

                    <div className="bg-white p-3 rounded border">
                      <p className="font-medium text-purple-800 mb-2">
                        🏢 Building Analogy:
                      </p>
                      <div className="text-sm text-gray-700">
                        {selectedBlock ? (
                          <div>
                            Building{" "}
                            {calculateBlockId(
                              selectedBlock.x,
                              selectedBlock.y,
                              selectedBlock.z
                            )}{" "}
                            has {threadsPerBlock} residents.
                            <br />
                            Their global IDs start from{" "}
                            {calculateBlockId(
                              selectedBlock.x,
                              selectedBlock.y,
                              selectedBlock.z
                            ) * threadsPerBlock}{" "}
                            and go up to{" "}
                            {(calculateBlockId(
                              selectedBlock.x,
                              selectedBlock.y,
                              selectedBlock.z
                            ) +
                              1) *
                              threadsPerBlock -
                              1}
                            .
                          </div>
                        ) : (
                          "Select a block to see the building analogy!"
                        )}
                      </div>
                    </div>
                  </div>
                </div>

                <div className="space-y-2 text-sm text-gray-700">
                  <p>
                    <strong>blockDim.x, blockDim.y, blockDim.z</strong>:
                    Built-in variables giving block dimensions
                  </p>
                  <p>
                    <strong>threads_per_block</strong>: Total number of threads
                    in each block (same for all blocks)
                  </p>
                  <p>
                    <strong>block_offset</strong>: Global thread ID where this
                    block's threads start
                  </p>
                  <p>
                    <strong>Example</strong>: If each block has 64 threads,
                    block 0 starts at 0, block 1 starts at 64, block 2 starts at
                    128, etc.
                  </p>
                  <p>
                    <strong>Apartment Analogy</strong>: Like saying "Building 5
                    residents have IDs starting from 320"
                  </p>
                </div>
              </div>

              <div className="border-l-4 border-orange-500 pl-4">
                <h3 className="text-lg font-semibold mb-2 text-orange-700">
                  👤 Block 4: Thread Position Within Block
                </h3>
                <div className="bg-gray-900 text-green-400 p-4 rounded-lg mb-3 overflow-x-auto">
                  <pre className="whitespace-pre text-sm">
                    <code>
                      {`// 3. Derive a linear thread index within its own block.
int thread_offset =
    threadIdx.x +                       // position within a row
    threadIdx.y * blockDim.x +          // row within a column
    threadIdx.z * blockDim.x * blockDim.y; // layer within the stack`}
                    </code>
                  </pre>
                </div>

                {/* Interactive Example */}
                <div className="bg-orange-50 border border-orange-200 p-4 rounded-lg mb-3">
                  <h4 className="font-semibold text-orange-800 mb-2">
                    🧮 Interactive Example:
                  </h4>
                  <div className="space-y-3">
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <p className="text-sm font-medium text-orange-700">
                          Block Dimensions:
                        </p>
                        <div className="text-sm font-mono bg-white p-2 rounded text-gray-800">
                          blockDim.x = {config.blockX}
                          <br />
                          blockDim.y = {config.blockY}
                          <br />
                          blockDim.z = {config.blockZ}
                        </div>
                      </div>
                      <div>
                        <p className="text-sm font-medium text-orange-700">
                          Example Thread Coordinates:
                        </p>
                        <div className="text-sm font-mono bg-white p-2 rounded text-gray-800">
                          threadIdx.x = {selectedThread ? selectedThread.x : 1}
                          <br />
                          threadIdx.y = {selectedThread ? selectedThread.y : 1}
                          <br />
                          threadIdx.z = {selectedThread ? selectedThread.z : 1}
                        </div>
                      </div>
                    </div>

                    <div className="bg-white p-3 rounded border">
                      <p className="font-medium text-orange-800 mb-2">
                        Step-by-step calculation:
                      </p>
                      <div className="text-sm font-mono space-y-1 text-gray-800">
                        <div>
                          thread_offset = threadIdx.x + threadIdx.y × blockDim.x
                          + threadIdx.z × blockDim.x × blockDim.y
                        </div>
                        <div>
                          thread_offset ={" "}
                          {selectedThread ? selectedThread.x : 1} +{" "}
                          {selectedThread ? selectedThread.y : 1} ×{" "}
                          {config.blockX} +{" "}
                          {selectedThread ? selectedThread.z : 1} ×{" "}
                          {config.blockX} × {config.blockY}
                        </div>
                        <div>
                          thread_offset ={" "}
                          {selectedThread ? selectedThread.x : 1} +{" "}
                          {selectedThread
                            ? selectedThread.y * config.blockX
                            : config.blockX}{" "}
                          +{" "}
                          {selectedThread
                            ? selectedThread.z * config.blockX * config.blockY
                            : config.blockX * config.blockY}
                        </div>
                        <div className="font-bold text-orange-600">
                          thread_offset ={" "}
                          {selectedThread
                            ? calculateThreadOffset(
                                selectedThread.x,
                                selectedThread.y,
                                selectedThread.z
                              )
                            : 1 + config.blockX + config.blockX * config.blockY}
                        </div>
                      </div>
                    </div>

                    <div className="bg-white p-3 rounded border">
                      <p className="font-medium text-orange-800 mb-2">
                        🏠 Apartment Analogy:
                      </p>
                      <div className="text-sm text-gray-700">
                        {selectedThread ? (
                          <div>
                            In the selected building, this resident lives in
                            apartment #
                            {calculateThreadOffset(
                              selectedThread.x,
                              selectedThread.y,
                              selectedThread.z
                            )}
                            .<br />
                            <span className="text-orange-600">
                              Position within building:
                            </span>{" "}
                            Floor {selectedThread.z}, Row {selectedThread.y},
                            Unit {selectedThread.x}
                          </div>
                        ) : (
                          "Select a thread to see the apartment analogy!"
                        )}
                      </div>
                    </div>

                    <div className="text-xs text-orange-600">
                      💡{" "}
                      {selectedThread
                        ? `Thread (${selectedThread.x}, ${selectedThread.y}, ${selectedThread.z})`
                        : "Select a thread from the block view"}{" "}
                      to see live calculations!
                    </div>
                  </div>
                </div>

                <div className="space-y-2 text-sm text-gray-700">
                  <p>
                    <strong>threadIdx.x, threadIdx.y, threadIdx.z</strong>:
                    Built-in variables giving this thread's position within its
                    block
                  </p>
                  <p>
                    <strong>Same Formula</strong>: Uses the same 3D→1D
                    conversion as for blocks, but within the block
                  </p>
                  <p>
                    <strong>Range</strong>: thread_offset goes from 0 to
                    (threads_per_block - 1)
                  </p>
                  <p>
                    <strong>Apartment Analogy</strong>: Like apartment numbers
                    within a single building (0, 1, 2, ... 63)
                  </p>
                  <p>
                    <strong>Important</strong>: This is relative to the block,
                    not the entire grid!
                  </p>
                </div>
              </div>

              <div className="border-l-4 border-red-500 pl-4">
                <h3 className="text-lg font-semibold mb-2 text-red-700">
                  🌍 Block 5: Global Thread ID
                </h3>
                <div className="bg-gray-900 text-green-400 p-4 rounded-lg mb-3 overflow-x-auto">
                  <pre className="whitespace-pre text-sm">
                    <code>
                      {`// 4. Combine the two to obtain a unique global thread ID.
int global_id = block_offset + thread_offset;`}
                    </code>
                  </pre>
                </div>

                {/* Interactive Example */}
                <div className="bg-red-50 border border-red-200 p-4 rounded-lg mb-3">
                  <h4 className="font-semibold text-red-800 mb-2">
                    🧮 Interactive Example:
                  </h4>
                  <div className="space-y-3">
                    {selectedBlock && selectedThread ? (
                      <>
                        <div className="grid grid-cols-2 gap-4">
                          <div>
                            <p className="text-sm font-medium text-red-700">
                              From Previous Steps:
                            </p>
                            <div className="text-sm font-mono bg-white p-2 rounded text-gray-800">
                              block_offset ={" "}
                              {calculateBlockId(
                                selectedBlock.x,
                                selectedBlock.y,
                                selectedBlock.z
                              ) * threadsPerBlock}
                              <br />
                              thread_offset ={" "}
                              {calculateThreadOffset(
                                selectedThread.x,
                                selectedThread.y,
                                selectedThread.z
                              )}
                            </div>
                          </div>
                          <div>
                            <p className="text-sm font-medium text-red-700">
                              Current Selection:
                            </p>
                            <div className="text-sm font-mono bg-white p-2 rounded text-gray-800">
                              Block: ({selectedBlock.x}, {selectedBlock.y},{" "}
                              {selectedBlock.z})<br />
                              Thread: ({selectedThread.x}, {selectedThread.y},{" "}
                              {selectedThread.z})
                            </div>
                          </div>
                        </div>

                        <div className="bg-white p-3 rounded border">
                          <p className="font-medium text-red-800 mb-2">
                            Final calculation:
                          </p>
                          <div className="text-sm font-mono space-y-1 text-gray-800">
                            <div>global_id = block_offset + thread_offset</div>
                            <div>
                              global_id ={" "}
                              {calculateBlockId(
                                selectedBlock.x,
                                selectedBlock.y,
                                selectedBlock.z
                              ) * threadsPerBlock}{" "}
                              +{" "}
                              {calculateThreadOffset(
                                selectedThread.x,
                                selectedThread.y,
                                selectedThread.z
                              )}
                            </div>
                            <div className="font-bold text-red-600 text-lg">
                              global_id ={" "}
                              {calculateGlobalId(
                                calculateBlockId(
                                  selectedBlock.x,
                                  selectedBlock.y,
                                  selectedBlock.z
                                ),
                                calculateThreadOffset(
                                  selectedThread.x,
                                  selectedThread.y,
                                  selectedThread.z
                                )
                              )}
                            </div>
                          </div>
                        </div>

                        <div className="bg-white p-3 rounded border">
                          <p className="font-medium text-red-800 mb-2">
                            🏘️ City Analogy:
                          </p>
                          <div className="text-sm text-gray-700">
                            This person lives in Building{" "}
                            {calculateBlockId(
                              selectedBlock.x,
                              selectedBlock.y,
                              selectedBlock.z
                            )}
                            , Apartment{" "}
                            {calculateThreadOffset(
                              selectedThread.x,
                              selectedThread.y,
                              selectedThread.z
                            )}
                            .<br />
                            <span className="text-red-600 font-medium">
                              Their unique city ID (like a social security
                              number) is:{" "}
                              {calculateGlobalId(
                                calculateBlockId(
                                  selectedBlock.x,
                                  selectedBlock.y,
                                  selectedBlock.z
                                ),
                                calculateThreadOffset(
                                  selectedThread.x,
                                  selectedThread.y,
                                  selectedThread.z
                                )
                              )}
                            </span>
                          </div>
                        </div>

                        <div className="bg-gradient-to-r from-red-100 to-red-200 p-3 rounded border border-red-300">
                          <p className="font-medium text-red-800 mb-1">
                            🎯 Complete Thread Identity:
                          </p>
                          <div className="text-sm text-gray-800">
                            <div>
                              <strong>3D Block:</strong> ({selectedBlock.x},{" "}
                              {selectedBlock.y}, {selectedBlock.z}) →{" "}
                              <strong>Linear Block ID:</strong>{" "}
                              {calculateBlockId(
                                selectedBlock.x,
                                selectedBlock.y,
                                selectedBlock.z
                              )}
                            </div>
                            <div>
                              <strong>3D Thread:</strong> ({selectedThread.x},{" "}
                              {selectedThread.y}, {selectedThread.z}) →{" "}
                              <strong>Linear Thread Offset:</strong>{" "}
                              {calculateThreadOffset(
                                selectedThread.x,
                                selectedThread.y,
                                selectedThread.z
                              )}
                            </div>
                            <div className="font-bold text-red-700 mt-1">
                              🌟 <strong>Global Thread ID:</strong>{" "}
                              {calculateGlobalId(
                                calculateBlockId(
                                  selectedBlock.x,
                                  selectedBlock.y,
                                  selectedBlock.z
                                ),
                                calculateThreadOffset(
                                  selectedThread.x,
                                  selectedThread.y,
                                  selectedThread.z
                                )
                              )}{" "}
                              (out of {totalThreads} total threads)
                            </div>
                          </div>
                        </div>
                      </>
                    ) : (
                      <div className="text-center py-4 text-red-600">
                        <p className="font-medium">
                          Select a block and thread to see the complete
                          calculation!
                        </p>
                        <p className="text-sm mt-1">
                          {!selectedBlock &&
                            "1️⃣ First select a block from the Grid View"}
                          {selectedBlock &&
                            !selectedThread &&
                            "2️⃣ Then select a thread from the Block View"}
                        </p>
                      </div>
                    )}
                  </div>
                </div>

                <div className="space-y-2 text-sm text-gray-700">
                  <p>
                    <strong>Simple Addition</strong>: Add the block's starting
                    position to the thread's position within the block
                  </p>
                  <p>
                    <strong>Unique ID</strong>: Every thread in the entire grid
                    gets a different global_id
                  </p>
                  <p>
                    <strong>Range</strong>: From 0 to (total_threads - 1)
                  </p>
                  <p>
                    <strong>Apartment Analogy</strong>: Like social security
                    numbers - everyone has a unique one
                  </p>
                  <p>
                    <strong>Usage</strong>: This ID can be used to determine
                    which data element each thread should process
                  </p>
                </div>
              </div>

              <div className="border-l-4 border-indigo-500 pl-4">
                <h3 className="text-lg font-semibold mb-2 text-indigo-700">
                  📄 Block 6: Print Thread Information
                </h3>
                <div className="bg-gray-900 text-green-400 p-4 rounded-lg mb-3 overflow-x-auto">
                  <pre className="whitespace-pre text-sm">
                    <code>
                      {`// 5. Report the mapping for this thread.
//    "%04d" pads the ID so the output aligns nicely.
printf("%04d | Block(%d %d %d) = %3d | Thread(%d %d %d) = %3d\\n",
    global_id,
    blockIdx.x, blockIdx.y, blockIdx.z, block_id,
    threadIdx.x, threadIdx.y, threadIdx.z, thread_offset);`}
                    </code>
                  </pre>
                </div>

                {/* Interactive Example */}
                <div className="bg-indigo-50 border border-indigo-200 p-4 rounded-lg mb-3">
                  <h4 className="font-semibold text-indigo-800 mb-2">
                    🧮 Interactive Example:
                  </h4>
                  <div className="space-y-3">
                    {selectedBlock && selectedThread ? (
                      <>
                        <div className="bg-white p-3 rounded border">
                          <p className="font-medium text-indigo-800 mb-2">
                            🖥️ Actual Printf Output:
                          </p>
                          <div className="bg-gray-900 text-green-400 p-3 rounded font-mono text-sm">
                            {String(
                              calculateGlobalId(
                                calculateBlockId(
                                  selectedBlock.x,
                                  selectedBlock.y,
                                  selectedBlock.z
                                ),
                                calculateThreadOffset(
                                  selectedThread.x,
                                  selectedThread.y,
                                  selectedThread.z
                                )
                              )
                            ).padStart(4, "0")}{" "}
                            | Block({selectedBlock.x} {selectedBlock.y}{" "}
                            {selectedBlock.z}) ={" "}
                            {String(
                              calculateBlockId(
                                selectedBlock.x,
                                selectedBlock.y,
                                selectedBlock.z
                              )
                            ).padStart(3, " ")}{" "}
                            | Thread({selectedThread.x} {selectedThread.y}{" "}
                            {selectedThread.z}) ={" "}
                            {String(
                              calculateThreadOffset(
                                selectedThread.x,
                                selectedThread.y,
                                selectedThread.z
                              )
                            ).padStart(3, " ")}
                          </div>
                        </div>

                        <div className="bg-white p-3 rounded border">
                          <p className="font-medium text-indigo-800 mb-2">
                            🔍 Format String Breakdown:
                          </p>
                          <div className="text-sm space-y-1 text-gray-700">
                            <div>
                              <code>%04d</code> →{" "}
                              <span className="font-mono bg-gray-100 px-1">
                                {String(
                                  calculateGlobalId(
                                    calculateBlockId(
                                      selectedBlock.x,
                                      selectedBlock.y,
                                      selectedBlock.z
                                    ),
                                    calculateThreadOffset(
                                      selectedThread.x,
                                      selectedThread.y,
                                      selectedThread.z
                                    )
                                  )
                                ).padStart(4, "0")}
                              </span>{" "}
                              (global_id with leading zeros)
                            </div>
                            <div>
                              <code>%d %d %d</code> →{" "}
                              <span className="font-mono bg-gray-100 px-1">
                                {selectedBlock.x} {selectedBlock.y}{" "}
                                {selectedBlock.z}
                              </span>{" "}
                              (block coordinates)
                            </div>
                            <div>
                              <code>%3d</code> →{" "}
                              <span className="font-mono bg-gray-100 px-1">
                                {String(
                                  calculateBlockId(
                                    selectedBlock.x,
                                    selectedBlock.y,
                                    selectedBlock.z
                                  )
                                ).padStart(3, " ")}
                              </span>{" "}
                              (block_id right-aligned)
                            </div>
                            <div>
                              <code>%d %d %d</code> →{" "}
                              <span className="font-mono bg-gray-100 px-1">
                                {selectedThread.x} {selectedThread.y}{" "}
                                {selectedThread.z}
                              </span>{" "}
                              (thread coordinates)
                            </div>
                            <div>
                              <code>%3d</code> →{" "}
                              <span className="font-mono bg-gray-100 px-1">
                                {String(
                                  calculateThreadOffset(
                                    selectedThread.x,
                                    selectedThread.y,
                                    selectedThread.z
                                  )
                                ).padStart(3, " ")}
                              </span>{" "}
                              (thread_offset right-aligned)
                            </div>
                          </div>
                        </div>

                        <div className="bg-white p-3 rounded border">
                          <p className="font-medium text-indigo-800 mb-2">
                            📊 Sample Output from Multiple Threads:
                          </p>
                          <div className="bg-gray-900 text-green-400 p-3 rounded font-mono text-xs space-y-1">
                            <div>
                              0000 | Block(0 0 0) = 0 | Thread(0 0 0) = 0
                            </div>
                            <div>
                              0001 | Block(0 0 0) = 0 | Thread(1 0 0) = 1
                            </div>
                            <div>
                              0002 | Block(0 0 0) = 0 | Thread(0 1 0) ={" "}
                              {config.blockX}
                            </div>
                            <div>
                              0003 | Block(0 0 0) = 0 | Thread(1 1 0) ={" "}
                              {config.blockX + 1}
                            </div>
                            <div className="text-yellow-400">
                              ... (output from {totalThreads} threads total)
                            </div>
                            <div className="text-green-300">
                              ← Current thread:{" "}
                              {String(
                                calculateGlobalId(
                                  calculateBlockId(
                                    selectedBlock.x,
                                    selectedBlock.y,
                                    selectedBlock.z
                                  ),
                                  calculateThreadOffset(
                                    selectedThread.x,
                                    selectedThread.y,
                                    selectedThread.z
                                  )
                                )
                              ).padStart(4, "0")}{" "}
                              | Block({selectedBlock.x} {selectedBlock.y}{" "}
                              {selectedBlock.z}) ={" "}
                              {String(
                                calculateBlockId(
                                  selectedBlock.x,
                                  selectedBlock.y,
                                  selectedBlock.z
                                )
                              ).padStart(3, " ")}{" "}
                              | Thread({selectedThread.x} {selectedThread.y}{" "}
                              {selectedThread.z}) ={" "}
                              {String(
                                calculateThreadOffset(
                                  selectedThread.x,
                                  selectedThread.y,
                                  selectedThread.z
                                )
                              ).padStart(3, " ")}
                            </div>
                          </div>
                        </div>
                      </>
                    ) : (
                      <div className="text-center py-4 text-indigo-600">
                        <p className="font-medium">
                          Select a block and thread to see the printf output!
                        </p>
                      </div>
                    )}
                  </div>
                </div>

                <div className="space-y-2 text-sm text-gray-700">
                  <p>
                    <strong>printf</strong>: Each thread prints its own
                    information (output will be mixed from all threads)
                  </p>
                  <p>
                    <strong>%04d</strong>: Formats the global_id with leading
                    zeros (e.g., 0042 instead of 42)
                  </p>
                  <p>
                    <strong>Output Format</strong>: Shows global ID, 3D block
                    coords, linear block ID, 3D thread coords, linear thread
                    offset
                  </p>
                  <p>
                    <strong>Example Output</strong>:{" "}
                    <code>0067 | Block(1 0 2) = 5 | Thread(3 0 1) = 19</code>
                  </p>
                  <p>
                    <strong>Debugging Tool</strong>: This helps you verify that
                    your thread indexing is working correctly
                  </p>
                </div>
              </div>

              <div className="border-l-4 border-teal-500 pl-4">
                <h3 className="text-lg font-semibold mb-2 text-teal-700">
                  💻 Host Code: Setting Up the Launch
                </h3>
                <div className="bg-gray-900 text-green-400 p-4 rounded-lg mb-3 overflow-x-auto">
                  <pre className="whitespace-pre text-sm">
                    <code>
                      {`// Define dimensions
const int b_x = 2, b_y = 3, b_z = 4;  // gridDim: 2×3×4 = 24 blocks
const int t_x = 4, t_y = 4, t_z = 4;  // blockDim: 4×4×4 = 64 threads

// Launch kernel
dim3 blocksPerGrid(b_x, b_y, b_z);   // <<<grid>>>
dim3 threadsPerBlock(t_x, t_y, t_z); // <<<block>>>
whoami<<<blocksPerGrid, threadsPerBlock>>>();`}
                    </code>
                  </pre>
                </div>
                <div className="space-y-2 text-sm text-gray-700">
                  <p>
                    <strong>dim3</strong>: CUDA type for 3D dimensions
                    (automatically fills missing dimensions with 1)
                  </p>
                  <p>
                    <strong>blocksPerGrid</strong>: Defines how many blocks in
                    each dimension of the grid
                  </p>
                  <p>
                    <strong>threadsPerBlock</strong>: Defines how many threads
                    in each dimension of each block
                  </p>
                  <p>
                    <strong>&lt;&lt;&lt;&gt;&gt;&gt; syntax</strong>: CUDA's
                    special syntax for kernel launches
                  </p>
                  <p>
                    <strong>Total Threads</strong>: 24 blocks × 64 threads/block
                    = 1,536 threads total!
                  </p>
                  <p>
                    <strong>Parallel Execution</strong>: All 1,536 threads run
                    the whoami() function simultaneously
                  </p>
                </div>
              </div>

              <Alert>
                <AlertDescription>
                  <strong>🎯 Key Takeaway:</strong> This code demonstrates the
                  fundamental CUDA concept of giving every thread a unique
                  identifier. Understanding this mapping is essential for
                  writing effective parallel algorithms, as it determines how
                  your problem data gets distributed among thousands of GPU
                  threads.
                </AlertDescription>
              </Alert>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="code" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>GPU Kernel Code</CardTitle>
              <CardDescription>
                The whoami kernel that calculates thread IDs
              </CardDescription>
            </CardHeader>
            <CardContent>
              <pre className="bg-gray-900 text-green-400 p-4 rounded-lg overflow-x-auto text-sm">
                <code>{cudaKernelCode}</code>
              </pre>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Host Code</CardTitle>
              <CardDescription>
                How to launch the kernel with your current configuration
              </CardDescription>
            </CardHeader>
            <CardContent>
              <pre className="bg-gray-900 text-green-400 p-4 rounded-lg overflow-x-auto text-sm">
                <code>{hostCode}</code>
              </pre>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="explanation" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Understanding CUDA 3D Indexing</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <h4 className="font-semibold mb-2 text-gray-800">
                  🏙️ The Apartment Complex Analogy
                </h4>
                <ul className="text-sm space-y-1 text-gray-700 ml-4">
                  <li>
                    • <strong>Grid</strong> = The entire city containing
                    multiple buildings
                  </li>
                  <li>
                    • <strong>Block</strong> = A single building/apartment
                    complex
                  </li>
                  <li>
                    • <strong>Thread</strong> = An individual person/resident
                  </li>
                  <li>
                    • <strong>Global ID</strong> = Social security number
                    (unique for everyone)
                  </li>
                </ul>
              </div>

              <div>
                <h4 className="font-semibold mb-2 text-gray-800">
                  📐 3D Coordinate System
                </h4>
                <ul className="text-sm space-y-1 text-gray-700 ml-4">
                  <li>
                    • <strong>X-dimension</strong>: Apartments on each floor
                    (left to right)
                  </li>
                  <li>
                    • <strong>Y-dimension</strong>: Floors in each building
                    (bottom to top)
                  </li>
                  <li>
                    • <strong>Z-dimension</strong>: Buildings in the city (front
                    to back)
                  </li>
                </ul>
              </div>

              <div>
                <h4 className="font-semibold mb-2 text-gray-800">
                  🔢 Linear ID Calculation
                </h4>
                <div className="text-sm space-y-2 text-gray-700 ml-4">
                  <p>
                    <strong>Block ID Formula:</strong>{" "}
                    <code>z×(width×height) + y×width + x</code>
                  </p>
                  <p>
                    <strong>Thread ID Formula:</strong> Same formula applied
                    within each block
                  </p>
                  <p>
                    <strong>Global ID:</strong> Block starting position + thread
                    position within block
                  </p>
                </div>
              </div>

              <div>
                <h4 className="font-semibold mb-2 text-gray-800">
                  💡 Why This Matters
                </h4>
                <ul className="text-sm space-y-1 text-gray-700 ml-4">
                  <li>• Every thread needs a unique way to identify itself</li>
                  <li>• 3D coordinates are intuitive for many problems</li>
                  <li>• Linear IDs are needed for memory access</li>
                  <li>
                    • Understanding this mapping is crucial for CUDA programming
                  </li>
                </ul>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default CudaIndexingTutorial;
