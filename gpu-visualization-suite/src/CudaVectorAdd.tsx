import { useState, useEffect } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Slider } from "@/components/ui/slider";
import { Play, Pause, RotateCcw, Code, Zap, Grid3X3 } from "lucide-react";

export default function CudaVectorAdd() {
  const [vectorSize, setVectorSize] = useState(12);
  const [threadsPerBlock, setThreadsPerBlock] = useState(4);
  const [animationStep, setAnimationStep] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);
  const [selectedThread, setSelectedThread] = useState<number | null>(null);

  // Calculate blocks needed
  const blocksNeeded = Math.ceil(vectorSize / threadsPerBlock);
  const totalThreads = blocksNeeded * threadsPerBlock;

  // Sample vectors for visualization
  const vectorA = Array.from({ length: vectorSize }, (_, i) =>
    Math.round((i + 1) * 1.5)
  );
  const vectorB = Array.from({ length: vectorSize }, (_, i) =>
    Math.round((i + 1) * 0.8)
  );
  const vectorC = vectorA.map((a, i) => a + vectorB[i]);

  useEffect(() => {
    if (isAnimating) {
      const interval = setInterval(() => {
        setAnimationStep((prev) => {
          if (prev >= vectorSize - 1) {
            setIsAnimating(false);
            return prev;
          }
          return prev + 1;
        });
      }, 1000);
      return () => clearInterval(interval);
    }
  }, [isAnimating, vectorSize]);

  const resetAnimation = () => {
    setAnimationStep(0);
    setIsAnimating(false);
  };

  const toggleAnimation = () => {
    if (animationStep >= vectorSize - 1) {
      resetAnimation();
    } else {
      setIsAnimating(!isAnimating);
    }
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-green-600 to-blue-600 bg-clip-text text-transparent">
          CUDA Vector Addition
        </h1>
        <p className="text-xl text-muted-foreground">
          Learn how CUDA threads work together to add vectors in parallel
        </p>
        <div className="mt-4">
          <a
            href="https://leetgpu.com/challenges/vector-addition"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
          >
            <Code className="w-4 h-4" />
            Practice on LeetGPU
          </a>
        </div>
      </div>

      <Tabs defaultValue="concept" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="concept">Concept</TabsTrigger>
          <TabsTrigger value="visualization">Visualization</TabsTrigger>
          <TabsTrigger value="code">CUDA Code</TabsTrigger>
          <TabsTrigger value="optimization">Optimization</TabsTrigger>
        </TabsList>

        <TabsContent value="concept" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Zap className="w-5 h-5 text-yellow-500" />
                Vector Addition Fundamentals
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <h3 className="text-lg font-semibold mb-3">
                    Sequential vs Parallel
                  </h3>
                  <div className="space-y-3">
                    <div className="p-4 bg-red-50 dark:bg-red-950/20 rounded-lg border border-red-200 dark:border-red-800">
                      <h4 className="font-medium text-red-800 dark:text-red-200">
                        CPU (Sequential)
                      </h4>
                      <p className="text-sm text-red-700 dark:text-red-300">
                        Processes one element at a time:
                        <br />
                        C[0] = A[0] + B[0]
                        <br />
                        C[1] = A[1] + B[1]
                        <br />
                        C[2] = A[2] + B[2]
                        <br />
                        ...
                      </p>
                    </div>
                    <div className="p-4 bg-green-50 dark:bg-green-950/20 rounded-lg border border-green-200 dark:border-green-800">
                      <h4 className="font-medium text-green-800 dark:text-green-200">
                        GPU (Parallel)
                      </h4>
                      <p className="text-sm text-green-700 dark:text-green-300">
                        All elements computed simultaneously:
                        <br />
                        Thread 0: C[0] = A[0] + B[0]
                        <br />
                        Thread 1: C[1] = A[1] + B[1]
                        <br />
                        Thread 2: C[2] = A[2] + B[2]
                        <br />
                        ...
                      </p>
                    </div>
                  </div>
                </div>

                <div>
                  <h3 className="text-lg font-semibold mb-3">
                    Thread Organization
                  </h3>
                  <div className="space-y-3">
                    <div className="p-4 bg-blue-50 dark:bg-blue-950/20 rounded-lg">
                      <h4 className="font-medium mb-2">1D Thread Layout</h4>
                      <p className="text-sm text-muted-foreground">
                        Each thread has a unique ID calculated as:
                      </p>
                      <code className="block mt-2 p-2 bg-background rounded text-sm font-mono">
                        thread_id = blockIdx.x * blockDim.x + threadIdx.x
                      </code>
                    </div>
                    <div className="p-4 bg-purple-50 dark:bg-purple-950/20 rounded-lg">
                      <h4 className="font-medium mb-2">Bounds Checking</h4>
                      <p className="text-sm text-muted-foreground">
                        Essential for cases where total threads &gt; vector
                        size:
                      </p>
                      <code className="block mt-2 p-2 bg-background rounded text-sm font-mono">
                        if (i &lt; N) &#123;
                        <br />
                        &nbsp;&nbsp;C[i] = A[i] + B[i];
                        <br />
                        &#125;
                      </code>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="visualization" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Interactive Vector Addition Visualization</CardTitle>
              <CardDescription>
                Watch how CUDA threads map to vector elements and compute
                results in parallel
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div>
                    <label className="text-sm font-medium mb-2 block">
                      Vector Size: {vectorSize}
                    </label>
                    <Slider
                      value={[vectorSize]}
                      onValueChange={(value) => {
                        setVectorSize(value[0]);
                        resetAnimation();
                      }}
                      max={16}
                      min={4}
                      step={1}
                      className="w-full"
                    />
                  </div>
                  <div>
                    <label className="text-sm font-medium mb-2 block">
                      Threads per Block: {threadsPerBlock}
                    </label>
                    <Slider
                      value={[threadsPerBlock]}
                      onValueChange={(value) => {
                        setThreadsPerBlock(value[0]);
                        resetAnimation();
                      }}
                      max={8}
                      min={2}
                      step={1}
                      className="w-full"
                    />
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    onClick={toggleAnimation}
                    className="flex items-center gap-2"
                  >
                    {isAnimating ? (
                      <Pause className="w-4 h-4" />
                    ) : (
                      <Play className="w-4 h-4" />
                    )}
                    {isAnimating ? "Pause" : "Play"}
                  </Button>
                  <Button
                    onClick={resetAnimation}
                    variant="outline"
                    className="flex items-center gap-2"
                  >
                    <RotateCcw className="w-4 h-4" />
                    Reset
                  </Button>
                </div>
              </div>

              <div className="space-y-4">
                <div className="p-4 bg-muted rounded-lg">
                  <h4 className="font-medium mb-2">Configuration</h4>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      Vector Size: <Badge variant="outline">{vectorSize}</Badge>
                    </div>
                    <div>
                      Threads per Block:{" "}
                      <Badge variant="outline">{threadsPerBlock}</Badge>
                    </div>
                    <div>
                      Blocks Needed:{" "}
                      <Badge variant="outline">{blocksNeeded}</Badge>
                    </div>
                    <div>
                      Total Threads:{" "}
                      <Badge variant="outline">{totalThreads}</Badge>
                    </div>
                  </div>
                </div>

                {/* Thread Block Visualization */}
                <div className="space-y-3">
                  <h4 className="font-medium">Thread Organization</h4>
                  <div className="space-y-2">
                    {Array.from({ length: blocksNeeded }, (_, blockIdx) => (
                      <div key={blockIdx} className="flex items-center gap-2">
                        <div className="w-16 text-sm font-mono">
                          Block {blockIdx}:
                        </div>
                        <div className="flex gap-1">
                          {Array.from(
                            { length: threadsPerBlock },
                            (_, threadIdx) => {
                              const globalThreadId =
                                blockIdx * threadsPerBlock + threadIdx;
                              const isActive = globalThreadId < vectorSize;
                              const isComputed =
                                globalThreadId <= animationStep;

                              return (
                                <div
                                  key={threadIdx}
                                  className={`w-8 h-8 rounded text-xs flex items-center justify-center font-mono cursor-pointer transition-colors ${
                                    !isActive
                                      ? "bg-gray-200 text-gray-400 dark:bg-gray-700"
                                      : isComputed
                                      ? "bg-green-500 text-white"
                                      : "bg-blue-500 text-white"
                                  }`}
                                  onClick={() =>
                                    setSelectedThread(globalThreadId)
                                  }
                                >
                                  {globalThreadId}
                                </div>
                              );
                            }
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Vector Visualization */}
                <div className="space-y-3">
                  <h4 className="font-medium">Vector Computation</h4>
                  <div className="space-y-2">
                    <div className="flex items-center gap-2">
                      <div className="w-8 text-sm font-medium">A:</div>
                      <div className="flex gap-1">
                        {vectorA.map((val, idx) => (
                          <div
                            key={idx}
                            className={`w-10 h-8 rounded text-xs flex items-center justify-center font-mono ${
                              idx <= animationStep
                                ? "bg-blue-100 dark:bg-blue-900"
                                : "bg-gray-100 dark:bg-gray-800"
                            }`}
                          >
                            {val}
                          </div>
                        ))}
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-8 text-sm font-medium">B:</div>
                      <div className="flex gap-1">
                        {vectorB.map((val, idx) => (
                          <div
                            key={idx}
                            className={`w-10 h-8 rounded text-xs flex items-center justify-center font-mono ${
                              idx <= animationStep
                                ? "bg-orange-100 dark:bg-orange-900"
                                : "bg-gray-100 dark:bg-gray-800"
                            }`}
                          >
                            {val}
                          </div>
                        ))}
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-8 text-sm font-medium">C:</div>
                      <div className="flex gap-1">
                        {vectorC.map((val, idx) => (
                          <div
                            key={idx}
                            className={`w-10 h-8 rounded text-xs flex items-center justify-center font-mono ${
                              idx <= animationStep
                                ? "bg-green-100 dark:bg-green-900 font-bold"
                                : "bg-gray-100 dark:bg-gray-800"
                            }`}
                          >
                            {idx <= animationStep ? val : "?"}
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>

                {selectedThread !== null && selectedThread < vectorSize && (
                  <div className="p-4 bg-blue-50 dark:bg-blue-950/20 rounded-lg border border-blue-200 dark:border-blue-800">
                    <h4 className="font-medium mb-2">
                      Thread {selectedThread} Details
                    </h4>
                    <div className="space-y-1 text-sm font-mono">
                      <div>
                        blockIdx.x ={" "}
                        {Math.floor(selectedThread / threadsPerBlock)}
                      </div>
                      <div>
                        threadIdx.x = {selectedThread % threadsPerBlock}
                      </div>
                      <div>blockDim.x = {threadsPerBlock}</div>
                      <div className="pt-2 border-t">
                        <div>
                          thread_id ={" "}
                          {Math.floor(selectedThread / threadsPerBlock)} ×{" "}
                          {threadsPerBlock} + {selectedThread % threadsPerBlock}{" "}
                          = {selectedThread}
                        </div>
                        <div className="pt-1">
                          <div>
                            C[{selectedThread}] = A[{selectedThread}] + B[
                            {selectedThread}]
                          </div>
                          <div>
                            C[{selectedThread}] = {vectorA[selectedThread]} +{" "}
                            {vectorB[selectedThread]} ={" "}
                            {vectorC[selectedThread]}
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="code" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Code className="w-5 h-5 text-green-500" />
                CUDA Vector Addition Implementation
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-4">
                <div>
                  <h3 className="text-lg font-semibold mb-3">
                    Kernel Function
                  </h3>
                  <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
                    <code>{`__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    // Calculate this thread's unique ID
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Bounds check - important when total threads > vector size
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}`}</code>
                  </pre>
                </div>

                <div>
                  <h3 className="text-lg font-semibold mb-3">Host Function</h3>
                  <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
                    <code>{`void solve(const float* A, const float* B, float* C, int N) {
    // Define thread block size
    int threadsPerBlock = 256;
    
    // Calculate number of blocks needed
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch kernel
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    
    // Wait for all threads to complete
    cudaDeviceSynchronize();
}`}</code>
                  </pre>
                </div>
              </div>

              <div className="grid md:grid-cols-2 gap-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Key Concepts</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="p-3 bg-blue-50 dark:bg-blue-950/20 rounded">
                      <h4 className="font-medium text-blue-800 dark:text-blue-200">
                        Thread ID Calculation
                      </h4>
                      <p className="text-sm text-blue-700 dark:text-blue-300">
                        Each thread computes its unique global ID to determine
                        which vector element to process
                      </p>
                    </div>
                    <div className="p-3 bg-yellow-50 dark:bg-yellow-950/20 rounded">
                      <h4 className="font-medium text-yellow-800 dark:text-yellow-200">
                        Bounds Checking
                      </h4>
                      <p className="text-sm text-yellow-700 dark:text-yellow-300">
                        Essential when total threads may exceed vector size
                      </p>
                    </div>
                    <div className="p-3 bg-green-50 dark:bg-green-950/20 rounded">
                      <h4 className="font-medium text-green-800 dark:text-green-200">
                        Memory Coalescing
                      </h4>
                      <p className="text-sm text-green-700 dark:text-green-300">
                        Consecutive threads access consecutive memory locations
                        for optimal performance
                      </p>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Performance Notes</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="p-3 bg-purple-50 dark:bg-purple-950/20 rounded">
                      <h4 className="font-medium text-purple-800 dark:text-purple-200">
                        Block Size
                      </h4>
                      <p className="text-sm text-purple-700 dark:text-purple-300">
                        Typically 256 or 512 threads per block for optimal
                        occupancy
                      </p>
                    </div>
                    <div className="p-3 bg-red-50 dark:bg-red-950/20 rounded">
                      <h4 className="font-medium text-red-800 dark:text-red-200">
                        Memory Bandwidth
                      </h4>
                      <p className="text-sm text-red-700 dark:text-red-300">
                        Vector operations are typically memory-bound, not
                        compute-bound
                      </p>
                    </div>
                    <div className="p-3 bg-indigo-50 dark:bg-indigo-950/20 rounded">
                      <h4 className="font-medium text-indigo-800 dark:text-indigo-200">
                        Synchronization
                      </h4>
                      <p className="text-sm text-indigo-700 dark:text-indigo-300">
                        cudaDeviceSynchronize() ensures all threads complete
                        before continuing
                      </p>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="optimization" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Grid3X3 className="w-5 h-5 text-purple-500" />
                Optimization Strategies
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold">Memory Optimization</h3>
                  <div className="space-y-3">
                    <div className="p-4 bg-green-50 dark:bg-green-950/20 rounded-lg">
                      <h4 className="font-medium text-green-800 dark:text-green-200">
                        Coalesced Access
                      </h4>
                      <p className="text-sm text-green-700 dark:text-green-300">
                        Adjacent threads access adjacent memory locations for
                        maximum bandwidth utilization
                      </p>
                    </div>
                    <div className="p-4 bg-blue-50 dark:bg-blue-950/20 rounded-lg">
                      <h4 className="font-medium text-blue-800 dark:text-blue-200">
                        Memory Alignment
                      </h4>
                      <p className="text-sm text-blue-700 dark:text-blue-300">
                        Ensure vectors are aligned to 128-byte boundaries for
                        optimal performance
                      </p>
                    </div>
                  </div>
                </div>

                <div className="space-y-4">
                  <h3 className="text-lg font-semibold">
                    Execution Configuration
                  </h3>
                  <div className="space-y-3">
                    <div className="p-4 bg-purple-50 dark:bg-purple-950/20 rounded-lg">
                      <h4 className="font-medium text-purple-800 dark:text-purple-200">
                        Block Size Selection
                      </h4>
                      <p className="text-sm text-purple-700 dark:text-purple-300">
                        Choose block sizes that are multiples of 32 (warp size)
                        for optimal occupancy
                      </p>
                    </div>
                    <div className="p-4 bg-orange-50 dark:bg-orange-950/20 rounded-lg">
                      <h4 className="font-medium text-orange-800 dark:text-orange-200">
                        Grid Size Calculation
                      </h4>
                      <p className="text-sm text-orange-700 dark:text-orange-300">
                        Use ceiling division to handle vector sizes that aren't
                        multiples of block size
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
                <h3 className="text-lg font-semibold mb-3">
                  Optimized Implementation
                </h3>
                <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
                  <code>{`// Optimized version with better memory access patterns
__global__ void vector_add_optimized(const float4* A, const float4* B, float4* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < N) {
        float4 a = A[i];
        float4 b = B[i];
        float4 c;
        
        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;
        
        C[i] = c;
    }
}

// Process 4 elements per thread for better memory bandwidth utilization
void solve_optimized(const float* A, const float* B, float* C, int N) {
    int elements_per_thread = 4;
    int vector_size = (N + elements_per_thread - 1) / elements_per_thread;
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (vector_size + threadsPerBlock - 1) / threadsPerBlock;
    
    vector_add_optimized<<<blocksPerGrid, threadsPerBlock>>>(
        (float4*)A, (float4*)B, (float4*)C, vector_size
    );
    
    cudaDeviceSynchronize();
}`}</code>
                </pre>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
