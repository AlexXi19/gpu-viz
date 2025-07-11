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

interface MatrixCell {
  row: number;
  col: number;
  value: number;
  threadId: number;
  blockId: number;
  highlighted: boolean;
}

interface CudaConfig {
  matrixSize: number;
  blockSize: number;
  gridSize: number;
}

const CudaMatrixMultiply: React.FC = () => {
  const [config, setConfig] = useState<CudaConfig>(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("cuda-matrix-config");
      if (saved) {
        try {
          return JSON.parse(saved);
        } catch {
          // Fall back to default if parsing fails
        }
      }
    }
    return {
      matrixSize: 4,
      blockSize: 2,
      gridSize: 2,
    };
  });

  const [currentStep, setCurrentStep] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);
  const [highlightedCell, setHighlightedCell] = useState<{
    row: number;
    col: number;
  } | null>(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("cuda-matrix-highlighted-cell");
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
    blockId: number;
    threadId: number;
  } | null>(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("cuda-matrix-selected-thread");
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
  const [selectedA, setSelectedA] = useState<{
    row: number;
    col: number;
  } | null>(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("cuda-matrix-selected-a");
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
  const [selectedB, setSelectedB] = useState<{
    row: number;
    col: number;
  } | null>(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("cuda-matrix-selected-b");
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

  // Update grid size when matrix size or block size changes
  useEffect(() => {
    const newGridSize = Math.ceil(config.matrixSize / config.blockSize);
    setConfig((prev) => ({ ...prev, gridSize: newGridSize }));
  }, [config.matrixSize, config.blockSize]);

  // Generate matrices with sample values
  const generateMatrix = (
    size: number,
    isIdentity: boolean = false,
    isB: boolean = false
  ) => {
    const matrix: number[][] = [];

    if (size === 4 && !isIdentity) {
      // Fixed values for 4x4 demonstration
      if (isB) {
        // Matrix B - different pattern for clearer demonstration
        return [
          [2, 0, 1, 0],
          [0, 2, 0, 1],
          [1, 0, 2, 0],
          [0, 1, 0, 2],
        ];
      } else {
        // Matrix A - simple pattern
        return [
          [1, 2, 3, 4],
          [5, 6, 7, 8],
          [9, 10, 11, 12],
          [13, 14, 15, 16],
        ];
      }
    }

    // For other sizes or identity matrix, use original logic
    for (let i = 0; i < size; i++) {
      matrix[i] = [];
      for (let j = 0; j < size; j++) {
        if (isIdentity) {
          matrix[i][j] = i === j ? 1 : 0;
        } else {
          matrix[i][j] = Math.floor(Math.random() * 10) + 1;
        }
      }
    }
    return matrix;
  };

  const [matrixA] = useState(generateMatrix(config.matrixSize, false, false));
  const [matrixB] = useState(generateMatrix(config.matrixSize, false, true));
  const [matrixC, setMatrixC] = useState(
    generateMatrix(config.matrixSize, true, false)
  );

  // Generate transposed version of Matrix B for better memory access
  const [showTranspose, setShowTranspose] = useState(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("cuda-matrix-show-transpose");
      return saved === "true";
    }
    return false;
  });

  // Save state to localStorage when it changes
  useEffect(() => {
    if (typeof window !== "undefined") {
      localStorage.setItem("cuda-matrix-config", JSON.stringify(config));
    }
  }, [config]);

  useEffect(() => {
    if (typeof window !== "undefined") {
      localStorage.setItem(
        "cuda-matrix-highlighted-cell",
        JSON.stringify(highlightedCell)
      );
    }
  }, [highlightedCell]);

  useEffect(() => {
    if (typeof window !== "undefined") {
      localStorage.setItem(
        "cuda-matrix-selected-thread",
        JSON.stringify(selectedThread)
      );
    }
  }, [selectedThread]);

  useEffect(() => {
    if (typeof window !== "undefined") {
      localStorage.setItem("cuda-matrix-selected-a", JSON.stringify(selectedA));
    }
  }, [selectedA]);

  useEffect(() => {
    if (typeof window !== "undefined") {
      localStorage.setItem("cuda-matrix-selected-b", JSON.stringify(selectedB));
    }
  }, [selectedB]);

  useEffect(() => {
    if (typeof window !== "undefined") {
      localStorage.setItem("cuda-matrix-show-transpose", String(showTranspose));
    }
  }, [showTranspose]);
  const matrixBT = matrixB[0].map((_, colIndex) =>
    matrixB.map((row) => row[colIndex])
  );

  // Calculate thread and block mapping
  const getThreadMapping = (row: number, col: number) => {
    const blockRow = Math.floor(row / config.blockSize);
    const blockCol = Math.floor(col / config.blockSize);
    const blockId = blockRow * config.gridSize + blockCol;

    const threadRow = row % config.blockSize;
    const threadCol = col % config.blockSize;
    const threadId = threadRow * config.blockSize + threadCol;

    return { blockId, threadId, blockRow, blockCol, threadRow, threadCol };
  };

  // CUDA kernel code
  const cudaKernelCode = showTranspose
    ? `__global__ void matrixMulTransposed(float *A, float *BT, float *C, int N) {
    // Calculate thread's position in the grid
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check
    if (row < N && col < N) {
        float sum = 0.0f;
        
        // Compute dot product using transposed B
        // Now both A and BT are accessed row-wise (stride-1 access)
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * BT[col * N + k];
        }
        
        // Store result
        C[row * N + col] = sum;
    }
}`
    : `__global__ void matrixMul(float *A, float *B, float *C, int N) {
    // Calculate thread's position in the grid
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check
    if (row < N && col < N) {
        float sum = 0.0f;
        
        // Compute dot product of row and column
        // A: stride-1 access (good), B: stride-N access (bad)
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        
        // Store result
        C[row * N + col] = sum;
    }
}`;

  const launchConfiguration = showTranspose
    ? `// Launch configuration
dim3 blockSize(${config.blockSize}, ${config.blockSize});
dim3 gridSize(${config.gridSize}, ${config.gridSize});

// Launch optimized kernel with transposed B
matrixMulTransposed<<<gridSize, blockSize>>>(d_A, d_BT, d_C, ${config.matrixSize});`
    : `// Launch configuration
dim3 blockSize(${config.blockSize}, ${config.blockSize});
dim3 gridSize(${config.gridSize}, ${config.gridSize});

// Launch standard kernel
matrixMul<<<gridSize, blockSize>>>(d_A, d_B, d_C, ${config.matrixSize});`;

  // Matrix visualization component
  const MatrixVisualization: React.FC<{
    matrix: number[][];
    title: string;
    matrixType: "A" | "B" | "C";
    showThreadInfo?: boolean;
  }> = ({ matrix, title, matrixType, showThreadInfo = false }) => {
    const getCellColor = (row: number, col: number) => {
      // Check for selected elements in A and B
      if (
        matrixType === "A" &&
        selectedA &&
        selectedA.row === row &&
        selectedA.col === col
      ) {
        return "bg-green-200 border-green-500";
      }
      if (
        matrixType === "B" &&
        selectedB &&
        selectedB.row === row &&
        selectedB.col === col
      ) {
        return "bg-purple-200 border-purple-500";
      }

      // Check for result element in C
      if (
        matrixType === "C" &&
        selectedA &&
        selectedB &&
        selectedA.col === selectedB.row
      ) {
        if (row === selectedA.row && col === selectedB.col) {
          return "bg-orange-200 border-orange-500";
        }
      }

      // Original highlighting logic
      if (
        highlightedCell &&
        highlightedCell.row === row &&
        highlightedCell.col === col
      ) {
        return "bg-yellow-200 border-yellow-500";
      }
      if (selectedThread) {
        const mapping = getThreadMapping(row, col);
        if (
          mapping.blockId === selectedThread.blockId &&
          mapping.threadId === selectedThread.threadId
        ) {
          return "bg-blue-200 border-blue-500";
        }
      }
      return "bg-gray-50 border-gray-300";
    };

    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle className="text-lg">{title}</CardTitle>
        </CardHeader>
        <CardContent>
          <div
            className="grid gap-1"
            style={{ gridTemplateColumns: `repeat(${config.matrixSize}, 1fr)` }}
          >
            {matrix.map((row, rowIndex) =>
              row.map((value, colIndex) => {
                const mapping = getThreadMapping(rowIndex, colIndex);
                return (
                  <div
                    key={`${rowIndex}-${colIndex}`}
                    className={`w-8 h-8 border-2 flex items-center justify-center text-xs font-mono cursor-pointer transition-colors font-semibold text-gray-800 ${getCellColor(
                      rowIndex,
                      colIndex
                    )}`}
                    onClick={() => {
                      if (matrixType === "A") {
                        setSelectedA({ row: rowIndex, col: colIndex });
                      } else if (matrixType === "B") {
                        setSelectedB({ row: rowIndex, col: colIndex });
                      } else {
                        setHighlightedCell({ row: rowIndex, col: colIndex });
                        setSelectedThread({
                          blockId: mapping.blockId,
                          threadId: mapping.threadId,
                        });
                      }
                    }}
                    title={
                      showThreadInfo
                        ? `Block ${mapping.blockId}, Thread ${mapping.threadId}`
                        : `[${rowIndex}][${colIndex}]`
                    }
                  >
                    {value}
                  </div>
                );
              })
            )}
          </div>
        </CardContent>
      </Card>
    );
  };

  // 1D Array visualization component
  const ArrayVisualization: React.FC<{
    matrix: number[][];
    title: string;
    matrixType: "A" | "B" | "C";
    showIndices?: boolean;
  }> = ({ matrix, title, matrixType, showIndices = false }) => {
    // Convert 2D matrix to 1D array (row-major order)
    const array1D = matrix.flat();

    const getCellColor = (index: number) => {
      const row = Math.floor(index / config.matrixSize);
      const col = index % config.matrixSize;

      // Check for selected elements in A and B
      if (
        matrixType === "A" &&
        selectedA &&
        selectedA.row === row &&
        selectedA.col === col
      ) {
        return "bg-green-200 border-green-500";
      }
      if (
        matrixType === "B" &&
        selectedB &&
        selectedB.row === row &&
        selectedB.col === col
      ) {
        return "bg-purple-200 border-purple-500";
      }

      // Check for result element in C
      if (
        matrixType === "C" &&
        selectedA &&
        selectedB &&
        selectedA.col === selectedB.row
      ) {
        if (row === selectedA.row && col === selectedB.col) {
          return "bg-orange-200 border-orange-500";
        }
      }

      // Original highlighting logic
      if (
        highlightedCell &&
        highlightedCell.row === row &&
        highlightedCell.col === col
      ) {
        return "bg-yellow-200 border-yellow-500";
      }
      if (selectedThread) {
        const mapping = getThreadMapping(row, col);
        if (
          mapping.blockId === selectedThread.blockId &&
          mapping.threadId === selectedThread.threadId
        ) {
          return "bg-blue-200 border-blue-500";
        }
      }
      return "bg-gray-50 border-gray-300";
    };

    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle className="text-lg">{title}</CardTitle>
          <CardDescription>1D Memory Layout (Row-Major Order)</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div
              className="grid gap-1"
              style={{
                gridTemplateColumns: `repeat(${
                  config.matrixSize * config.matrixSize
                }, 1fr)`,
              }}
            >
              {array1D.map((value, index) => {
                const row = Math.floor(index / config.matrixSize);
                const col = index % config.matrixSize;
                const mapping = getThreadMapping(row, col);

                return (
                  <div
                    key={index}
                    className={`w-8 h-8 border-2 flex items-center justify-center text-xs font-mono cursor-pointer transition-colors font-semibold text-gray-800 ${getCellColor(
                      index
                    )}`}
                    onClick={() => {
                      if (matrixType === "A") {
                        setSelectedA({ row, col });
                      } else if (matrixType === "B") {
                        setSelectedB({ row, col });
                      } else {
                        setHighlightedCell({ row, col });
                        setSelectedThread({
                          blockId: mapping.blockId,
                          threadId: mapping.threadId,
                        });
                      }
                    }}
                    title={`Index ${index} = [${row}][${col}]`}
                  >
                    {value}
                  </div>
                );
              })}
            </div>

            {showIndices && (
              <div className="text-xs text-gray-500 mt-2">
                <div className="mb-2 font-semibold">1D Array Indices:</div>
                <div
                  className="grid gap-1"
                  style={{
                    gridTemplateColumns: `repeat(${
                      config.matrixSize * config.matrixSize
                    }, 1fr)`,
                  }}
                >
                  {array1D.map((_, index) => (
                    <div
                      key={index}
                      className="w-8 h-6 flex items-center justify-center text-xs border border-gray-200 bg-gray-100"
                    >
                      {index}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    );
  };

  // Element Selection Explanation - shows how selected A and B elements contribute
  const ElementSelectionExplanation: React.FC = () => {
    if (!selectedA || !selectedB) {
      return (
        <Card>
          <CardHeader>
            <CardTitle>Element Selection</CardTitle>
            <CardDescription>
              Select one element from Matrix A and one from Matrix B to see how
              they contribute to the result
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <h4 className="font-semibold mb-2 text-gray-800">
                    Matrix A Selection:
                  </h4>
                  <div className="text-sm text-gray-600">
                    {selectedA
                      ? `A[${selectedA.row}][${selectedA.col}] = ${
                          matrixA[selectedA.row][selectedA.col]
                        }`
                      : "Click any element in Matrix A"}
                  </div>
                </div>
                <div>
                  <h4 className="font-semibold mb-2 text-gray-800">
                    Matrix B Selection:
                  </h4>
                  <div className="text-sm text-gray-600">
                    {selectedB
                      ? `B[${selectedB.row}][${selectedB.col}] = ${
                          matrixB[selectedB.row][selectedB.col]
                        }`
                      : "Click any element in Matrix B"}
                  </div>
                </div>
              </div>
              <Alert>
                <AlertDescription>
                  <strong>Green:</strong> Selected element from Matrix A<br />
                  <strong>Purple:</strong> Selected element from Matrix B<br />
                  <strong>Orange:</strong> Resulting element in Matrix C (when A
                  column matches B row)
                </AlertDescription>
              </Alert>
            </div>
          </CardContent>
        </Card>
      );
    }

    const aValue = matrixA[selectedA.row][selectedA.col];
    const bValue = matrixB[selectedB.row][selectedB.col];
    const aIndex = selectedA.row * config.matrixSize + selectedA.col;
    const bIndex = selectedB.row * config.matrixSize + selectedB.col;

    // Check if these elements can multiply together in matrix multiplication
    const canMultiply = selectedA.col === selectedB.row;
    const resultRow = selectedA.row;
    const resultCol = selectedB.col;
    const resultIndex = resultRow * config.matrixSize + resultCol;

    return (
      <Card>
        <CardHeader>
          <CardTitle>
            Selected Elements: A[{selectedA.row}][{selectedA.col}] × B[
            {selectedB.row}][{selectedB.col}]
          </CardTitle>
          <CardDescription>
            How these specific elements contribute to matrix multiplication
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold mb-2 text-gray-800">
                Matrix A Element:
              </h4>
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 border-2 border-green-500 bg-green-100 flex items-center justify-center text-lg font-mono font-bold text-green-800">
                  {aValue}
                </div>
                <div className="text-sm text-gray-700">
                  <div>
                    Position: A[{selectedA.row}][{selectedA.col}]
                  </div>
                  <div>1D Index: {aIndex}</div>
                  <div>Memory: A[{aIndex}]</div>
                </div>
              </div>
            </div>

            <div>
              <h4 className="font-semibold mb-2 text-gray-800">
                Matrix B Element:
              </h4>
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 border-2 border-purple-500 bg-purple-100 flex items-center justify-center text-lg font-mono font-bold text-purple-800">
                  {bValue}
                </div>
                <div className="text-sm text-gray-700">
                  <div>
                    Position: B[{selectedB.row}][{selectedB.col}]
                  </div>
                  <div>1D Index: {bIndex}</div>
                  <div>Memory: B[{bIndex}]</div>
                </div>
              </div>
            </div>
          </div>

          <Separator />

          <div>
            <h4 className="font-semibold mb-2 text-gray-800">
              Contribution to Result:
            </h4>
            <div className="bg-gray-50 p-4 rounded-lg">
              {canMultiply ? (
                <div className="space-y-2">
                  <div className="text-sm font-mono text-gray-800">
                    A[{selectedA.row}][{selectedA.col}] × B[{selectedB.row}][
                    {selectedB.col}] = {aValue} × {bValue} = {aValue * bValue}
                  </div>
                  <div className="text-sm text-gray-700">
                    This contributes to C[{resultRow}][{resultCol}] (1D index{" "}
                    {resultIndex})
                  </div>
                  <div className="flex items-center gap-4 mt-4">
                    <div className="w-12 h-12 border-2 border-orange-500 bg-orange-100 flex items-center justify-center text-lg font-mono font-bold text-orange-800">
                      {aValue * bValue}
                    </div>
                    <div className="text-sm text-gray-700">
                      <div>
                        This is ONE term in the sum for C[{resultRow}][
                        {resultCol}]
                      </div>
                      <div>
                        The full calculation includes all A[{resultRow}][k] ×
                        B[k][{resultCol}] terms
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-sm text-orange-600">
                  <strong>Note:</strong> These elements cannot multiply together
                  in matrix multiplication!
                  <br />
                  For A[{selectedA.row}][{selectedA.col}] to multiply with B[
                  {selectedB.row}][{selectedB.col}], we need {selectedA.col} ={" "}
                  {selectedB.row}.<br />
                  Try selecting A[{selectedA.row}][{selectedB.row}] or B[
                  {selectedA.col}][{selectedB.col}] instead.
                </div>
              )}
            </div>
          </div>

          <Alert>
            <AlertDescription>
              <strong>Memory Layout:</strong> A[{selectedA.row}][{selectedA.col}
              ] is at 1D index {aIndex}, B[{selectedB.row}][{selectedB.col}] is
              at 1D index {bIndex}
              {canMultiply &&
                `, contributing to C[${resultRow}][${resultCol}] at 1D index ${resultIndex}`}
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    );
  };

  // Thread and Block visualization
  const ThreadBlockVisualization: React.FC = () => {
    const blocks: React.ReactElement[] = [];
    for (let blockRow = 0; blockRow < config.gridSize; blockRow++) {
      for (let blockCol = 0; blockCol < config.gridSize; blockCol++) {
        const blockId = blockRow * config.gridSize + blockCol;
        const isSelected = selectedThread && selectedThread.blockId === blockId;

        blocks.push(
          <div
            key={blockId}
            className={`border-2 p-2 rounded-lg ${
              isSelected ? "border-blue-500 bg-blue-50" : "border-gray-300"
            }`}
          >
            <div className="text-xs font-bold mb-1 text-gray-800">
              Block {blockId}
            </div>
            <div className="text-xs text-gray-600 mb-2">
              ({blockRow}, {blockCol})
            </div>
            <div
              className="grid gap-1"
              style={{
                gridTemplateColumns: `repeat(${config.blockSize}, 1fr)`,
              }}
            >
              {Array.from({ length: config.blockSize * config.blockSize }).map(
                (_, threadIndex) => {
                  const threadRow = Math.floor(threadIndex / config.blockSize);
                  const threadCol = threadIndex % config.blockSize;
                  const isThreadSelected =
                    selectedThread &&
                    selectedThread.blockId === blockId &&
                    selectedThread.threadId === threadIndex;

                  return (
                    <div
                      key={threadIndex}
                      className={`w-6 h-6 border text-xs flex items-center justify-center cursor-pointer font-semibold ${
                        isThreadSelected
                          ? "bg-blue-300 border-blue-500 text-blue-900"
                          : "bg-gray-100 border-gray-300 text-gray-700"
                      }`}
                      onClick={() =>
                        setSelectedThread({ blockId, threadId: threadIndex })
                      }
                      title={`Block ${blockId}, Thread ${threadIndex} (${threadRow}, ${threadCol})`}
                    >
                      {threadIndex}
                    </div>
                  );
                }
              )}
            </div>
          </div>
        );
      }
    }

    return (
      <Card>
        <CardHeader>
          <CardTitle>CUDA Grid Organization</CardTitle>
          <CardDescription>
            Grid: {config.gridSize}×{config.gridSize} blocks, Block:{" "}
            {config.blockSize}×{config.blockSize} threads
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div
            className="grid gap-4"
            style={{ gridTemplateColumns: `repeat(${config.gridSize}, 1fr)` }}
          >
            {blocks}
          </div>
        </CardContent>
      </Card>
    );
  };

  // Index calculation explanation
  const IndexCalculationExplanation: React.FC = () => {
    if (!highlightedCell) return null;

    const mapping = getThreadMapping(highlightedCell.row, highlightedCell.col);

    return (
      <Card>
        <CardHeader>
          <CardTitle>
            Index Calculation for Cell [{highlightedCell.row}][
            {highlightedCell.col}]
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <h4 className="font-semibold mb-2">2D to 1D Mapping:</h4>
              <div className="space-y-1 text-sm font-mono">
                <div>blockIdx.y = {mapping.blockRow}</div>
                <div>blockIdx.x = {mapping.blockCol}</div>
                <div>
                  blockId = {mapping.blockRow} × {config.gridSize} +{" "}
                  {mapping.blockCol} = {mapping.blockId}
                </div>
              </div>
            </div>
            <div>
              <h4 className="font-semibold mb-2">Thread Coordinates:</h4>
              <div className="space-y-1 text-sm font-mono">
                <div>threadIdx.y = {mapping.threadRow}</div>
                <div>threadIdx.x = {mapping.threadCol}</div>
                <div>
                  threadId = {mapping.threadRow} × {config.blockSize} +{" "}
                  {mapping.threadCol} = {mapping.threadId}
                </div>
              </div>
            </div>
          </div>

          <Separator />

          <div>
            <h4 className="font-semibold mb-2">Global Index Calculation:</h4>
            <div className="space-y-1 text-sm font-mono bg-gray-50 p-3 rounded">
              <div>row = blockIdx.y × blockDim.y + threadIdx.y</div>
              <div>
                row = {mapping.blockRow} × {config.blockSize} +{" "}
                {mapping.threadRow} = {highlightedCell.row}
              </div>
              <div className="mt-2">
                col = blockIdx.x × blockDim.x + threadIdx.x
              </div>
              <div>
                col = {mapping.blockCol} × {config.blockSize} +{" "}
                {mapping.threadCol} = {highlightedCell.col}
              </div>
            </div>
          </div>

          <Alert>
            <AlertDescription>
              Each thread processes exactly one element of the result matrix C[
              {highlightedCell.row}][{highlightedCell.col}]
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
            CUDA Matrix Multiplication Visualization
          </CardTitle>
          <CardDescription>
            Interactive visualization showing how CUDA threads and blocks map to
            matrix elements
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-4 mb-6">
            <div>
              <label className="block text-sm font-medium mb-2">
                Matrix Size: {config.matrixSize}×{config.matrixSize}
              </label>
              <Slider
                value={[config.matrixSize]}
                onValueChange={(value) =>
                  setConfig((prev) => ({ ...prev, matrixSize: value[0] }))
                }
                min={4}
                max={16}
                step={4}
                className="w-full"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">
                Block Size: {config.blockSize}×{config.blockSize}
              </label>
              <Slider
                value={[config.blockSize]}
                onValueChange={(value) =>
                  setConfig((prev) => ({ ...prev, blockSize: value[0] }))
                }
                min={2}
                max={8}
                step={2}
                className="w-full"
              />
            </div>
            <div className="flex items-end">
              <Badge variant="outline" className="text-sm">
                Grid: {config.gridSize}×{config.gridSize} blocks
              </Badge>
            </div>
          </div>
        </CardContent>
      </Card>

      <Tabs defaultValue="visualization" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="visualization">Matrix Visualization</TabsTrigger>
          <TabsTrigger value="grid">Grid Organization</TabsTrigger>
          <TabsTrigger value="code">CUDA Code</TabsTrigger>
          <TabsTrigger value="explanation">How It Works</TabsTrigger>
        </TabsList>

        <TabsContent value="visualization" className="space-y-6">
          <div className="mb-4">
            <div className="flex items-center gap-4">
              <label className="flex items-center gap-2 text-sm font-medium cursor-pointer">
                <input
                  type="checkbox"
                  checked={showTranspose}
                  onChange={(e) => setShowTranspose(e.target.checked)}
                  className="rounded"
                />
                Show Matrix B Transpose (B^T) for better memory access
              </label>
            </div>
          </div>

          {/* 2D Matrix View */}
          <Card>
            <CardHeader>
              <CardTitle>2D Matrix Representation</CardTitle>
              <CardDescription>
                Traditional matrix view - click elements to see their
                relationships
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-3 gap-4">
                <MatrixVisualization
                  matrix={matrixA}
                  title="Matrix A"
                  matrixType="A"
                  showThreadInfo
                />
                <MatrixVisualization
                  matrix={showTranspose ? matrixBT : matrixB}
                  title={showTranspose ? "Matrix B^T (Transposed)" : "Matrix B"}
                  matrixType="B"
                  showThreadInfo
                />
                <MatrixVisualization
                  matrix={matrixC}
                  title="Matrix C (Result)"
                  matrixType="C"
                  showThreadInfo
                />
              </div>
            </CardContent>
          </Card>

          {/* 1D Array View */}
          <Card>
            <CardHeader>
              <CardTitle>1D Memory Layout (How GPU Actually Sees It)</CardTitle>
              <CardDescription>
                Row-major memory layout - same data as above but showing actual
                memory organization
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                <ArrayVisualization
                  matrix={matrixA}
                  title="Matrix A"
                  matrixType="A"
                  showIndices
                />
                <ArrayVisualization
                  matrix={showTranspose ? matrixBT : matrixB}
                  title={showTranspose ? "Matrix B^T (Transposed)" : "Matrix B"}
                  matrixType="B"
                  showIndices
                />
                <ArrayVisualization
                  matrix={matrixC}
                  title="Matrix C (Result)"
                  matrixType="C"
                  showIndices
                />
              </div>

              <div className="mt-6 bg-gray-50 p-4 rounded-lg">
                <h4 className="font-semibold mb-2 text-gray-800">
                  Memory Layout Key Concepts:
                </h4>
                <div className="grid grid-cols-2 gap-4 text-sm text-gray-700">
                  <div>
                    <p>
                      • <strong>Row-major order:</strong> Elements stored row by
                      row
                    </p>
                    <p>
                      • <strong>Index formula:</strong> element[i][j] → index
                      i×N + j
                    </p>
                  </div>
                  <div>
                    <p>
                      • <strong>CUDA access:</strong> A[row×N + k], B[k×N + col]
                    </p>
                    <p>
                      • <strong>Cache efficiency:</strong> Adjacent elements =
                      better performance
                    </p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <ElementSelectionExplanation />

          {showTranspose && (
            <Card>
              <CardHeader>
                <CardTitle>Memory Access Pattern Comparison</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-6">
                    <div>
                      <h4 className="font-semibold mb-2 text-red-600">
                        ❌ Original Matrix B (Column Access)
                      </h4>
                      <div className="text-sm space-y-2">
                        <p>For C[i][j], we need column j from B:</p>
                        <p className="font-mono bg-gray-50 p-2 rounded">
                          B[0][j], B[1][j], B[2][j], B[3][j]
                        </p>
                        <p>
                          🚫 <strong>Problem:</strong> Memory indices are far
                          apart!
                        </p>
                        <p className="font-mono text-xs">
                          Index j → j+4 → j+8 → j+12 (stride of 4)
                        </p>
                      </div>
                    </div>
                    <div>
                      <h4 className="font-semibold mb-2 text-green-600">
                        ✅ Transposed Matrix B^T (Row Access)
                      </h4>
                      <div className="text-sm space-y-2">
                        <p>For C[i][j], we need row j from B^T:</p>
                        <p className="font-mono bg-gray-50 p-2 rounded">
                          B^T[j][0], B^T[j][1], B^T[j][2], B^T[j][3]
                        </p>
                        <p>
                          ✅ <strong>Benefit:</strong> Memory indices are
                          adjacent!
                        </p>
                        <p className="font-mono text-xs">
                          Index j*4 → j*4+1 → j*4+2 → j*4+3 (stride of 1)
                        </p>
                      </div>
                    </div>
                  </div>

                  <Alert>
                    <AlertDescription>
                      <strong>GPU Memory Optimization:</strong> Accessing
                      consecutive memory addresses (stride-1) is much faster
                      than scattered access (stride-4) due to memory coalescing
                      in GPU architectures.
                    </AlertDescription>
                  </Alert>
                </div>
              </CardContent>
            </Card>
          )}

          <Card>
            <CardHeader>
              <CardTitle>Instructions</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 text-sm">
                <p>
                  • <strong>Click any element in Matrix A or B</strong> to
                  select it (green/purple highlight)
                </p>
                <p>
                  • <strong>Orange highlight</strong> shows the resulting
                  element in Matrix C
                </p>
                <p>
                  • <strong>Compare 2D vs 1D views</strong> to understand memory
                  layout
                </p>
                <p>
                  • <strong>Toggle B^T</strong> to see the transposed version
                  for better memory access
                </p>
                <p>
                  • <strong>Yellow highlight</strong> shows thread mapping
                  (Matrix C only)
                </p>
                <p>
                  • Adjust matrix size and block size to see how the mapping
                  changes
                </p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="grid">
          <ThreadBlockVisualization />
          {selectedThread && (
            <Card className="mt-4">
              <CardHeader>
                <CardTitle>Selected Thread Information</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <h4 className="font-semibold mb-2">Block Information:</h4>
                    <div className="space-y-1 text-sm">
                      <div>Block ID: {selectedThread.blockId}</div>
                      <div>
                        Block Position: (
                        {Math.floor(selectedThread.blockId / config.gridSize)},{" "}
                        {selectedThread.blockId % config.gridSize})
                      </div>
                    </div>
                  </div>
                  <div>
                    <h4 className="font-semibold mb-2">Thread Information:</h4>
                    <div className="space-y-1 text-sm">
                      <div>Thread ID: {selectedThread.threadId}</div>
                      <div>
                        Thread Position: (
                        {Math.floor(selectedThread.threadId / config.blockSize)}
                        , {selectedThread.threadId % config.blockSize})
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="code">
          <div className="space-y-4">
            <div className="mb-4">
              <div className="flex items-center gap-4">
                <label className="flex items-center gap-2 text-sm font-medium cursor-pointer">
                  <input
                    type="checkbox"
                    checked={showTranspose}
                    onChange={(e) => setShowTranspose(e.target.checked)}
                    className="rounded"
                  />
                  Show optimized kernel with transposed B
                </label>
              </div>
            </div>

            <Card>
              <CardHeader>
                <CardTitle>CUDA Kernel Code</CardTitle>
                <CardDescription>
                  {showTranspose
                    ? "Optimized version using transposed B for better memory coalescing"
                    : "Standard version with potential memory access inefficiency"}
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
                <CardTitle>Kernel Launch Configuration</CardTitle>
              </CardHeader>
              <CardContent>
                <pre className="bg-gray-900 text-green-400 p-4 rounded-lg overflow-x-auto text-sm">
                  <code>{launchConfiguration}</code>
                </pre>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Memory Access Analysis</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-6">
                    <div>
                      <h4 className="font-semibold mb-2 text-red-600">
                        Standard Version Issues:
                      </h4>
                      <ul className="text-sm space-y-1">
                        <li>• A[row * N + k]: ✅ Stride-1 access (good)</li>
                        <li>• B[k * N + col]: ❌ Stride-N access (bad)</li>
                        <li>• Non-coalesced memory access for B</li>
                        <li>• Poor cache utilization</li>
                      </ul>
                    </div>
                    <div>
                      <h4 className="font-semibold mb-2 text-green-600">
                        Optimized Version Benefits:
                      </h4>
                      <ul className="text-sm space-y-1">
                        <li>• A[row * N + k]: ✅ Stride-1 access (good)</li>
                        <li>• BT[col * N + k]: ✅ Stride-1 access (good)</li>
                        <li>• Both matrices use coalesced access</li>
                        <li>• Better cache utilization</li>
                      </ul>
                    </div>
                  </div>

                  <Alert>
                    <AlertDescription>
                      <strong>Performance Impact:</strong> The optimized version
                      can be 2-4x faster on modern GPUs due to improved memory
                      coalescing, especially for larger matrices.
                    </AlertDescription>
                  </Alert>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="explanation">
          <div className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Understanding CUDA Matrix Multiplication</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <h4 className="font-semibold mb-2">
                    1. Thread-to-Element Mapping
                  </h4>
                  <p className="text-sm">
                    Each CUDA thread computes exactly one element of the result
                    matrix C. The thread's position in the grid determines which
                    element it processes.
                  </p>
                </div>

                <div>
                  <h4 className="font-semibold mb-2">
                    2. 2D Grid Organization
                  </h4>
                  <p className="text-sm">
                    The GPU organizes threads in a 2D grid of blocks, where each
                    block contains a 2D array of threads. This maps naturally to
                    matrix operations.
                  </p>
                </div>

                <div>
                  <h4 className="font-semibold mb-2">3. Index Calculation</h4>
                  <p className="text-sm">
                    Global indices are calculated as:{" "}
                    <code>row = blockIdx.y × blockDim.y + threadIdx.y</code> and{" "}
                    <code>col = blockIdx.x × blockDim.x + threadIdx.x</code>
                  </p>
                </div>

                <div>
                  <h4 className="font-semibold mb-2">
                    4. Memory Access Pattern
                  </h4>
                  <p className="text-sm">
                    Each thread reads a full row from matrix A and a full column
                    from matrix B to compute one element of matrix C.
                  </p>
                </div>

                <div>
                  <h4 className="font-semibold mb-2">5. Parallelism</h4>
                  <p className="text-sm">
                    All threads execute in parallel, making matrix
                    multiplication highly efficient on GPU architectures.
                  </p>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Key Concepts</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <h4 className="font-semibold mb-2">Grid Dimensions</h4>
                    <ul className="text-sm space-y-1">
                      <li>
                        • <code>gridDim.x</code>: Number of blocks in X
                        direction
                      </li>
                      <li>
                        • <code>gridDim.y</code>: Number of blocks in Y
                        direction
                      </li>
                      <li>
                        • <code>blockIdx.x/y</code>: Block's position in grid
                      </li>
                    </ul>
                  </div>
                  <div>
                    <h4 className="font-semibold mb-2">Block Dimensions</h4>
                    <ul className="text-sm space-y-1">
                      <li>
                        • <code>blockDim.x</code>: Number of threads in X
                        direction
                      </li>
                      <li>
                        • <code>blockDim.y</code>: Number of threads in Y
                        direction
                      </li>
                      <li>
                        • <code>threadIdx.x/y</code>: Thread's position in block
                      </li>
                    </ul>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default CudaMatrixMultiply;
