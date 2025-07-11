import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";

import { 
  Grid3X3, 
  Zap, 
  Code, 
  PlayCircle, 
  ArrowRight,
  MemoryStick,
  Cpu,
  Layers,
  Play,
  Pause,
  RotateCcw,
  StepForward
} from "lucide-react";

interface ThreadConfig {
  blockDimX: number;
  blockDimY: number;
  gridDimX: number;
  gridDimY: number;
}

interface MatrixElement {
  value: number;
  row: number;
  col: number;
  memoryIndex: number;
  isActive: boolean;
  threadId?: { x: number; y: number };
  blockId?: { x: number; y: number };
}

function create1DArray(size: number): number[] {
  return Array(size).fill(0).map((_, i) => i + 1);
}

function arrayTo2DMatrix(array: number[], rows: number, cols: number): MatrixElement[] {
  return array.map((value, index) => ({
    value,
    row: Math.floor(index / cols),
    col: index % cols,
    memoryIndex: index,
    isActive: false
  }));
}

function getThreadMapping(matrixSize: number, blockDim: { x: number; y: number }) {
  const gridDim = {
    x: Math.ceil(matrixSize / blockDim.x),
    y: Math.ceil(matrixSize / blockDim.y)
  };
  
  const threads: Array<{
    blockId: { x: number; y: number };
    threadId: { x: number; y: number };
    globalRow: number;
    globalCol: number;
    memoryIndex: number;
  }> = [];
  for (let by = 0; by < gridDim.y; by++) {
    for (let bx = 0; bx < gridDim.x; bx++) {
      for (let ty = 0; ty < blockDim.y; ty++) {
        for (let tx = 0; tx < blockDim.x; tx++) {
          const row = by * blockDim.y + ty;
          const col = bx * blockDim.x + tx;
          
          if (row < matrixSize && col < matrixSize) {
            threads.push({
              blockId: { x: bx, y: by },
              threadId: { x: tx, y: ty },
              globalRow: row,
              globalCol: col,
              memoryIndex: row * matrixSize + col
            });
          }
        }
      }
    }
  }
  
  return { threads, gridDim };
}

function MemoryLayoutVisualization({ 
  array, 
  matrixSize, 
  highlightIndex = -1,
  title 
}: { 
  array: number[]; 
  matrixSize: number;
  highlightIndex?: number;
  title: string;
}) {
  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-blue-400">{title}</h3>
      
      {/* 1D Memory Layout */}
      <div className="space-y-2">
        <h4 className="text-sm font-medium text-slate-300">1D Memory Layout (Row-major)</h4>
        <div className="flex flex-wrap gap-1">
          {array.map((value, index) => {
            const isHighlighted = index === highlightIndex;
            return (
              <div 
                key={index}
                className={`
                  w-8 h-8 flex items-center justify-center text-xs font-mono rounded border
                  ${isHighlighted 
                    ? 'bg-yellow-500 text-black border-yellow-400' 
                    : 'bg-slate-700 text-slate-300 border-slate-600'
                  }
                  transition-all duration-300
                `}
              >
                {value}
              </div>
            );
          })}
        </div>
        <div className="text-xs text-slate-400 font-mono">
          Memory indices: [0, 1, 2, ..., {array.length - 1}]
        </div>
      </div>
      
      <ArrowRight className="mx-auto text-slate-400" size={20} />
      
      {/* 2D Matrix View */}
      <div className="space-y-2">
        <h4 className="text-sm font-medium text-slate-300">2D Matrix View ({matrixSize}×{matrixSize})</h4>
        <div 
          className="grid gap-1 mx-auto w-fit"
          style={{ gridTemplateColumns: `repeat(${matrixSize}, 1fr)` }}
        >
          {array.map((value, index) => {
            const row = Math.floor(index / matrixSize);
            const col = index % matrixSize;
            const isHighlighted = index === highlightIndex;
            
            return (
              <div 
                key={index}
                className={`
                  w-10 h-10 flex items-center justify-center text-xs font-mono rounded border
                  ${isHighlighted 
                    ? 'bg-yellow-500 text-black border-yellow-400' 
                    : 'bg-slate-700 text-slate-300 border-slate-600'
                  }
                  transition-all duration-300 cursor-pointer hover:bg-slate-600
                `}
                onClick={() => console.log(`Clicked: [${row}][${col}] = memory[${index}]`)}
              >
                {value}
              </div>
            );
          })}
        </div>
        <div className="text-xs text-slate-400 text-center">
          Formula: memory_index = row × {matrixSize} + col
        </div>
      </div>
    </div>
  );
}

function ThreadGridVisualization({ 
  matrixSize, 
  blockDim, 
  activeThread = null,
  showAllThreads = false 
}: { 
  matrixSize: number;
  blockDim: { x: number; y: number };
  activeThread?: { blockId: { x: number; y: number }; threadId: { x: number; y: number } } | null;
  showAllThreads?: boolean;
}) {
  const { threads, gridDim } = getThreadMapping(matrixSize, blockDim);
  
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-purple-400">CUDA Thread Organization</h3>
        <div className="text-sm text-slate-400">
          Grid: {gridDim.x}×{gridDim.y} blocks, Block: {blockDim.x}×{blockDim.y} threads
        </div>
      </div>
      
      {/* Grid of Blocks */}
      <div className="space-y-2">
        <h4 className="text-sm font-medium text-slate-300">Grid of Blocks</h4>
        <div 
          className="grid gap-2 mx-auto w-fit"
          style={{ gridTemplateColumns: `repeat(${gridDim.x}, 1fr)` }}
        >
          {Array(gridDim.y).fill(0).map((_, by) =>
            Array(gridDim.x).fill(0).map((_, bx) => {
              const isActiveBlock = activeThread && 
                activeThread.blockId.x === bx && activeThread.blockId.y === by;
              
              return (
                <div 
                  key={`block-${bx}-${by}`}
                  className={`
                    w-20 h-20 border-2 rounded-lg flex flex-col items-center justify-center
                    ${isActiveBlock 
                      ? 'border-purple-400 bg-purple-900/50' 
                      : 'border-slate-600 bg-slate-800/50'
                    }
                    transition-all duration-300
                  `}
                >
                  <div className="text-xs font-mono text-slate-300">Block</div>
                  <div className="text-sm font-bold text-purple-400">({bx},{by})</div>
                  
                  {/* Threads within block */}
                  <div 
                    className="grid gap-0.5 mt-1"
                    style={{ gridTemplateColumns: `repeat(${blockDim.x}, 1fr)` }}
                  >
                    {Array(blockDim.y).fill(0).map((_, ty) =>
                      Array(blockDim.x).fill(0).map((_, tx) => {
                        const isActiveThread = activeThread &&
                          activeThread.blockId.x === bx && activeThread.blockId.y === by &&
                          activeThread.threadId.x === tx && activeThread.threadId.y === ty;
                        
                        return (
                          <div 
                            key={`thread-${tx}-${ty}`}
                            className={`
                              w-1.5 h-1.5 rounded-sm
                              ${isActiveThread 
                                ? 'bg-yellow-400' 
                                : showAllThreads 
                                  ? 'bg-blue-400' 
                                  : 'bg-slate-600'
                              }
                            `}
                          />
                        );
                      })
                    )}
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>
      
      {/* Thread Details */}
      {activeThread && (
        <div className="bg-slate-800/50 border border-slate-600 rounded-lg p-4">
          <h4 className="text-sm font-medium text-slate-300 mb-2">Active Thread Details</h4>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <div className="text-slate-400">Block ID:</div>
              <div className="font-mono text-purple-400">({activeThread.blockId.x}, {activeThread.blockId.y})</div>
            </div>
            <div>
              <div className="text-slate-400">Thread ID:</div>
              <div className="font-mono text-yellow-400">({activeThread.threadId.x}, {activeThread.threadId.y})</div>
            </div>
            <div>
              <div className="text-slate-400">Global Row:</div>
              <div className="font-mono text-green-400">{activeThread.blockId.y * blockDim.y + activeThread.threadId.y}</div>
            </div>
            <div>
              <div className="text-slate-400">Global Col:</div>
              <div className="font-mono text-green-400">{activeThread.blockId.x * blockDim.x + activeThread.threadId.x}</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function IndexCalculationDemo({ 
  matrixSize, 
  blockDim,
  activeElement = null 
}: { 
  matrixSize: number;
  blockDim: { x: number; y: number };
  activeElement?: { row: number; col: number } | null;
}) {
  if (!activeElement) return null;
  
  const { row, col } = activeElement;
  const blockId = {
    x: Math.floor(col / blockDim.x),
    y: Math.floor(row / blockDim.y)
  };
  const threadId = {
    x: col % blockDim.x,
    y: row % blockDim.y
  };
  const memoryIndex = row * matrixSize + col;
  
  return (
    <Card className="bg-slate-800/50 border-slate-600">
      <CardHeader>
        <CardTitle className="text-lg text-green-400">Index Calculation for Element [{row}][{col}]</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-6">
          <div className="space-y-3">
            <h4 className="font-semibold text-slate-300">CUDA Thread Mapping</h4>
            <div className="space-y-2 text-sm font-mono">
              <div className="flex justify-between">
                <span className="text-slate-400">blockIdx.x:</span>
                <span className="text-purple-400">{col} ÷ {blockDim.x} = {blockId.x}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">blockIdx.y:</span>
                <span className="text-purple-400">{row} ÷ {blockDim.y} = {blockId.y}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">threadIdx.x:</span>
                <span className="text-yellow-400">{col} % {blockDim.x} = {threadId.x}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">threadIdx.y:</span>
                <span className="text-yellow-400">{row} % {blockDim.y} = {threadId.y}</span>
              </div>
            </div>
          </div>
          
          <div className="space-y-3">
            <h4 className="font-semibold text-slate-300">Global Coordinates</h4>
            <div className="space-y-2 text-sm font-mono">
              <div className="flex justify-between">
                <span className="text-slate-400">row:</span>
                <span className="text-green-400">{blockId.y} × {blockDim.y} + {threadId.y} = {row}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">col:</span>
                <span className="text-green-400">{blockId.x} × {blockDim.x} + {threadId.x} = {col}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">memory_idx:</span>
                <span className="text-blue-400">{row} × {matrixSize} + {col} = {memoryIndex}</span>
              </div>
            </div>
          </div>
        </div>
        
        <Separator className="bg-slate-600" />
        
        <div className="bg-slate-900/50 p-3 rounded">
          <h4 className="text-sm font-semibold text-slate-300 mb-2">CUDA Kernel Code</h4>
          <pre className="text-xs font-mono text-slate-300">
{`int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
int idx = row * N + col;

if (row < N && col < N) {
    // This thread computes C[row][col]
    float sum = 0.0f;
    for (int k = 0; k < N; k++) {
        sum += A[row * N + k] * B[k * N + col];
    }
    C[idx] = sum;
}`}
          </pre>
        </div>
      </CardContent>
    </Card>
  );
}

function StepByStepComputation({ 
  matrixSize,
  targetRow,
  targetCol 
}: { 
  matrixSize: number;
  targetRow: number;
  targetCol: number;
}) {
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [runningSum, setRunningSum] = useState(0);
  
  // Create sample matrices for visualization
  const matrixA = Array(matrixSize).fill(0).map((_, i) => 
    Array(matrixSize).fill(0).map((_, j) => i + j + 1)
  );
  const matrixB = Array(matrixSize).fill(0).map((_, i) => 
    Array(matrixSize).fill(0).map((_, j) => (i + 1) * (j + 1))
  );
  
  const maxSteps = matrixSize;
  
  useEffect(() => {
    if (isPlaying && currentStep < maxSteps) {
      const timer = setTimeout(() => {
        const aValue = matrixA[targetRow][currentStep];
        const bValue = matrixB[currentStep][targetCol];
        setRunningSum(prev => prev + aValue * bValue);
        setCurrentStep(prev => prev + 1);
      }, 1500);
      return () => clearTimeout(timer);
    } else if (isPlaying && currentStep >= maxSteps) {
      setIsPlaying(false);
    }
  }, [isPlaying, currentStep, maxSteps, matrixA, matrixB, targetRow, targetCol]);
  
  const reset = () => {
    setCurrentStep(0);
    setRunningSum(0);
    setIsPlaying(false);
  };
  
  const stepForward = () => {
    if (currentStep < maxSteps) {
      const aValue = matrixA[targetRow][currentStep];
      const bValue = matrixB[currentStep][targetCol];
      setRunningSum(prev => prev + aValue * bValue);
      setCurrentStep(prev => prev + 1);
    }
  };
  
  const togglePlay = () => {
    setIsPlaying(!isPlaying);
  };
  
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="text-xl font-semibold text-orange-400">
          Computing C[{targetRow}][{targetCol}] = A[{targetRow}][*] · B[*][{targetCol}]
        </h3>
        <div className="flex items-center gap-2">
          <Button 
            size="sm" 
            onClick={togglePlay}
            disabled={currentStep >= maxSteps}
            className="bg-green-600 hover:bg-green-700"
          >
            {isPlaying ? <Pause size={16} /> : <Play size={16} />}
            {isPlaying ? "Pause" : "Play"}
          </Button>
          <Button 
            size="sm" 
            onClick={stepForward}
            disabled={isPlaying || currentStep >= maxSteps}
            variant="outline"
            className="border-slate-600 text-slate-300 hover:bg-slate-800"
          >
            <StepForward size={16} />
            Step
          </Button>
          <Button 
            size="sm" 
            onClick={reset}
            variant="outline"
            className="border-slate-600 text-slate-300 hover:bg-slate-800"
          >
            <RotateCcw size={16} />
            Reset
          </Button>
        </div>
      </div>
      
      <div className="grid lg:grid-cols-4 gap-6">
        {/* Matrix A */}
        <div className="space-y-2">
          <h4 className="text-lg font-medium text-blue-400">Matrix A</h4>
          <div 
            className="grid gap-1 mx-auto w-fit"
            style={{ gridTemplateColumns: `repeat(${matrixSize}, 1fr)` }}
          >
            {matrixA.flat().map((value, index) => {
              const row = Math.floor(index / matrixSize);
              const col = index % matrixSize;
              const isTargetRow = row === targetRow;
              const isCurrentElement = row === targetRow && col === currentStep;
              const isPastElement = row === targetRow && col < currentStep;
              
              return (
                <div 
                  key={index}
                  className={`
                    w-8 h-8 flex items-center justify-center text-xs font-mono rounded border
                    ${isCurrentElement 
                      ? 'bg-yellow-500 text-black border-yellow-400 animate-pulse' 
                      : isPastElement
                        ? 'bg-green-600/50 text-green-200 border-green-500'
                        : isTargetRow 
                          ? 'bg-blue-600/30 text-blue-200 border-blue-500' 
                          : 'bg-slate-700 text-slate-300 border-slate-600'
                    }
                    transition-all duration-300
                  `}
                >
                  {value}
                </div>
              );
            })}
          </div>
          <div className="text-center text-sm text-slate-400">
            Row {targetRow} highlighted
          </div>
        </div>
        
        {/* Matrix B */}
        <div className="space-y-2">
          <h4 className="text-lg font-medium text-purple-400">Matrix B</h4>
          <div 
            className="grid gap-1 mx-auto w-fit"
            style={{ gridTemplateColumns: `repeat(${matrixSize}, 1fr)` }}
          >
            {matrixB.flat().map((value, index) => {
              const row = Math.floor(index / matrixSize);
              const col = index % matrixSize;
              const isTargetCol = col === targetCol;
              const isCurrentElement = row === currentStep && col === targetCol;
              const isPastElement = row < currentStep && col === targetCol;
              
              return (
                <div 
                  key={index}
                  className={`
                    w-8 h-8 flex items-center justify-center text-xs font-mono rounded border
                    ${isCurrentElement 
                      ? 'bg-yellow-500 text-black border-yellow-400 animate-pulse' 
                      : isPastElement
                        ? 'bg-green-600/50 text-green-200 border-green-500'
                        : isTargetCol 
                          ? 'bg-purple-600/30 text-purple-200 border-purple-500' 
                          : 'bg-slate-700 text-slate-300 border-slate-600'
                    }
                    transition-all duration-300
                  `}
                >
                  {value}
                </div>
              );
            })}
          </div>
          <div className="text-center text-sm text-slate-400">
            Column {targetCol} highlighted
          </div>
        </div>
        
        {/* Computation */}
        <div className="space-y-4">
          <h4 className="text-lg font-medium text-green-400">Dot Product Computation</h4>
          
          <div className="bg-slate-800/50 border border-slate-600 rounded-lg p-4">
            <div className="space-y-2">
              <div className="text-sm text-slate-300">Current Step: {currentStep} / {maxSteps}</div>
              <div className="text-2xl font-mono text-green-400">Sum = {runningSum}</div>
              
              {currentStep > 0 && (
                <div className="space-y-1 text-sm">
                  <div className="text-slate-400">Latest calculation:</div>
                  <div className="font-mono text-yellow-300">
                    {matrixA[targetRow][currentStep - 1]} × {matrixB[currentStep - 1][targetCol]} = {matrixA[targetRow][currentStep - 1] * matrixB[currentStep - 1][targetCol]}
                  </div>
                </div>
              )}
              
              {currentStep < maxSteps && (
                <div className="space-y-1 text-sm">
                  <div className="text-slate-400">Next calculation:</div>
                  <div className="font-mono text-blue-300">
                    {matrixA[targetRow][currentStep]} × {matrixB[currentStep][targetCol]} = ?
                  </div>
                </div>
              )}
            </div>
          </div>
          
          <div className="bg-slate-900/50 p-3 rounded">
            <h5 className="text-sm font-semibold text-slate-300 mb-2">Formula</h5>
            <div className="text-xs font-mono text-slate-300">
              C[{targetRow}][{targetCol}] = Σ(k=0 to {matrixSize-1}) A[{targetRow}][k] × B[k][{targetCol}]
            </div>
          </div>
          
          {currentStep >= maxSteps && (
            <div className="bg-green-600/20 border border-green-600/40 rounded-lg p-3">
              <div className="text-sm text-green-300">
                <strong>Complete!</strong> Thread computed C[{targetRow}][{targetCol}] = {runningSum}
              </div>
            </div>
          )}
        </div>
        
        {/* CUDA Kernel Code */}
        <div className="space-y-4">
          <h4 className="text-lg font-medium text-cyan-400">CUDA Kernel Execution</h4>
          
          <div className="bg-slate-900/50 border border-slate-600 rounded-lg p-4">
            <div className="space-y-1 text-xs font-mono">
              <div className={`transition-colors duration-300 ${
                currentStep === 0 ? 'bg-yellow-500/20 text-yellow-300' : 'text-slate-400'
              }`}>
                <span className="text-slate-500">1</span> int row = blockIdx.y * blockDim.y + threadIdx.y;
              </div>
              <div className={`transition-colors duration-300 ${
                currentStep === 0 ? 'bg-yellow-500/20 text-yellow-300' : 'text-slate-400'
              }`}>
                <span className="text-slate-500">2</span> int col = blockIdx.x * blockDim.x + threadIdx.x;
              </div>
              <div className="text-slate-400">
                <span className="text-slate-500">3</span> 
              </div>
              <div className={`transition-colors duration-300 ${
                currentStep === 0 ? 'bg-yellow-500/20 text-yellow-300' : 'text-slate-400'
              }`}>
                <span className="text-slate-500">4</span> if (row &lt; N && col &lt; N) &#123;
              </div>
              <div className={`transition-colors duration-300 ${
                currentStep === 0 ? 'bg-yellow-500/20 text-yellow-300' : 'text-slate-400'
              }`}>
                <span className="text-slate-500">5</span> &nbsp;&nbsp;float sum = 0.0f;
              </div>
              <div className="text-slate-400">
                <span className="text-slate-500">6</span> 
              </div>
              <div className={`transition-colors duration-300 ${
                currentStep > 0 && currentStep <= maxSteps ? 'bg-blue-500/20 text-blue-300' : 'text-slate-400'
              }`}>
                <span className="text-slate-500">7</span> &nbsp;&nbsp;for (int k = 0; k &lt; N; k++) &#123;
              </div>
              <div className={`transition-colors duration-300 ${
                currentStep > 0 && currentStep <= maxSteps ? 'bg-green-500/20 text-green-300' : 'text-slate-400'
              }`}>
                <span className="text-slate-500">8</span> &nbsp;&nbsp;&nbsp;&nbsp;sum += A[row * N + k] * B[k * N + col];
              </div>
              <div className={`transition-colors duration-300 ${
                currentStep > 0 && currentStep <= maxSteps ? 'bg-blue-500/20 text-blue-300' : 'text-slate-400'
              }`}>
                <span className="text-slate-500">9</span> &nbsp;&nbsp;&#125;
              </div>
              <div className="text-slate-400">
                <span className="text-slate-500">10</span> 
              </div>
              <div className={`transition-colors duration-300 ${
                currentStep >= maxSteps ? 'bg-purple-500/20 text-purple-300' : 'text-slate-400'
              }`}>
                <span className="text-slate-500">11</span> &nbsp;&nbsp;C[row * N + col] = sum;
              </div>
              <div className="text-slate-400">
                <span className="text-slate-500">12</span> &#125;
              </div>
            </div>
            
            <div className="mt-4 pt-3 border-t border-slate-700 text-xs text-slate-400">
              {currentStep === 0 && "Initializing thread variables..."}
              {currentStep > 0 && currentStep <= maxSteps && `Loop iteration ${currentStep}/${maxSteps}: k = ${currentStep - 1}`}
              {currentStep >= maxSteps && "Writing final result to global memory"}
            </div>
          </div>
        </div>
      </div>
      
      <div className="bg-slate-800/50 border border-slate-600 rounded-lg p-4">
        <h5 className="text-sm font-semibold text-slate-300 mb-3">Step-by-Step Breakdown</h5>
        <div className="space-y-2 max-h-40 overflow-y-auto">
          {Array.from({ length: Math.max(1, currentStep) }, (_, i) => {
            if (i >= currentStep && currentStep === 0) {
              return (
                <div key={i} className="text-sm text-slate-400">
                  Click Play or Step to begin computation...
                </div>
              );
            }
            
            if (i >= currentStep) return null;
            
            const aValue = matrixA[targetRow][i];
            const bValue = matrixB[i][targetCol];
            const product = aValue * bValue;
            const partialSum = matrixA[targetRow].slice(0, i + 1).reduce((sum, val, idx) => 
              sum + val * matrixB[idx][targetCol], 0
            );
            
            return (
              <div key={i} className="text-sm font-mono flex justify-between items-center">
                <span className="text-slate-300">
                  Step {i + 1}: A[{targetRow}][{i}] × B[{i}][{targetCol}] = {aValue} × {bValue} = {product}
                </span>
                <span className="text-green-400">Sum = {partialSum}</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

export default function CudaMatrixMultiplication() {
  const [matrixSize, setMatrixSize] = useState([4]);
  const [blockDimX, setBlockDimX] = useState([2]);
  const [blockDimY, setBlockDimY] = useState([2]);
  const [activeTab, setActiveTab] = useState("step-by-step");
  const [selectedElement, setSelectedElement] = useState<{ row: number; col: number } | null>({ row: 1, col: 2 });
  const [highlightedMemoryIndex, setHighlightedMemoryIndex] = useState(-1);
  const [activeThread, setActiveThread] = useState<{ blockId: { x: number; y: number }; threadId: { x: number; y: number } } | null>(null);
  const [showAllThreads, setShowAllThreads] = useState(false);
  
  const size = matrixSize[0];
  const blockDim = { x: blockDimX[0], y: blockDimY[0] };
  const matrixArray = create1DArray(size * size);
  
  useEffect(() => {
    if (selectedElement) {
      const memIndex = selectedElement.row * size + selectedElement.col;
      setHighlightedMemoryIndex(memIndex);
      
      // Calculate thread mapping for selected element
      const blockId = {
        x: Math.floor(selectedElement.col / blockDim.x),
        y: Math.floor(selectedElement.row / blockDim.y)
      };
      const threadId = {
        x: selectedElement.col % blockDim.x,
        y: selectedElement.row % blockDim.y
      };
      
      setActiveThread({ blockId, threadId });
    } else {
      setHighlightedMemoryIndex(-1);
      setActiveThread(null);
    }
  }, [selectedElement, size, blockDim]);
  
  const handleElementClick = (row: number, col: number) => {
    setSelectedElement(selectedElement?.row === row && selectedElement?.col === col ? null : { row, col });
  };
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-white">
      {/* Hero Section */}
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-blue-600/20 via-purple-600/20 to-teal-600/20" />
        <div className="absolute inset-0 opacity-20">
          <div className="w-full h-full bg-gradient-to-br from-transparent via-white/5 to-transparent" />
        </div>
        
        <div className="relative z-10 px-6 py-16 text-center">
          <Badge className="mb-6 bg-blue-600/20 text-blue-300 border-blue-600/30">
            <Cpu size={12} className="mr-2" />
            CUDA Programming Tutorial
          </Badge>
          
          <h1 className="text-3xl md:text-5xl lg:text-6xl font-bold mb-6 bg-gradient-to-r from-blue-400 via-purple-400 to-teal-400 bg-clip-text text-transparent">
            Visual CUDA
            <br />
            Matrix Multiplication
          </h1>
          
          <p className="text-lg text-slate-300 max-w-3xl mx-auto mb-8 leading-relaxed">
            Learn CUDA programming concepts through interactive visualizations and step-by-step animations. 
            Watch how individual threads compute matrix elements in real-time.
          </p>
        </div>
      </div>
      
      {/* Controls */}
      <div className="container mx-auto px-6 py-8">
        <Card className="bg-slate-800/50 border-slate-600">
          <CardContent className="pt-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="space-y-2">
                <label className="text-sm font-medium text-slate-300">Matrix Size: {size}×{size}</label>
                <Slider
                  value={matrixSize}
                  onValueChange={setMatrixSize}
                  max={8}
                  min={2}
                  step={1}
                  className="w-full"
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-slate-300">Block Width: {blockDim.x}</label>
                <Slider
                  value={blockDimX}
                  onValueChange={setBlockDimX}
                  max={4}
                  min={1}
                  step={1}
                  className="w-full"
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-slate-300">Block Height: {blockDim.y}</label>
                <Slider
                  value={blockDimY}
                  onValueChange={setBlockDimY}
                  max={4}
                  min={1}
                  step={1}
                  className="w-full"
                />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
      
      {/* Main Content */}
      <div className="container mx-auto px-6 pb-12">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-8">
          <TabsList className="grid w-full grid-cols-5 bg-slate-800/50 border border-slate-700">
            <TabsTrigger value="memory-layout" className="data-[state=active]:bg-blue-600">
              <MemoryStick className="mr-2" size={16} />
              Memory Layout
            </TabsTrigger>
            <TabsTrigger value="thread-mapping" className="data-[state=active]:bg-blue-600">
              <Grid3X3 className="mr-2" size={16} />
              Thread Mapping
            </TabsTrigger>
            <TabsTrigger value="step-by-step" className="data-[state=active]:bg-blue-600">
              <PlayCircle className="mr-2" size={16} />
              Step-by-Step
            </TabsTrigger>
            <TabsTrigger value="index-calc" className="data-[state=active]:bg-blue-600">
              <Code className="mr-2" size={16} />
              Index Calculation
            </TabsTrigger>
            <TabsTrigger value="kernel-code" className="data-[state=active]:bg-blue-600">
              <Layers className="mr-2" size={16} />
              CUDA Kernel
            </TabsTrigger>
          </TabsList>
          
          <TabsContent value="memory-layout" className="space-y-8">
            <div className="grid lg:grid-cols-2 gap-8">
              <MemoryLayoutVisualization 
                array={matrixArray}
                matrixSize={size}
                highlightIndex={highlightedMemoryIndex}
                title="Matrix A Storage"
              />
              
              <div className="space-y-4">
                <h3 className="text-lg font-semibold text-green-400">Interactive Matrix</h3>
                <p className="text-sm text-slate-300">
                  Click on any element to see its memory layout and thread mapping
                </p>
                
                <div 
                  className="grid gap-1 mx-auto w-fit"
                  style={{ gridTemplateColumns: `repeat(${size}, 1fr)` }}
                >
                  {Array(size).fill(0).map((_, row) =>
                    Array(size).fill(0).map((_, col) => {
                      const isSelected = selectedElement?.row === row && selectedElement?.col === col;
                      const memIndex = row * size + col;
                      
                      return (
                        <div 
                          key={`${row}-${col}`}
                          className={`
                            w-12 h-12 flex flex-col items-center justify-center text-xs font-mono rounded border cursor-pointer
                            ${isSelected 
                              ? 'bg-yellow-500 text-black border-yellow-400' 
                              : 'bg-slate-700 text-slate-300 border-slate-600 hover:bg-slate-600'
                            }
                            transition-all duration-200
                          `}
                          onClick={() => handleElementClick(row, col)}
                        >
                          <div className="text-xs">[{row}][{col}]</div>
                          <div className="text-xs text-slate-400">{memIndex}</div>
                        </div>
                      );
                    })
                  )}
                </div>
                
                {selectedElement && (
                  <div className="bg-slate-800/50 border border-slate-600 rounded-lg p-4">
                    <h4 className="text-sm font-medium text-slate-300 mb-2">Selected Element</h4>
                    <div className="text-sm space-y-1">
                      <div>Position: [{selectedElement.row}][{selectedElement.col}]</div>
                      <div>Memory Index: {selectedElement.row * size + selectedElement.col}</div>
                      <div className="text-xs text-slate-400 mt-2">
                        Formula: memory_index = {selectedElement.row} × {size} + {selectedElement.col} = {selectedElement.row * size + selectedElement.col}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </TabsContent>
          
          <TabsContent value="thread-mapping" className="space-y-8">
            <div className="grid lg:grid-cols-2 gap-8">
              <ThreadGridVisualization 
                matrixSize={size}
                blockDim={blockDim}
                activeThread={activeThread}
                showAllThreads={showAllThreads}
              />
              
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-green-400">Matrix Element Mapping</h3>
                  <Button 
                    variant="outline" 
                    size="sm"
                    onClick={() => setShowAllThreads(!showAllThreads)}
                    className="border-slate-600 text-slate-300 hover:bg-slate-800"
                  >
                    {showAllThreads ? 'Hide' : 'Show'} All Threads
                  </Button>
                </div>
                
                <div 
                  className="grid gap-1 mx-auto w-fit"
                  style={{ gridTemplateColumns: `repeat(${size}, 1fr)` }}
                >
                  {Array(size).fill(0).map((_, row) =>
                    Array(size).fill(0).map((_, col) => {
                      const isSelected = selectedElement?.row === row && selectedElement?.col === col;
                      
                      // Color coding by block
                      const blockX = Math.floor(col / blockDim.x);
                      const blockY = Math.floor(row / blockDim.y);
                      const blockIndex = blockY * Math.ceil(size / blockDim.x) + blockX;
                      
                      const colors = [
                        'bg-red-600/30 border-red-500',
                        'bg-blue-600/30 border-blue-500', 
                        'bg-green-600/30 border-green-500',
                        'bg-yellow-600/30 border-yellow-500',
                        'bg-purple-600/30 border-purple-500',
                        'bg-pink-600/30 border-pink-500',
                        'bg-indigo-600/30 border-indigo-500',
                        'bg-teal-600/30 border-teal-500'
                      ];
                      
                      const colorClass = colors[blockIndex % colors.length];
                      
                      return (
                        <div 
                          key={`${row}-${col}`}
                          className={`
                            w-10 h-10 flex items-center justify-center text-xs font-mono rounded border cursor-pointer
                            ${isSelected 
                              ? 'bg-yellow-500 text-black border-yellow-400' 
                              : colorClass
                            }
                            transition-all duration-200 hover:opacity-80
                          `}
                          onClick={() => handleElementClick(row, col)}
                        >
                          {row},{col}
                        </div>
                      );
                    })
                  )}
                </div>
                
                <div className="text-xs text-slate-400 text-center">
                  Elements are colored by block assignment. Click to see thread details.
                </div>
              </div>
            </div>
          </TabsContent>
          
          <TabsContent value="step-by-step" className="space-y-8">
            <Card className="bg-slate-800/50 border-slate-600">
              <CardHeader>
                <CardTitle className="text-orange-400">Animated Thread Computation</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-slate-300 mb-4">
                  Watch how a single CUDA thread computes one matrix element by performing 
                  the dot product of a row from matrix A and a column from matrix B.
                </p>
                
                {selectedElement ? (
                  <StepByStepComputation 
                    matrixSize={size}
                    targetRow={selectedElement.row}
                    targetCol={selectedElement.col}
                  />
                ) : (
                  <div className="text-center py-8">
                    <p className="text-slate-400 mb-4">
                      Select an element from the Memory Layout or Thread Mapping tabs to see the step-by-step computation.
                    </p>
                    <Button 
                      variant="outline" 
                      className="border-slate-600 text-slate-300 hover:bg-slate-800"
                      onClick={() => setSelectedElement({ row: 1, col: 1 })}
                    >
                      Try Example: Element [1][1]
                    </Button>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="index-calc" className="space-y-8">
            <IndexCalculationDemo 
              matrixSize={size}
              blockDim={blockDim}
              activeElement={selectedElement}
            />
            
            {!selectedElement && (
              <Card className="bg-slate-800/50 border-slate-600">
                <CardContent className="pt-6 text-center">
                  <p className="text-slate-400">
                    Select an element from the Memory Layout or Thread Mapping tabs to see the index calculations.
                  </p>
                  <Button 
                    variant="outline" 
                    className="mt-4 border-slate-600 text-slate-300 hover:bg-slate-800"
                    onClick={() => setSelectedElement({ row: 1, col: 2 })}
                  >
                    Try Example: Element [1][2]
                  </Button>
                </CardContent>
              </Card>
            )}
          </TabsContent>
          
          <TabsContent value="kernel-code" className="space-y-6">
            <Card className="bg-slate-800/50 border-slate-600">
              <CardHeader>
                <CardTitle className="text-blue-400">Complete CUDA Matrix Multiplication Kernel</CardTitle>
              </CardHeader>
              <CardContent>
                <pre className="text-sm bg-slate-900/50 p-4 rounded overflow-x-auto text-slate-300">
                  <code>{`#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_SIZE 16

// Naive kernel - each thread computes one element
__global__ void matrixMulNaive(float* A, float* B, float* C, int N) {
    // Calculate global thread coordinates
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check
    if (row < N && col < N) {
        float sum = 0.0f;
        
        // Compute dot product of row and column
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        
        // Store result
        C[row * N + col] = sum;
    }
}

// Optimized kernel with shared memory tiling
__global__ void matrixMulTiled(float* A, float* B, float* C, int N) {
    // Shared memory tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    // Block and thread indices
    int bx = blockIdx.x;    int by = blockIdx.y;
    int tx = threadIdx.x;   int ty = threadIdx.y;
    
    // Global coordinates for this thread
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile into shared memory
        if (row < N && t * TILE_SIZE + tx < N)
            As[ty][tx] = A[row * N + t * TILE_SIZE + tx];
        else
            As[ty][tx] = 0.0f;
            
        if (col < N && t * TILE_SIZE + ty < N)
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;
            
        // Synchronize to ensure tiles are loaded
        __syncthreads();
        
        // Compute partial sum using shared memory
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        // Synchronize before loading next tile
        __syncthreads();
    }
    
    // Store final result
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// Host function
void matrixMultiplyCUDA(float* h_A, float* h_B, float* h_C, int N) {
    size_t size = N * N * sizeof(float);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    // Copy data to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // Configure kernel launch parameters
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, 
                   (N + TILE_SIZE - 1) / TILE_SIZE);
    
    // Launch kernel
    matrixMulTiled<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    
    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}`}</code>
                </pre>
              </CardContent>
            </Card>
            
            <div className="grid md:grid-cols-2 gap-6">
              <Card className="bg-slate-800/50 border-slate-600">
                <CardHeader>
                  <CardTitle className="text-purple-400">Key CUDA Concepts</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4 text-slate-300">
                  <div className="space-y-3">
                    <div className="border-l-2 border-blue-500 pl-4">
                      <h4 className="font-semibold text-white">Thread Hierarchy</h4>
                      <p className="text-sm">Grid → Blocks → Threads</p>
                      <p className="text-xs text-slate-400">blockIdx, threadIdx coordinates</p>
                    </div>
                    <div className="border-l-2 border-purple-500 pl-4">
                      <h4 className="font-semibold text-white">Memory Types</h4>
                      <p className="text-sm">Global, Shared, Local memory</p>
                      <p className="text-xs text-slate-400">Different access speeds and scope</p>
                    </div>
                    <div className="border-l-2 border-teal-500 pl-4">
                      <h4 className="font-semibold text-white">Synchronization</h4>
                      <p className="text-sm">__syncthreads() barrier</p>
                      <p className="text-xs text-slate-400">Coordinate shared memory access</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
              
              <Card className="bg-slate-800/50 border-slate-600">
                <CardHeader>
                  <CardTitle className="text-yellow-400">Optimization Strategies</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4 text-slate-300">
                  <div className="space-y-3">
                    <div className="border-l-2 border-green-500 pl-4">
                      <h4 className="font-semibold text-white">Memory Coalescing</h4>
                      <p className="text-sm">Adjacent threads access consecutive memory</p>
                    </div>
                    <div className="border-l-2 border-orange-500 pl-4">
                      <h4 className="font-semibold text-white">Shared Memory Tiling</h4>
                      <p className="text-sm">Reduce global memory accesses</p>
                    </div>
                    <div className="border-l-2 border-red-500 pl-4">
                      <h4 className="font-semibold text-white">Occupancy</h4>
                      <p className="text-sm">Maximize active warps per SM</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </div>
      
      {/* Footer */}
      <footer className="border-t border-slate-800 bg-slate-900/50 py-8">
        <div className="container mx-auto px-6 text-center text-slate-400">
          <p className="mb-2">Interactive CUDA Learning Platform</p>
          <p className="text-sm">Learn more at <span className="text-blue-400">developer.nvidia.com/cuda</span></p>
        </div>
      </footer>
    </div>
  );
}
