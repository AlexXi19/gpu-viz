import React, { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import {
  Cpu,
  Grid3X3,
  Layers,
  Zap,
  Box,
  Languages,
  ChevronRight,
  Play,
  ArrowRight,
  Code,
  MemoryStick,
  Workflow,
  TrendingUp,
} from "lucide-react";

type Language = "en" | "zh";

const content = {
  en: {
    title: "GPU Architecture Tutorial",
    subtitle: "Master the fundamentals of GPU programming",
    languageToggle: "中文",
    sections: {
      intro: "Introduction",
      kernels: "Kernels",
      threads: "Threads",
      blocks: "Blocks",
      grids: "Grids",
      warps: "Warps",
      memory: "Memory",
      performance: "Performance",
    },
    intro: {
      title: "GPU vs CPU Architecture",
      description:
        "Understanding the fundamental differences between GPU and CPU design",
      content: {
        cpuDesc:
          "CPU (Central Processing Unit): Optimized for sequential processing with complex control logic and large caches. Typically has 4-16 cores.",
        gpuDesc:
          "GPU (Graphics Processing Unit): Optimized for parallel processing with thousands of simpler cores. Designed for data-parallel workloads.",
        keyDiff:
          "Key Difference: CPUs excel at complex tasks with branching logic, while GPUs excel at simple operations on large amounts of data in parallel.",
      },
    },
    kernels: {
      title: "Kernels",
      description: "Your code that runs on the GPU",
      content: {
        definition:
          "A kernel is a function that runs on the GPU. It's written once but executed by thousands of threads simultaneously.",
        characteristics: [
          "Written in languages like CUDA C++, OpenCL, or HLSL",
          "Executed by many threads in parallel",
          "Each thread executes the same code but with different data",
          "Cannot recursively call other kernels",
        ],
        example:
          "Example: Adding two arrays element-wise across 1000+ elements simultaneously",
      },
    },
    threads: {
      title: "Threads",
      description: "The smallest unit of execution on the GPU",
      content: {
        definition:
          "A thread is the basic unit of parallel execution. Each thread runs the same kernel code but operates on different data.",
        characteristics: [
          "Lightweight - thousands can run simultaneously",
          "Each has a unique thread ID",
          "Can access different memory locations",
          "Communicate through shared memory",
        ],
        analogy:
          "Think of threads like workers in a factory assembly line - each doing the same task but on different items.",
      },
    },
    blocks: {
      title: "Thread Blocks",
      description: "Groups of threads that can cooperate",
      content: {
        definition:
          "A block is a group of threads that can synchronize and share data through fast shared memory.",
        characteristics: [
          "Typically 32-1024 threads per block",
          "Threads in a block can synchronize",
          "Share fast shared memory",
          "Executed on the same Streaming Multiprocessor (SM)",
        ],
        limits:
          "Block size is limited by shared memory and register usage per thread.",
      },
    },
    grids: {
      title: "Thread Grids",
      description: "Collections of thread blocks",
      content: {
        definition:
          "A grid is the complete collection of all thread blocks launched for a kernel execution.",
        characteristics: [
          "Can contain thousands of blocks",
          "Blocks within a grid are independent",
          "Grid size determines total parallelism",
          "Can be 1D, 2D, or 3D for different problem structures",
        ],
        calculation: "Total threads = Grid size × Block size",
      },
    },
    warps: {
      title: "Warps",
      description:
        "Groups of 32 threads scheduled together - the foundation of performance",
      content: {
        definition:
          "A warp is a group of 32 consecutive threads that execute the same instruction simultaneously (SIMT - Single Instruction, Multiple Thread).",
        characteristics: [
          "Always 32 threads (NVIDIA GPUs)",
          "Scheduled as a unit by the hardware",
          "All threads in a warp execute the same instruction",
          "Thread divergence reduces performance",
        ],
        performance:
          "Performance Tip: Keep threads in a warp following the same execution path for optimal performance.",
      },
    },
    memory: {
      title: "GPU Memory Hierarchy",
      description:
        "Understanding different types of memory and their performance characteristics",
      content: {
        types: [
          {
            name: "Global Memory",
            desc: "Large but slow memory accessible by all threads",
            size: "GB scale",
            latency: "High (400-800 cycles)",
          },
          {
            name: "Shared Memory",
            desc: "Fast memory shared within a thread block",
            size: "KB scale",
            latency: "Low (1-2 cycles)",
          },
          {
            name: "Registers",
            desc: "Fastest memory private to each thread",
            size: "Limited per thread",
            latency: "Lowest (1 cycle)",
          },
          {
            name: "Constant Memory",
            desc: "Read-only cached memory for constants",
            size: "KB scale",
            latency: "Low when cached",
          },
        ],
      },
    },
    performance: {
      title: "Performance Optimization",
      description: "Key concepts for writing efficient GPU code",
      content: {
        concepts: [
          {
            name: "Occupancy",
            desc: "Ratio of active warps to maximum possible warps on an SM",
          },
          {
            name: "Memory Coalescing",
            desc: "Accessing contiguous memory locations for efficient memory bandwidth usage",
          },
          {
            name: "Thread Divergence",
            desc: "When threads in a warp take different execution paths, reducing efficiency",
          },
          {
            name: "Memory Bandwidth",
            desc: "The rate at which data can be read from or written to memory",
          },
        ],
      },
    },
  },
  zh: {
    title: "GPU架构教程",
    subtitle: "掌握GPU编程基础知识",
    languageToggle: "English",
    sections: {
      intro: "介绍",
      kernels: "核函数",
      threads: "线程",
      blocks: "线程块",
      grids: "线程网格",
      warps: "线程束",
      memory: "内存",
      performance: "性能",
    },
    intro: {
      title: "GPU vs CPU 架构",
      description: "理解GPU和CPU设计的根本差异",
      content: {
        cpuDesc:
          "CPU（中央处理器）：针对顺序处理进行优化，具有复杂的控制逻辑和大型缓存。通常有4-16个核心。",
        gpuDesc:
          "GPU（图形处理器）：针对并行处理进行优化，拥有数千个较简单的核心。专为数据并行工作负载设计。",
        keyDiff:
          "关键差异：CPU擅长处理具有分支逻辑的复杂任务，而GPU擅长对大量数据进行简单的并行操作。",
      },
    },
    kernels: {
      title: "核函数",
      description: "在GPU上运行的代码",
      content: {
        definition:
          "核函数是在GPU上运行的函数。它只编写一次，但由数千个线程同时执行。",
        characteristics: [
          "使用CUDA C++、OpenCL或HLSL等语言编写",
          "由许多线程并行执行",
          "每个线程执行相同的代码但处理不同的数据",
          "不能递归调用其他核函数",
        ],
        example: "示例：同时对1000+个元素进行两个数组的逐元素相加",
      },
    },
    threads: {
      title: "线程",
      description: "GPU上执行的最小单位",
      content: {
        definition:
          "线程是并行执行的基本单位。每个线程运行相同的核函数代码，但操作不同的数据。",
        characteristics: [
          "轻量级 - 数千个可以同时运行",
          "每个都有唯一的线程ID",
          "可以访问不同的内存位置",
          "通过共享内存进行通信",
        ],
        analogy:
          "将线程想象成工厂流水线上的工人 - 每个人做相同的任务但处理不同的物品。",
      },
    },
    blocks: {
      title: "线程块",
      description: "可以协作的线程组",
      content: {
        definition: "线程块是一组可以同步并通过快速共享内存共享数据的线程。",
        characteristics: [
          "通常每个块有32-1024个线程",
          "块内线程可以同步",
          "共享快速共享内存",
          "在同一个流式多处理器(SM)上执行",
        ],
        limits: "块大小受每个线程的共享内存和寄存器使用量限制。",
      },
    },
    grids: {
      title: "线程网格",
      description: "线程块的集合",
      content: {
        definition: "网格是为核函数执行启动的所有线程块的完整集合。",
        characteristics: [
          "可以包含数千个块",
          "网格内的块是独立的",
          "网格大小决定总并行度",
          "可以是1D、2D或3D以适应不同的问题结构",
        ],
        calculation: "总线程数 = 网格大小 × 块大小",
      },
    },
    warps: {
      title: "线程束",
      description: "32个线程一起调度的组 - 性能优化的基础",
      content: {
        definition:
          "线程束是32个连续线程的组，它们同时执行相同的指令（SIMT - 单指令多线程）。",
        characteristics: [
          "始终是32个线程（NVIDIA GPU）",
          "由硬件作为一个单位进行调度",
          "线程束中的所有线程执行相同的指令",
          "线程分化会降低性能",
        ],
        performance:
          "性能提示：保持线程束中的线程遵循相同的执行路径以获得最佳性能。",
      },
    },
    memory: {
      title: "GPU内存层次结构",
      description: "理解不同类型的内存及其性能特征",
      content: {
        types: [
          {
            name: "全局内存",
            desc: "所有线程都可访问的大但慢的内存",
            size: "GB级别",
            latency: "高（400-800周期）",
          },
          {
            name: "共享内存",
            desc: "线程块内共享的快速内存",
            size: "KB级别",
            latency: "低（1-2周期）",
          },
          {
            name: "寄存器",
            desc: "每个线程私有的最快内存",
            size: "每线程有限",
            latency: "最低（1周期）",
          },
          {
            name: "常量内存",
            desc: "用于常量的只读缓存内存",
            size: "KB级别",
            latency: "缓存时较低",
          },
        ],
      },
    },
    performance: {
      title: "性能优化",
      description: "编写高效GPU代码的关键概念",
      content: {
        concepts: [
          {
            name: "占用率",
            desc: "SM上活跃线程束与最大可能线程束的比率",
          },
          {
            name: "内存合并",
            desc: "访问连续内存位置以高效使用内存带宽",
          },
          {
            name: "线程分化",
            desc: "当线程束中的线程采取不同执行路径时，会降低效率",
          },
          {
            name: "内存带宽",
            desc: "从内存读取或写入数据的速率",
          },
        ],
      },
    },
  },
};

function GPUVisualization({ type, lang }: { type: string; lang: Language }) {
  const [animationStep, setAnimationStep] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setAnimationStep((prev) => (prev + 1) % 4);
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  if (type === "hierarchy") {
    return (
      <div className="relative p-8 bg-gradient-to-br from-blue-50 to-purple-50 dark:from-blue-950/20 dark:to-purple-950/20 rounded-xl border">
        <div className="text-center mb-6">
          <h3 className="text-lg font-semibold mb-2">
            {lang === "en" ? "GPU Execution Hierarchy" : "GPU执行层次结构"}
          </h3>
        </div>

        {/* Grid */}
        <div className="border-2 border-blue-500 rounded-lg p-4 mb-4 bg-blue-50/50 dark:bg-blue-950/10">
          <div className="text-sm font-medium text-blue-700 dark:text-blue-300 mb-2">
            {lang === "en" ? "Grid" : "网格"}
          </div>

          {/* Blocks */}
          <div className="grid grid-cols-3 gap-2">
            {[...Array(6)].map((_, blockIndex) => (
              <div
                key={blockIndex}
                className="border border-green-500 rounded p-2 bg-green-50/50 dark:bg-green-950/10"
              >
                <div className="text-xs font-medium text-green-700 dark:text-green-300 mb-1">
                  {lang === "en" ? "Block" : "块"} {blockIndex}
                </div>

                {/* Warps */}
                <div className="grid grid-cols-2 gap-1">
                  {[...Array(4)].map((_, warpIndex) => (
                    <div
                      key={warpIndex}
                      className="border border-purple-400 rounded p-1 bg-purple-50/50 dark:bg-purple-950/10"
                    >
                      <div className="text-xs text-purple-700 dark:text-purple-300">
                        {lang === "en" ? "Warp" : "线程束"}
                      </div>

                      {/* Threads */}
                      <div className="grid grid-cols-4 gap-px mt-1">
                        {[...Array(8)].map((_, threadIndex) => (
                          <div
                            key={threadIndex}
                            className={`w-2 h-2 rounded-sm transition-all duration-300 ${
                              animationStep ===
                              (blockIndex + warpIndex + threadIndex) % 4
                                ? "bg-orange-500"
                                : "bg-orange-200 dark:bg-orange-800"
                            }`}
                          />
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="flex justify-between text-xs text-muted-foreground">
          <span>
            {lang === "en" ? "Threads (orange squares)" : "线程（橙色方块）"}
          </span>
          <span>
            {lang === "en"
              ? "Animation shows parallel execution"
              : "动画显示并行执行"}
          </span>
        </div>
      </div>
    );
  }

  return null;
}

function InteractiveDemo({ type, lang }: { type: string; lang: Language }) {
  const [isRunning, setIsRunning] = useState(false);
  const [cpuProgress, setCpuProgress] = useState(0);
  const [gpuProgress, setGpuProgress] = useState(0);
  const [problemSize] = useState(1048576); // 1M elements for good demo performance
  const [cpuTime, setCpuTime] = useState(0);
  const [gpuTime, setGpuTime] = useState(0);
  const [cpuElements, setCpuElements] = useState<number[]>([]);
  const [gpuElements, setGpuElements] = useState<number[]>([]);

  const startDemo = () => {
    setIsRunning(true);
    setCpuProgress(0);
    setGpuProgress(0);
    setCpuTime(0);
    setGpuTime(0);
    setCpuElements([]);
    setGpuElements([]);

    // Initialize elements array for visualization
    const elements = Array.from(
      { length: Math.min(problemSize, 32) },
      (_, i) => i
    );

    // CPU execution (sequential) - process in chunks for reasonable animation
    const totalCpuTime = Math.max(800, Math.min(3000, problemSize / 5000)); // 0.8-3s based on problem size
    const animationSteps = 20; // 20 animation steps
    const stepDuration = totalCpuTime / animationSteps;
    const elementsPerStep = Math.ceil(problemSize / animationSteps);

    let cpuStep = 0;
    const cpuInterval = setInterval(() => {
      cpuStep++;
      const elementsProcessed = Math.min(
        cpuStep * elementsPerStep,
        problemSize
      );
      const progress = (elementsProcessed / problemSize) * 100;
      const timeElapsed = (cpuStep / animationSteps) * totalCpuTime;

      setCpuProgress(progress);
      setCpuTime(timeElapsed);

      // Update visual elements (show progression through first 32 elements)
      const visualProgress = Math.min(cpuStep * (32 / animationSteps), 32);
      setCpuElements(elements.slice(0, Math.floor(visualProgress)));

      if (cpuStep >= animationSteps) {
        clearInterval(cpuInterval);
        // Ensure final values are exact
        setCpuProgress(100);
        setCpuTime(totalCpuTime);
        setCpuElements(elements);
      }
    }, stepDuration);

    // GPU execution (parallel - much faster)
    setTimeout(() => {
      // GPU has setup overhead but then processes all elements simultaneously
      const gpuSetupTime = 100; // 100ms setup time
      const gpuComputeTime = Math.max(20, problemSize / 500000); // Much faster than CPU
      const totalGpuTime = gpuSetupTime + gpuComputeTime;

      // Setup phase
      setTimeout(() => {
        setGpuProgress(100);
        setGpuTime(totalGpuTime);
        setGpuElements(elements); // All elements processed simultaneously
        setIsRunning(false);
      }, gpuSetupTime);
    }, 20);
  };

  if (type === "kernel") {
    const speedup =
      cpuTime > 0 && gpuTime > 0 ? (cpuTime / gpuTime).toFixed(1) : "0";

    return (
      <div className="mt-4 space-y-4">
        {/* Run Button */}
        <div className="flex justify-center">
          <Button onClick={startDemo} disabled={isRunning} size="lg">
            <Play className="w-4 h-4 mr-2" />
            {isRunning
              ? lang === "en"
                ? "Running..."
                : "运行中..."
              : lang === "en"
              ? "Run Performance Comparison"
              : "运行性能对比"}
          </Button>
        </div>

        {/* CPU vs GPU Comparison */}
        <div className="grid md:grid-cols-2 gap-4">
          {/* CPU Execution */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Cpu className="w-5 h-5 text-red-500" />
                {lang === "en" ? "CPU (Sequential)" : "CPU（顺序执行）"}
              </CardTitle>
              <CardDescription>
                {lang === "en"
                  ? "Processes one element at a time"
                  : "逐个处理元素"}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="bg-gray-900 text-green-400 p-3 rounded-lg font-mono text-xs">
                <div>{lang === "en" ? "// CPU code" : "// CPU代码"}</div>
                <div>for (int i = 0; i &lt; n; i++) {"{"}</div>
                <div className="pl-4">c[i] = a[i] + b[i];</div>
                <div>{"}"}</div>
              </div>

              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>{lang === "en" ? "Progress:" : "进度："}</span>
                  <span>{cpuProgress.toFixed(1)}%</span>
                </div>
                <Progress value={cpuProgress} className="w-full h-3" />
                <div className="text-xs text-muted-foreground">
                  {lang === "en" ? "Time:" : "时间："} {cpuTime.toFixed(1)}ms
                </div>
              </div>

              {/* Visual representation */}
              <div className="space-y-2">
                <div className="text-sm font-medium">
                  {lang === "en" ? "Processing Elements:" : "处理元素："}
                </div>
                <div className="grid grid-cols-8 gap-1">
                  {Array.from({ length: 32 }, (_, i) => (
                    <div
                      key={i}
                      className={`w-6 h-6 rounded text-xs flex items-center justify-center font-mono ${
                        cpuElements.includes(i)
                          ? "bg-red-500 text-white"
                          : "bg-gray-200 text-gray-500"
                      }`}
                    >
                      {i}
                    </div>
                  ))}
                </div>
                <div className="text-xs text-muted-foreground">
                  {lang === "en" ? "Red = Completed" : "红色 = 已完成"}
                </div>
              </div>
            </CardContent>
          </Card>

          {/* GPU Execution */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Zap className="w-5 h-5 text-green-500" />
                {lang === "en" ? "GPU (Parallel)" : "GPU（并行执行）"}
              </CardTitle>
              <CardDescription>
                {lang === "en"
                  ? "Processes all elements simultaneously"
                  : "同时处理所有元素"}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="bg-gray-900 text-green-400 p-3 rounded-lg font-mono text-xs">
                <div>{lang === "en" ? "// GPU kernel" : "// GPU核函数"}</div>
                <div>__global__ void addArrays(...) {"{"}</div>
                <div className="pl-4">
                  int i = blockIdx.x * blockDim.x + threadIdx.x;
                </div>
                <div className="pl-4">if (i &lt; n) c[i] = a[i] + b[i];</div>
                <div>{"}"}</div>
              </div>

              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>{lang === "en" ? "Progress:" : "进度："}</span>
                  <span>{gpuProgress.toFixed(1)}%</span>
                </div>
                <Progress value={gpuProgress} className="w-full h-3" />
                <div className="text-xs text-muted-foreground">
                  {lang === "en" ? "Time:" : "时间："} {gpuTime.toFixed(1)}ms
                </div>
              </div>

              {/* Visual representation */}
              <div className="space-y-2">
                <div className="text-sm font-medium">
                  {lang === "en" ? "Processing Elements:" : "处理元素："}
                </div>
                <div className="grid grid-cols-8 gap-1">
                  {Array.from({ length: 32 }, (_, i) => (
                    <div
                      key={i}
                      className={`w-6 h-6 rounded text-xs flex items-center justify-center font-mono ${
                        gpuElements.includes(i)
                          ? "bg-green-500 text-white"
                          : "bg-gray-200 text-gray-500"
                      }`}
                    >
                      {i}
                    </div>
                  ))}
                </div>
                <div className="text-xs text-muted-foreground">
                  {lang === "en"
                    ? "Green = Completed (All at Once!)"
                    : "绿色 = 已完成（同时完成！）"}
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Performance Summary */}
        {!isRunning && cpuTime > 0 && gpuTime > 0 && (
          <Card className="border-2 border-green-500">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-green-500" />
                {lang === "en" ? "Performance Results" : "性能结果"}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center">
                  <div className="text-2xl font-bold text-red-500">
                    {cpuTime.toFixed(1)}ms
                  </div>
                  <div className="text-sm text-muted-foreground">
                    CPU {lang === "en" ? "Time" : "时间"}
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-500">
                    {gpuTime.toFixed(1)}ms
                  </div>
                  <div className="text-sm text-muted-foreground">
                    GPU {lang === "en" ? "Time" : "时间"}
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-500">
                    {speedup}x
                  </div>
                  <div className="text-sm text-muted-foreground">
                    {lang === "en" ? "Speedup" : "加速比"}
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-purple-500">
                    {problemSize.toLocaleString()}
                  </div>
                  <div className="text-sm text-muted-foreground">
                    {lang === "en" ? "Elements" : "元素"}
                  </div>
                </div>
              </div>

              <div className="mt-4 p-4 bg-gradient-to-r from-green-50 to-blue-50 dark:from-green-950/20 dark:to-blue-950/20 rounded-lg">
                <h4 className="font-semibold mb-2">
                  {lang === "en" ? "Key Insights:" : "关键要点："}
                </h4>
                <ul className="text-sm space-y-1">
                  <li>
                    •{" "}
                    {lang === "en"
                      ? "GPU has setup overhead but massive parallelism"
                      : "GPU有启动开销，但具有大规模并行性"}
                  </li>
                  <li>
                    •{" "}
                    {lang === "en"
                      ? "CPU processes sequentially, GPU processes everything simultaneously"
                      : "CPU顺序处理，GPU同时处理所有内容"}
                  </li>
                  <li>
                    •{" "}
                    {lang === "en"
                      ? "Larger problems benefit more from GPU parallelism"
                      : "更大的问题从GPU并行性中受益更多"}
                  </li>
                  <li>
                    •{" "}
                    {lang === "en"
                      ? "GPU excels at simple operations on large datasets"
                      : "GPU擅长对大数据集进行简单操作"}
                  </li>
                </ul>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    );
  }

  return null;
}

interface GpuArchitectureTutorialProps {
  language?: string;
  setLanguage?: (language: string) => void;
}

export default function GpuArchitectureTutorial({
  language: propLanguage,
  setLanguage: propSetLanguage,
}: GpuArchitectureTutorialProps = {}) {
  const [internalLanguage, setInternalLanguage] = useState<Language>("en");
  const [activeSection, setActiveSection] = useState("intro");

  // Use prop language if provided, otherwise use internal state
  const language = (propLanguage as Language) || internalLanguage;
  const setLanguage = propSetLanguage
    ? (lang: string) => propSetLanguage(lang)
    : (lang: Language) => setInternalLanguage(lang);

  const t = content[language];

  const sections = [
    { id: "intro", title: t.sections.intro, icon: Cpu },
    { id: "kernels", title: t.sections.kernels, icon: Code },
    { id: "threads", title: t.sections.threads, icon: Zap },
    { id: "blocks", title: t.sections.blocks, icon: Box },
    { id: "grids", title: t.sections.grids, icon: Grid3X3 },
    { id: "warps", title: t.sections.warps, icon: Layers },
    { id: "memory", title: t.sections.memory, icon: MemoryStick },
    { id: "performance", title: t.sections.performance, icon: Workflow },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-blue-50/30 dark:to-blue-950/10">
      {/* Header */}
      <header className="border-b bg-background">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                {t.title}
              </h1>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Navigation */}
          <div className="lg:col-span-1">
            <div>
              <nav className="space-y-2">
                {sections.map((section) => {
                  const Icon = section.icon;
                  return (
                    <button
                      key={section.id}
                      onClick={() => setActiveSection(section.id)}
                      className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-left transition-all ${
                        activeSection === section.id
                          ? "bg-primary text-primary-foreground shadow-md"
                          : "hover:bg-accent hover:text-accent-foreground"
                      }`}
                    >
                      <Icon className="w-5 h-5" />
                      <span className="font-medium">{section.title}</span>
                      {activeSection === section.id && (
                        <ChevronRight className="w-4 h-4 ml-auto" />
                      )}
                    </button>
                  );
                })}
              </nav>
            </div>
          </div>

          {/* Content */}
          <div className="lg:col-span-3">
            <div className="space-y-8">
              {/* Introduction */}
              {activeSection === "intro" && (
                <div className="space-y-6">
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Cpu className="w-6 h-6" />
                        {t.intro.title}
                      </CardTitle>
                      <CardDescription>{t.intro.description}</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="grid md:grid-cols-2 gap-4">
                        <Card>
                          <CardHeader>
                            <CardTitle className="text-lg">CPU</CardTitle>
                          </CardHeader>
                          <CardContent>
                            <p className="text-sm">{t.intro.content.cpuDesc}</p>
                          </CardContent>
                        </Card>

                        <Card>
                          <CardHeader>
                            <CardTitle className="text-lg">GPU</CardTitle>
                          </CardHeader>
                          <CardContent>
                            <p className="text-sm">{t.intro.content.gpuDesc}</p>
                          </CardContent>
                        </Card>
                      </div>

                      <div className="p-4 bg-blue-50 dark:bg-blue-950/20 rounded-lg border border-blue-200 dark:border-blue-800">
                        <p className="text-sm font-medium text-blue-900 dark:text-blue-100">
                          {t.intro.content.keyDiff}
                        </p>
                      </div>
                    </CardContent>
                  </Card>

                  <GPUVisualization type="hierarchy" lang={language} />
                </div>
              )}

              {/* Kernels */}
              {activeSection === "kernels" && (
                <div className="space-y-6">
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Code className="w-6 h-6" />
                        {t.kernels.title}
                      </CardTitle>
                      <CardDescription>{t.kernels.description}</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <p>{t.kernels.content.definition}</p>

                      <div>
                        <h4 className="font-semibold mb-2">
                          {language === "en" ? "Characteristics:" : "特征："}
                        </h4>
                        <ul className="space-y-1">
                          {t.kernels.content.characteristics.map(
                            (char, index) => (
                              <li
                                key={index}
                                className="flex items-start gap-2"
                              >
                                <ArrowRight className="w-4 h-4 mt-0.5 text-muted-foreground" />
                                <span className="text-sm">{char}</span>
                              </li>
                            )
                          )}
                        </ul>
                      </div>

                      <div className="p-4 bg-green-50 dark:bg-green-950/20 rounded-lg border border-green-200 dark:border-green-800">
                        <p className="text-sm text-green-900 dark:text-green-100">
                          {t.kernels.content.example}
                        </p>
                      </div>
                    </CardContent>
                  </Card>

                  <InteractiveDemo type="kernel" lang={language} />
                </div>
              )}

              {/* Threads */}
              {activeSection === "threads" && (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Zap className="w-6 h-6" />
                      {t.threads.title}
                    </CardTitle>
                    <CardDescription>{t.threads.description}</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <p>{t.threads.content.definition}</p>

                    <div>
                      <h4 className="font-semibold mb-2">
                        {language === "en" ? "Characteristics:" : "特征："}
                      </h4>
                      <ul className="space-y-1">
                        {t.threads.content.characteristics.map(
                          (char, index) => (
                            <li key={index} className="flex items-start gap-2">
                              <ArrowRight className="w-4 h-4 mt-0.5 text-muted-foreground" />
                              <span className="text-sm">{char}</span>
                            </li>
                          )
                        )}
                      </ul>
                    </div>

                    <div className="p-4 bg-orange-50 dark:bg-orange-950/20 rounded-lg border border-orange-200 dark:border-orange-800">
                      <p className="text-sm text-orange-900 dark:text-orange-100">
                        <strong>
                          {language === "en" ? "Analogy: " : "类比："}
                        </strong>
                        {t.threads.content.analogy}
                      </p>
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Blocks */}
              {activeSection === "blocks" && (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Box className="w-6 h-6" />
                      {t.blocks.title}
                    </CardTitle>
                    <CardDescription>{t.blocks.description}</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <p>{t.blocks.content.definition}</p>

                    <div>
                      <h4 className="font-semibold mb-2">
                        {language === "en" ? "Characteristics:" : "特征："}
                      </h4>
                      <ul className="space-y-1">
                        {t.blocks.content.characteristics.map((char, index) => (
                          <li key={index} className="flex items-start gap-2">
                            <ArrowRight className="w-4 h-4 mt-0.5 text-muted-foreground" />
                            <span className="text-sm">{char}</span>
                          </li>
                        ))}
                      </ul>
                    </div>

                    <div className="p-4 bg-yellow-50 dark:bg-yellow-950/20 rounded-lg border border-yellow-200 dark:border-yellow-800">
                      <p className="text-sm text-yellow-900 dark:text-yellow-100">
                        <strong>
                          {language === "en" ? "Important: " : "重要："}
                        </strong>
                        {t.blocks.content.limits}
                      </p>
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Grids */}
              {activeSection === "grids" && (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Grid3X3 className="w-6 h-6" />
                      {t.grids.title}
                    </CardTitle>
                    <CardDescription>{t.grids.description}</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <p>{t.grids.content.definition}</p>

                    <div>
                      <h4 className="font-semibold mb-2">
                        {language === "en" ? "Characteristics:" : "特征："}
                      </h4>
                      <ul className="space-y-1">
                        {t.grids.content.characteristics.map((char, index) => (
                          <li key={index} className="flex items-start gap-2">
                            <ArrowRight className="w-4 h-4 mt-0.5 text-muted-foreground" />
                            <span className="text-sm">{char}</span>
                          </li>
                        ))}
                      </ul>
                    </div>

                    <div className="p-4 bg-purple-50 dark:bg-purple-950/20 rounded-lg border border-purple-200 dark:border-purple-800">
                      <p className="text-sm font-medium text-purple-900 dark:text-purple-100">
                        {t.grids.content.calculation}
                      </p>
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Warps */}
              {activeSection === "warps" && (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Layers className="w-6 h-6" />
                      {t.warps.title}
                    </CardTitle>
                    <CardDescription>{t.warps.description}</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <p>{t.warps.content.definition}</p>

                    <div>
                      <h4 className="font-semibold mb-2">
                        {language === "en" ? "Characteristics:" : "特征："}
                      </h4>
                      <ul className="space-y-1">
                        {t.warps.content.characteristics.map((char, index) => (
                          <li key={index} className="flex items-start gap-2">
                            <ArrowRight className="w-4 h-4 mt-0.5 text-muted-foreground" />
                            <span className="text-sm">{char}</span>
                          </li>
                        ))}
                      </ul>
                    </div>

                    <div className="p-4 bg-red-50 dark:bg-red-950/20 rounded-lg border border-red-200 dark:border-red-800">
                      <p className="text-sm text-red-900 dark:text-red-100">
                        <strong>
                          {language === "en"
                            ? "Performance Tip: "
                            : "性能提示："}
                        </strong>
                        {t.warps.content.performance}
                      </p>
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Memory */}
              {activeSection === "memory" && (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <MemoryStick className="w-6 h-6" />
                      {t.memory.title}
                    </CardTitle>
                    <CardDescription>{t.memory.description}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid gap-4">
                      {t.memory.content.types.map((type, index) => (
                        <Card
                          key={index}
                          className="border-l-4 border-l-blue-500"
                        >
                          <CardHeader className="pb-2">
                            <CardTitle className="text-lg">
                              {type.name}
                            </CardTitle>
                          </CardHeader>
                          <CardContent className="pt-0">
                            <p className="text-sm mb-2">{type.desc}</p>
                            <div className="flex gap-4 text-xs text-muted-foreground">
                              <Badge variant="outline">
                                {language === "en" ? "Size: " : "大小："}
                                {type.size}
                              </Badge>
                              <Badge variant="outline">
                                {language === "en" ? "Latency: " : "延迟："}
                                {type.latency}
                              </Badge>
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Performance */}
              {activeSection === "performance" && (
                <div className="space-y-6">
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Workflow className="w-6 h-6" />
                        {t.performance.title}
                      </CardTitle>
                      <CardDescription>
                        {t.performance.description}
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="grid gap-4">
                        {t.performance.content.concepts.map(
                          (concept, index) => (
                            <Card
                              key={index}
                              className="border-l-4 border-l-green-500"
                            >
                              <CardHeader className="pb-2">
                                <CardTitle className="text-lg">
                                  {concept.name}
                                </CardTitle>
                              </CardHeader>
                              <CardContent className="pt-0">
                                <p className="text-sm">{concept.desc}</p>
                              </CardContent>
                            </Card>
                          )
                        )}
                      </div>
                    </CardContent>
                  </Card>

                  <InteractiveDemo type="kernel" lang={language} />
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
