import { useState, useEffect } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Grid3X3,
  GraduationCap,
  Zap,
  Code,
  MemoryStick,
  Cpu,
  ArrowRight,
} from "lucide-react";

import CudaMatrixMultiply from "./CudaMatrixMultiply";
import GpuArchitectureTutorial from "./GpuArchitectureTutorial";
import CudaIndexingTutorial from "./CudaIndexingTutorial";

export default function App() {
  const [activeTab, setActiveTab] = useState(() => {
    // Load from localStorage on initial render
    if (typeof window !== "undefined") {
      return localStorage.getItem("gpu-viz-active-tab") || "overview";
    }
    return "overview";
  });

  // Save to localStorage whenever activeTab changes
  useEffect(() => {
    if (typeof window !== "undefined") {
      localStorage.setItem("gpu-viz-active-tab", activeTab);
    }
  }, [activeTab]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-blue-50/30 dark:to-blue-950/10">
      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <div className="sticky top-0 z-50 border-b bg-background/80 backdrop-blur-sm">
          <div className="container mx-auto px-4">
            <div className="flex flex-col space-y-4 py-4">
              <div className="flex items-center justify-between">
                <div>
                  <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                    GPU Visualization Suite
                  </h1>
                  <p className="text-muted-foreground mt-1">
                    Interactive tools for understanding GPU architecture and
                    CUDA programming
                  </p>
                </div>
                <div className="flex items-center space-x-2">
                  <Badge variant="outline" className="text-sm">
                    Educational
                  </Badge>
                  <Badge variant="outline" className="text-sm">
                    Interactive
                  </Badge>
                </div>
              </div>

              <TabsList className="grid w-full grid-cols-4">
                <TabsTrigger
                  value="overview"
                  className="flex items-center gap-2"
                >
                  <Cpu className="w-4 h-4" />
                  Overview
                </TabsTrigger>
                <TabsTrigger
                  value="tutorial"
                  className="flex items-center gap-2"
                >
                  <GraduationCap className="w-4 h-4" />
                  GPU Architecture Tutorial
                </TabsTrigger>
                <TabsTrigger
                  value="indexing"
                  className="flex items-center gap-2"
                >
                  <Code className="w-4 h-4" />
                  CUDA 3D Indexing
                </TabsTrigger>
                <TabsTrigger value="matrix" className="flex items-center gap-2">
                  <Grid3X3 className="w-4 h-4" />
                  CUDA Matrix Multiply
                </TabsTrigger>
              </TabsList>
            </div>
          </div>
        </div>

        <TabsContent value="overview" className="container mx-auto px-4 py-8">
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            <Card
              className="cursor-pointer hover:shadow-lg transition-shadow"
              onClick={() => setActiveTab("tutorial")}
            >
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <GraduationCap className="w-5 h-5 text-blue-500" />
                  GPU Architecture Tutorial
                </CardTitle>
                <CardDescription>
                  Learn the fundamentals of GPU architecture and parallel
                  computing
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="flex items-center gap-2 text-sm">
                      <Code className="w-4 h-4 text-green-500" />
                      <span>Interactive Lessons</span>
                    </div>
                    <div className="flex items-center gap-2 text-sm">
                      <Zap className="w-4 h-4 text-yellow-500" />
                      <span>Visual Demos</span>
                    </div>
                    <div className="flex items-center gap-2 text-sm">
                      <MemoryStick className="w-4 h-4 text-purple-500" />
                      <span>Memory Hierarchy</span>
                    </div>
                    <div className="flex items-center gap-2 text-sm">
                      <Cpu className="w-4 h-4 text-red-500" />
                      <span>Parallel Processing</span>
                    </div>
                  </div>
                  <div className="bg-muted p-4 rounded-lg">
                    <h4 className="font-semibold mb-2">What you'll learn:</h4>
                    <ul className="text-sm space-y-1 text-muted-foreground">
                      <li>• GPU vs CPU architecture differences</li>
                      <li>• Kernels, threads, blocks, and grids</li>
                      <li>• Memory hierarchy and optimization</li>
                      <li>• Performance considerations</li>
                    </ul>
                  </div>
                  <Button className="w-full" variant="outline">
                    <span>Start Learning</span>
                    <ArrowRight className="w-4 h-4 ml-2" />
                  </Button>
                </div>
              </CardContent>
            </Card>

            <Card
              className="cursor-pointer hover:shadow-lg transition-shadow"
              onClick={() => setActiveTab("indexing")}
            >
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Code className="w-5 h-5 text-orange-500" />
                  CUDA 3D Indexing Tutorial
                </CardTitle>
                <CardDescription>
                  Master CUDA's 3D thread indexing with the apartment complex
                  analogy
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="flex items-center gap-2 text-sm">
                      <Cpu className="w-4 h-4 text-blue-500" />
                      <span>3D Visualization</span>
                    </div>
                    <div className="flex items-center gap-2 text-sm">
                      <Code className="w-4 h-4 text-green-500" />
                      <span>Real CUDA Code</span>
                    </div>
                    <div className="flex items-center gap-2 text-sm">
                      <Zap className="w-4 h-4 text-yellow-500" />
                      <span>Interactive Demos</span>
                    </div>
                    <div className="flex items-center gap-2 text-sm">
                      <MemoryStick className="w-4 h-4 text-purple-500" />
                      <span>ID Calculations</span>
                    </div>
                  </div>
                  <div className="bg-muted p-4 rounded-lg">
                    <h4 className="font-semibold mb-2">Based on whoami.cu:</h4>
                    <ul className="text-sm space-y-1 text-muted-foreground">
                      <li>• 3D grid and block organization</li>
                      <li>• Linear ID calculation formulas</li>
                      <li>• Apartment complex analogy</li>
                      <li>• Step-by-step ID derivation</li>
                    </ul>
                  </div>
                  <Button className="w-full" variant="outline">
                    <span>Learn 3D Indexing</span>
                    <ArrowRight className="w-4 h-4 ml-2" />
                  </Button>
                </div>
              </CardContent>
            </Card>

            <Card
              className="cursor-pointer hover:shadow-lg transition-shadow"
              onClick={() => setActiveTab("matrix")}
            >
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Grid3X3 className="w-5 h-5 text-green-500" />
                  CUDA Matrix Multiplication
                </CardTitle>
                <CardDescription>
                  Visualize how CUDA threads work together to multiply matrices
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="flex items-center gap-2 text-sm">
                      <Grid3X3 className="w-4 h-4 text-blue-500" />
                      <span>Thread Mapping</span>
                    </div>
                    <div className="flex items-center gap-2 text-sm">
                      <MemoryStick className="w-4 h-4 text-purple-500" />
                      <span>Memory Layout</span>
                    </div>
                    <div className="flex items-center gap-2 text-sm">
                      <Zap className="w-4 h-4 text-yellow-500" />
                      <span>Step-by-step</span>
                    </div>
                    <div className="flex items-center gap-2 text-sm">
                      <Code className="w-4 h-4 text-green-500" />
                      <span>Live Computation</span>
                    </div>
                  </div>
                  <div className="bg-muted p-4 rounded-lg">
                    <h4 className="font-semibold mb-2">
                      Interactive Features:
                    </h4>
                    <ul className="text-sm space-y-1 text-muted-foreground">
                      <li>• Visualize thread organization</li>
                      <li>• See memory access patterns</li>
                      <li>• Watch computation step-by-step</li>
                      <li>• Understand index calculations</li>
                    </ul>
                  </div>
                  <Button className="w-full" variant="outline">
                    <span>Explore Visualization</span>
                    <ArrowRight className="w-4 h-4 ml-2" />
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="tutorial" className="w-full">
          <GpuArchitectureTutorial />
        </TabsContent>

        <TabsContent value="indexing" className="w-full">
          <CudaIndexingTutorial />
        </TabsContent>

        <TabsContent value="matrix" className="w-full">
          <CudaMatrixMultiply />
        </TabsContent>
      </Tabs>
    </div>
  );
}
