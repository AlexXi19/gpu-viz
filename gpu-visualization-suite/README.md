# GPU Visualization Suite

An interactive educational platform for learning GPU architecture and CUDA programming concepts through hands-on visualizations.

## Features

### 🎓 GPU Architecture Tutorial

- **Interactive Lessons**: Learn GPU fundamentals with visual demonstrations
- **Multi-language Support**: Available in English and Chinese
- **Comprehensive Topics**: Covers kernels, threads, blocks, warps, memory hierarchy, and performance optimization
- **Visual Demonstrations**: Interactive animations showing GPU execution hierarchy

### 🧮 CUDA Matrix Multiplication Visualizer

- **Thread Mapping**: Visualize how CUDA threads are organized and mapped to matrix elements
- **Memory Layout**: Understand row-major memory organization and access patterns
- **Step-by-Step Computation**: Watch matrix multiplication unfold with animated execution
- **Index Calculation**: Learn how global coordinates map to thread indices
- **Live Kernel Execution**: See CUDA kernel code highlighted during execution

## Getting Started

First, run the development server:

```bash
bun dev
```

Open [http://localhost:5173](http://localhost:5173) with your browser to see the application.

## Usage

The application provides three main sections:

1. **Overview**: Introduction to both tools with feature highlights
2. **GPU Architecture Tutorial**: Comprehensive tutorial covering GPU fundamentals
3. **CUDA Matrix Multiply**: Interactive visualization of matrix multiplication

Navigate between sections using the tab interface at the top of the application.

## Educational Benefits

- **Visual Learning**: Complex concepts made accessible through interactive visualizations
- **Hands-on Experience**: Click and explore to understand GPU programming concepts
- **Real-time Feedback**: See immediate results of parameter changes
- **Practical Application**: Learn through concrete examples like matrix multiplication

## Technologies Used

- **React 19** with TypeScript
- **Vite** for fast development
- **Tailwind CSS v4** for styling
- **ShadCN UI** for components
- **Lucide React** for icons

## Project Structure

```
src/
├── App.tsx                    # Main navigation and layout
├── CudaMatrixMultiply.tsx     # Matrix multiplication visualizer
├── GpuArchitectureTutorial.tsx # GPU architecture tutorial
├── components/ui/             # Reusable UI components
├── lib/                       # Utility functions
└── backend/                   # API endpoints
```

## Build and Deploy

Build the project:

```bash
bun run build
```

Preview the production build:

```bash
bun run preview
```

## Contributing

This project is designed for educational purposes. Contributions that improve the learning experience are welcome:

- Additional visualization modes
- More interactive examples
- Performance optimization demonstrations
- Enhanced explanations and documentation

## License

Educational use encouraged. Perfect for:

- Computer science courses
- GPU programming workshops
- Self-directed learning
- Technical demonstrations
