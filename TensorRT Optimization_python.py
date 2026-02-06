# src/detection/tensorrt_optimizer.py
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import torch
import os

class TensorRTEngine:
    """TensorRT optimization for YOLO model"""
    
    def __init__(self, onnx_path, engine_path=None, fp16=True, workspace_size=1<<30):
        """
        Initialize TensorRT engine
        
        Args:
            onnx_path: Path to ONNX model
            engine_path: Path to save/load TensorRT engine
            fp16: Use FP16 precision
            workspace_size: Workspace size in bytes
        """
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.fp16 = fp16
        self.workspace_size = workspace_size
        
        if engine_path and os.path.exists(engine_path):
            print(f"Loading existing TensorRT engine: {engine_path}")
            self.engine = self.load_engine(engine_path)
        else:
            print(f"Building TensorRT engine from ONNX: {onnx_path}")
            self.engine = self.build_engine(onnx_path, engine_path)
        
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
        # Allocate buffers
        self.inputs, self.outputs, self.bindings = self.allocate_buffers()
        
    def build_engine(self, onnx_path, engine_path):
        """Build TensorRT engine from ONNX model"""
        
        builder = trt.Builder(self.logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, self.logger)
        
        # Parse ONNX model
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                raise ValueError("Failed to parse ONNX model")
        
        # Build configuration
        config = builder.create_builder_config()
        config.max_workspace_size = self.workspace_size
        
        if self.fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        
        # Optimization profiles
        profile = builder.create_optimization_profile()
        
        # Get input shape (assuming batch, channel, height, width)
        input_shape = network.get_input(0).shape
        min_shape = (1, input_shape[1], 320, 320)  # Minimum size
        opt_shape = (1, input_shape[1], 640, 640)  # Optimal size
        max_shape = (1, input_shape[1], 1280, 1280)  # Maximum size
        
        profile.set_shape("images", min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
        
        # Build engine
        engine = builder.build_engine(network, config)
        
        if engine_path:
            with open(engine_path, 'wb') as f:
                f.write(engine.serialize())
            print(f"TensorRT engine saved to: {engine_path}")
        
        return engine
    
    def load_engine(self, engine_path):
        """Load serialized TensorRT engine"""
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            return runtime.deserialize_cuda_engine(f.read())
    
    def allocate_buffers(self):
        """Allocate CUDA buffers for input and output"""
        inputs = []
        outputs = []
        bindings = []
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})
        
        return inputs, outputs, bindings
    
    def infer(self, input_tensor):
        """Run inference with TensorRT"""
        
        # Copy input to device
        np.copyto(self.inputs[0]['host'], input_tensor.ravel())
        cuda.memcpy_htod_async(
            self.inputs[0]['device'], 
            self.inputs[0]['host'], 
            self.stream
        )
        
        # Run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )
        
        # Copy output from device
        for output in self.outputs:
            cuda.memcpy_dtoh_async(
                output['host'], 
                output['device'], 
                self.stream
            )
        
        self.stream.synchronize()
        
        # Process outputs
        outputs = []
        for output in self.outputs:
            outputs.append(output['host'])
        
        return outputs
    
    def preprocess(self, image):
        """Preprocess image for TensorRT inference"""
        # Resize, normalize, and convert to CHW format
        image_resized = cv2.resize(image, (640, 640))
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        image_normalized = image_rgb.astype(np.float32) / 255.0
        image_chw = np.transpose(image_normalized, (2, 0, 1))
        image_batched = np.expand_dims(image_chw, axis=0)
        
        return image_batched
    
    def postprocess(self, outputs, original_shape, conf_threshold=0.5):
        """Postprocess TensorRT outputs to detections"""
        
        # YOLOv8 output format: [batch, 84, 8400]
        predictions = outputs[0].reshape(1, 84, -1)
        
        detections = []
        for i in range(predictions.shape[2]):
            confidence = predictions[0, 4, i]
            
            if confidence > conf_threshold:
                # Get bounding box
                cx = predictions[0, 0, i]
                cy = predictions[0, 1, i]
                w = predictions[0, 2, i]
                h = predictions[0, 3, i]
                
                # Convert from center to corner coordinates
                x1 = (cx - w/2) * original_shape[1]
                y1 = (cy - h/2) * original_shape[0]
                x2 = (cx + w/2) * original_shape[1]
                y2 = (cy + h/2) * original_shape[0]
                
                # Get class probabilities
                class_probs = predictions[0, 5:85, i]
                class_id = np.argmax(class_probs)
                class_score = class_probs[class_id] * confidence
                
                if class_score > conf_threshold:
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(class_score),
                        'class_id': int(class_id),
                        'class_name': f"class_{class_id}"
                    })
        
        return detections

class TRTOptimizer:
    """TensorRT optimization manager"""
    
    @staticmethod
    def optimize_yolo(model_path, output_dir="models"):
        """
        Full pipeline for YOLO model optimization with TensorRT
        
        Args:
            model_path: Path to YOLO model (.pt file)
            output_dir: Directory to save optimized models
        """
        
        import subprocess
        import sys
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print("=" * 50)
        print("TENSORRT OPTIMIZATION PIPELINE")
        print("=" * 50)
        
        # Step 1: Export to ONNX
        print("\n[1/4] Exporting to ONNX format...")
        onnx_path = os.path.join(output_dir, "model.onnx")
        
        export_cmd = [
            sys.executable, "-m", "ultralytics.export",
            "model=" + model_path,
            "format=onnx",
            "imgsz=640",
            "half=True",
            "simplify=True"
        ]
        
        try:
            subprocess.run(export_cmd, check=True)
            print(f"✓ ONNX model saved to: {onnx_path}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to export ONNX: {e}")
            return None
        
        # Step 2: Optimize ONNX with onnxsim
        print("\n[2/4] Optimizing ONNX model...")
        try:
            import onnx
            from onnxsim import simplify
            
            model_onnx = onnx.load(onnx_path)
            model_simplified, check = simplify(model_onnx)
            
            if check:
                onnx.save(model_simplified, onnx_path)
                print("✓ ONNX model simplified successfully")
            else:
                print("✗ Failed to simplify ONNX model")
        except ImportError:
            print("⚠ onnxsim not installed, skipping simplification")
        
        # Step 3: Build TensorRT engine
        print("\n[3/4] Building TensorRT engine...")
        engine_path = os.path.join(output_dir, "model.engine")
        
        try:
            trt_engine = TensorRTEngine(
                onnx_path=onnx_path,
                engine_path=engine_path,
                fp16=True,
                workspace_size=1<<30
            )
            print(f"✓ TensorRT engine built: {engine_path}")
        except Exception as e:
            print(f"✗ Failed to build TensorRT engine: {e}")
            return None
        
        # Step 4: Benchmark performance
        print("\n[4/4] Benchmarking performance...")
        benchmark_results = TRTOptimizer.benchmark(trt_engine)
        
        print("\n" + "=" * 50)
        print("OPTIMIZATION COMPLETE")
        print("=" * 50)
        print(f"Original model: {model_path}")
        print(f"ONNX model: {onnx_path}")
        print(f"TensorRT engine: {engine_path}")
        print(f"\nPerformance improvement: {benchmark_results['speedup']:.2f}x")
        
        return engine_path
    
    @staticmethod
    def benchmark(trt_engine, num_iterations=100):
        """Benchmark TensorRT engine performance"""
        
        import time
        
        # Create dummy input
        dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
        
        # Warm-up
        for _ in range(10):
            trt_engine.infer(dummy_input)
        
        # Benchmark
        start_time = time.time()
        for i in range(num_iterations):
            trt_engine.infer(dummy_input)
        
        end_time = time.time()
        
        inference_time = (end_time - start_time) / num_iterations
        fps = 1.0 / inference_time
        
        return {
            'inference_time_ms': inference_time * 1000,
            'fps': fps,
            'speedup': 30.0 / inference_time  # Assuming 30ms baseline
        }