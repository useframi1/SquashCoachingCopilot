import torch.nn as nn
import torch


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, pad=1, stride=1, bias=True
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=pad,
                bias=bias,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.block(x)


class BallTrackerNet(nn.Module):
    def __init__(self, out_channels=256):
        super().__init__()
        self.out_channels = out_channels

        self.conv1 = ConvBlock(in_channels=9, out_channels=64)
        self.conv2 = ConvBlock(in_channels=64, out_channels=64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = ConvBlock(in_channels=64, out_channels=128)
        self.conv4 = ConvBlock(in_channels=128, out_channels=128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = ConvBlock(in_channels=128, out_channels=256)
        self.conv6 = ConvBlock(in_channels=256, out_channels=256)
        self.conv7 = ConvBlock(in_channels=256, out_channels=256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv8 = ConvBlock(in_channels=256, out_channels=512)
        self.conv9 = ConvBlock(in_channels=512, out_channels=512)
        self.conv10 = ConvBlock(in_channels=512, out_channels=512)
        self.ups1 = nn.Upsample(scale_factor=2)
        self.conv11 = ConvBlock(in_channels=512, out_channels=256)
        self.conv12 = ConvBlock(in_channels=256, out_channels=256)
        self.conv13 = ConvBlock(in_channels=256, out_channels=256)
        self.ups2 = nn.Upsample(scale_factor=2)
        self.conv14 = ConvBlock(in_channels=256, out_channels=128)
        self.conv15 = ConvBlock(in_channels=128, out_channels=128)
        self.ups3 = nn.Upsample(scale_factor=2)
        self.conv16 = ConvBlock(in_channels=128, out_channels=64)
        self.conv17 = ConvBlock(in_channels=64, out_channels=64)
        self.conv18 = ConvBlock(in_channels=64, out_channels=self.out_channels)

        self.softmax = nn.Softmax(dim=1)
        self._init_weights()

    def forward(self, x, testing=False):
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.ups1(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.ups2(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.ups3(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)
        # x = self.softmax(x)
        out = x.reshape(batch_size, self.out_channels, -1)
        if testing:
            out = self.softmax(out)
        return out

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.uniform_(module.weight, -0.05, 0.05)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)


if __name__ == "__main__":
    import time
    from squashcopilot.common.utils import get_package_dir

    # Test on GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load pretrained weights
    parent_dir = get_package_dir(__file__)
    model_path = parent_dir + "/weights/ball_tracker.pt"
    print(f"Loading weights from: {model_path}")

    # Create model in FP32 with pretrained weights
    model_fp32 = BallTrackerNet()
    model_fp32.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=False)
    )
    model_fp32 = model_fp32.to(device)
    model_fp32.eval()

    # Create model in FP16 with SAME pretrained weights
    model_fp16 = BallTrackerNet()
    model_fp16.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=False)
    )
    model_fp16 = model_fp16.to(device).half()
    model_fp16.eval()

    # Test input (batch_size=8 to avoid OOM on smaller GPUs)
    batch_size = 8
    inp_fp32 = torch.rand(batch_size, 9, 360, 640, device=device)
    inp_fp16 = inp_fp32.half()

    # Warmup
    print("\nWarming up...")
    with torch.no_grad():
        for _ in range(5):
            _ = model_fp32(inp_fp32)
            if device == "cuda":
                _ = model_fp16(inp_fp16)
    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark FP32
    print("\nBenchmarking FP32...")
    num_iterations = 50
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iterations):
            out_fp32 = model_fp32(inp_fp32)
    if device == "cuda":
        torch.cuda.synchronize()
    fp32_time = (time.perf_counter() - start) / num_iterations
    print(f"FP32 output shape: {out_fp32.shape}")
    print(f"FP32 inference time: {fp32_time * 1000:.2f} ms per batch")
    print(f"FP32 throughput: {batch_size / fp32_time:.1f} frames/sec")

    # Benchmark FP16 (only on CUDA)
    if device == "cuda":
        print("\nBenchmarking FP16...")
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_iterations):
                out_fp16 = model_fp16(inp_fp16)
        torch.cuda.synchronize()
        fp16_time = (time.perf_counter() - start) / num_iterations
        print(f"FP16 output shape: {out_fp16.shape}")
        print(f"FP16 inference time: {fp16_time * 1000:.2f} ms per batch")
        print(f"FP16 throughput: {batch_size / fp16_time:.1f} frames/sec")

        # Speedup
        speedup = fp32_time / fp16_time
        print(f"\nFP16 Speedup: {speedup:.2f}x")

        # Memory comparison
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model_fp32(inp_fp32)
        fp32_memory = torch.cuda.max_memory_allocated() / 1024**2

        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model_fp16(inp_fp16)
        fp16_memory = torch.cuda.max_memory_allocated() / 1024**2

        print(f"\nMemory usage:")
        print(f"FP32: {fp32_memory:.1f} MB")
        print(f"FP16: {fp16_memory:.1f} MB")
        print(f"Memory reduction: {(1 - fp16_memory / fp32_memory) * 100:.1f}%")

        # Accuracy comparison (output difference)
        with torch.no_grad():
            out_fp32 = model_fp32(inp_fp32)
            out_fp16_as_fp32 = model_fp16(inp_fp16).float()
            # Compare argmax outputs (what we actually use for ball detection)
            argmax_fp32 = out_fp32.argmax(dim=1)
            argmax_fp16 = out_fp16_as_fp32.argmax(dim=1)
            match_rate = (argmax_fp32 == argmax_fp16).float().mean().item()
            print(f"\nArgmax match rate: {match_rate * 100:.2f}%")
    else:
        print("\nFP16 benchmarking skipped (requires CUDA)")
