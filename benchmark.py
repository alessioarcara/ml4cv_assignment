import torch
import timm
from models.decoder import DecoderWithFAM
from models.model import ChimeraSeg
import numpy as np

torch.backends.cuda.matmul.allow_tf32 = True


def main():
    model_name = "resnet18d"
    d = 128
    fpn_features = [64, 64, 128, 512]
    atrous_rates = (4, 8, 12)
    encoder = timm.create_model(
        model_name,
        features_only=True,
        pretrained=True,
        out_indices=(0, 1, 2, 4),
        output_stride=16,
    )
    decoder = DecoderWithFAM(
        fpn_features, 13, input_size=(320, 640), d=d, atrous_rates=atrous_rates
    )

    model = ChimeraSeg(encoder, decoder).to("cuda")
    opt_model = torch.compile(model, mode="default", fullgraph=True).to("cuda")

    model.eval()
    opt_model.eval()

    input_tensor = torch.rand((16, 3, 640, 360)).to("cuda")
    import time

    def benchmark(model, input_tensor, n_iter=30):
        times = []
        for _ in range(10):  # warm-up
            _ = model(input_tensor)
        torch.cuda.synchronize()
        for _ in range(n_iter):
            start = time.time()
            _ = model(input_tensor)
            torch.cuda.synchronize()
            end = time.time()
            times.append((end - start) * 1000)
        return np.mean(times), np.std(times)

    time_original = benchmark(model, input_tensor)
    time_optimized = benchmark(opt_model, input_tensor)

    print(f"Tempo originale: {time_original[0]:.3f} ms per iterazione")
    print(f"Tempo compilato: {time_optimized[0]:.3f} ms per iterazione")
    print(f"Speedup: {time_original[0] / time_optimized[0]:.2f}x")


if __name__ == "__main__":
    main()
