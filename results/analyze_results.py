from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
SHAPE = (128, 165, 165)  # [z, y, x], float32

pairs = [
    ("FDK", 0), ("FDK", 1),
    ("Katsevich", 0), ("Katsevich", 1),
]

metrics = []
for algo, detector_type in pairs:
    cpu_path = ROOT / f"recon_type{detector_type}_{algo}_CPU_165x165x128.raw"
    gpu_path = ROOT / f"recon_type{detector_type}_{algo}_GPU_165x165x128.raw"
    cpu = np.fromfile(cpu_path, dtype=np.float32).reshape(SHAPE)
    gpu = np.fromfile(gpu_path, dtype=np.float32).reshape(SHAPE)
    diff = gpu.astype(np.float64) - cpu.astype(np.float64)
    denom = np.linalg.norm(cpu.astype(np.float64).ravel())
    metrics.append((algo, detector_type, float(np.max(np.abs(diff))),
                    float(np.mean(np.abs(diff))), float(np.linalg.norm(diff.ravel()) / denom)))

    z = SHAPE[0] // 2
    lo, hi = np.percentile(cpu[z], [1, 99.5])
    geom = "flat" if detector_type == 0 else "equiangular"
    def gray(arr):
        x=np.clip((arr-lo)/(hi-lo+1e-12),0,1)
        return Image.fromarray((x*255).astype(np.uint8),"L").resize((330,330),Image.Resampling.NEAREST).convert("RGB")
    dmax=max(np.percentile(np.abs(diff[z]),99.5),1e-12)
    q=np.clip(diff[z]/dmax,-1,1)
    rgb=np.zeros((*q.shape,3),dtype=np.uint8)
    rgb[...,0]=(np.clip(q,0,1)*255).astype(np.uint8)
    rgb[...,2]=(np.clip(-q,0,1)*255).astype(np.uint8)
    rgb[...,1]=0
    dimg=Image.fromarray(rgb,"RGB").resize((330,330),Image.Resampling.NEAREST)
    canvas=Image.new("RGB",(1050,405),"white"); draw=ImageDraw.Draw(canvas)
    draw.text((20,12),f"{algo} reconstruction - {geom} detector - central axial slice",fill="black")
    for i,(im,label) in enumerate([(gray(cpu[z]),"CPU"),(gray(gpu[z]),"GPU"),(dimg,"GPU - CPU difference")]):
        x=15+i*345; canvas.paste(im,(x,55)); draw.text((x,385),label,fill="black")
    canvas.save(ROOT/f"result_{algo.lower()}_type{detector_type}.png")

with (ROOT / "cpu_gpu_metrics.csv").open("w", encoding="utf-8") as f:
    f.write("algorithm,detector_type,max_abs_error,mean_abs_error,relative_l2_error\n")
    for row in metrics:
        f.write(",".join([row[0], str(row[1]), *(f"{v:.9g}" for v in row[2:])]) + "\n")

times = {
    ("FDK", 0): (47.199, 3.093),
    ("FDK", 1): (40.787, 3.078),
    ("Katsevich", 0): (20.804, 2.355),
    ("Katsevich", 1): (12.674, 2.388),
}
with (ROOT / "performance_summary.csv").open("w", encoding="utf-8") as f:
    f.write("algorithm,detector_type,cpu_seconds,gpu_seconds,speedup\n")
    for (algo, dtype), (cpu, gpu) in times.items():
        f.write(f"{algo},{dtype},{cpu:.3f},{gpu:.3f},{cpu/gpu:.2f}\n")
