# CT Reconstruction: CPU/GPU FDK and Katsevich

A C++/CUDA implementation of cone-beam CT reconstruction for helical scanning, providing CPU and GPU implementations of:

- **FDK reconstruction**
- **Katsevich reconstruction**
- **Flat (equidistant) detector geometry**
- **Equiangular detector geometry**

The project uses consistent geometry conventions and data layouts across CPU and GPU implementations so that both backends can be validated stage by stage.

> This repository is intended for research, algorithm development, and engineering validation. Geometry, normalization, detector calibration, and projection preprocessing should be verified for each physical CT system before production use.

---

## Features

- CPU FDK reconstruction
- CUDA-accelerated FDK reconstruction
- CPU Katsevich helical reconstruction
- CUDA-accelerated Katsevich reconstruction
- Flat and equiangular detector support
- Positive or negative rotation direction through `angleStep`
- Positive or negative helical travel through `pitch`
- Configurable source starting position through `zStart`
- Sub-pixel detector V-center calibration through `detectorVCenterOffsetPix`
- Shared CPU/GPU projection geometry and bilinear interpolation helpers
- Spatial-domain ramp filtering with common CT windows
- PI-line lookup table acceleration for Katsevich reconstruction
- CPU/GPU implementations designed to follow the same mathematical conventions

---

## 1. Geometry Convention

The rotation axis is the world **z-axis**.

For projection view `i`, the source angle is

$$
\beta_i = i\,\Delta\beta
$$

where

```text
Δβ = angleStep
```

and `angleStep` may be positive or negative.

The axial source displacement per view is

$$
\Delta z =
\frac{\mathrm{pitch}\,|\Delta\beta|}{2\pi}
$$

so the source z coordinate is

$$
z_s(i) =
zStart +
i\frac{\mathrm{pitch}\,|\Delta\beta|}{2\pi}.
$$

This convention separates the two directions:

- `angleStep > 0` / `< 0` controls the **rotation direction**.
- `pitch > 0` / `< 0` controls the **z travel direction as the view index increases**.
- `zStart` is the source z position at view 0.

Examples:

```text
zStart < 0, pitch > 0  -> scan from lower z toward higher z
zStart > 0, pitch < 0  -> scan from higher z toward lower z
```

provided that the projection ordering matches the specified geometry.

### Katsevich signed pitch

Katsevich formulas are expressed as functions of the continuous angle `β`. The implementation therefore defines

$$
P_\beta =
\mathrm{pitch}\,
\frac{|\mathrm{angleStep}|}{\mathrm{angleStep}}
$$

and

$$
h_\beta =
\frac{P_\beta}{2\pi}.
$$

The continuous source trajectory is then

$$
z_s(\beta) =
zStart + h_\beta\beta.
$$

This produces the same source-z trajectory as the view-based formula above.

---

## 2. Detector Geometry

Two detector models are supported.

### 2.1 Flat / equidistant detector

For a voxel `(x,y,z)` and source angle `β`,

$$
den =
SID - x\cos\beta - y\sin\beta.
$$

The ideal detector coordinates are

$$
u =
\frac{SDD}{den}
\left(-x\sin\beta+y\cos\beta\right),
$$

$$
v =
\frac{SDD}{den}
\left(z-z_s\right).
$$

The corresponding detector indices are

$$
u_{idx} =
\frac{u}{du}
+
\frac{nDetU-1}{2},
$$

$$
v_{idx} =
\frac{v}{dv}
+
\frac{nDetV-1}{2}
+
detectorVCenterOffsetPix.
$$

### 2.2 Equiangular detector

For the equiangular geometry, `du` represents angular detector spacing in **radians per pixel** rather than millimetres per pixel.

Define

$$
t =
-x\sin\beta+y\cos\beta.
$$

The horizontal fan angle is

$$
\gamma =
\mathrm{atan2}(t,den).
$$

Then

$$
u_{idx} =
\frac{\gamma}{du}
+
\frac{nDetU-1}{2}.
$$

The vertical detector coordinate is computed from the source-to-voxel transverse geometry and mapped to the detector V index using the same V-offset convention.

---

## 3. Detector V-Center Offset

`detectorVCenterOffsetPix` represents a constant sub-pixel displacement between the theoretical detector center and the actual detector array center.

It should normally be stored as a `float`, because calibration may produce fractional-pixel offsets.

The convention is

$$
v_{idx} =
\frac{v_{physical}}{dv}
+
v_{center}
+
offset,
$$

with

$$
v_{center} =
\frac{nDetV-1}{2}.
$$

The inverse mapping is

$$
v_{physical} =
(v_{idx}-v_{center}-offset)\,dv.
$$

Therefore:

```text
physical V -> detector index : + detectorVCenterOffsetPix
detector index -> physical V : - detectorVCenterOffsetPix
```

A constant offset corrects a translated detector coordinate origin. It does **not** model detector tilt, gantry wobble, or other view-dependent pose errors, and it cannot recover rays that were not measured because of detector truncation.

---

## 4. Data Layout

All CPU and GPU implementations use the same flattened memory layout.

### Projection data

Logical shape:

```text
[nViews][nDetV][nDetU]
```

Linear index:

```cpp
index =
    view * nDetV * nDetU
    + iv * nDetU
    + iu;
```

Expected input size:

```cpp
nViews * nDetV * nDetU
```

### Reconstruction volume

Logical shape:

```text
[nz][ny][nx]
```

Linear index:

```cpp
index =
    iz * ny * nx
    + iy * nx
    + ix;
```

The current reconstruction grid is centered at the world origin:

$$
x(ix)=
\left(-\frac{nx}{2}+ix+\frac12\right)dx,
$$

$$
y(iy)=
\left(-\frac{ny}{2}+iy+\frac12\right)dy,
$$

$$
z(iz)=
\left(-\frac{nz}{2}+iz+\frac12\right)dz.
$$

If an ROI is not centered at the world origin, reconstruction-center offsets should be added explicitly to the voxel coordinates. Do not use `detectorVCenterOffsetPix` to represent an object or ROI displacement.

---

## 5. Input Projection Requirements

`Reconstruct()` expects projection values that are already suitable for filtered backprojection.

The reconstruction module itself does not perform the complete raw-detector preprocessing chain. Raw intensity data may require upstream operations such as:

- detector offset correction,
- air/reference normalization,
- `-log(I/I0)`,
- bad-pixel correction,
- scatter correction,
- beam-hardening correction,
- other system-specific calibration.

The exact preprocessing chain depends on the acquisition system.

---

# 6. FDK Reconstruction

FDK is implemented as a helical FDK-style approximation.

## Processing pipeline

```text
Input projection
      |
      v
Geometry pre-weighting
      |
      v
1-D filtering along detector U
      |
      v
Select views around the voxel z position
      |
      v
Voxel -> detector projection
      |
      v
Bilinear interpolation
      |
      v
Backprojection weighting
      |
      v
Output volume
```

## 6.1 Flat-detector pre-weighting

For physical detector coordinates `(u,v)`,

$$
w_{pre} =
\frac{SDD}
{\sqrt{SDD^2+u^2+v^2}}.
$$

The weighted projection is

$$
p_w = p\,w_{pre}.
$$

## 6.2 Ramp filtering

For the flat detector, the discrete ramp kernel is

$$
h[0] =
\frac{1}{4\,du^2},
$$

$$
h[k] =
-\frac{1}{\pi^2 k^2 du^2},
\qquad k\ \text{odd},
$$

and

$$
h[k]=0
$$

for non-zero even `k`.

The implementation supports common windows such as:

- Ram-Lak
- Shepp-Logan
- Cosine
- Hamming
- Hann

For equiangular data, the angular detector spacing is used in the corresponding discrete filter formulation.

## 6.3 Flat-detector backprojection

For a filtered projection sample `q`,

$$
w_{bp} =
\frac{SID^2}{den^2}.
$$

The implemented accumulation is

$$
f(\mathbf{x})
\mathrel{+}=
\frac12
w_{bp}\,
q\,
|\Delta\beta|.
$$

The factor `1/2` is part of the normalization used by this implementation and should not be changed independently of the filter normalization and angular integration convention.

## 6.4 CPU implementation

Main class:

```cpp
CpuFDKRecon
```

Main stages:

```text
Filter::GetFilter()
        |
        v
CpuFDKRecon::ParallelPreprocessProj()
        |
        v
CpuFDKRecon::BackProjectOMP()
```

CPU parallelism is primarily implemented with `std::thread`.

## 6.5 GPU implementation

Main class:

```cpp
GpuFDKRecon
```

Main CUDA stages:

```text
proj_geom_filted  -> one thread per projection pixel
proj_filtered     -> one thread per output projection pixel
FDKKernel         -> one thread per voxel
```

CPU and GPU implementations are intended to use the same geometry, weighting, interpolation, and normalization.

---

# 7. Katsevich Reconstruction

Katsevich reconstruction uses the PI-line formulation for helical cone-beam CT.

## Processing pipeline

```text
Input projection
      |
      v
G1 directional derivative
      |
      v
K-line resampling: G1 -> G2
      |
      v
Hilbert filtering along U: G2 -> G3
      |
      v
Inverse K-line mapping
      |
      v
Filtered projection
      |
      v
PI-line interval
      |
      v
PI-LUT lookup
      |
      v
Backprojection over the PI interval
      |
      v
Output volume
```

## 7.1 K-line geometry

For the flat detector, define

$$
q(\psi) =
\frac{\psi}{\tan\psi}.
$$

Near zero, the implementation uses

$$
\frac{\psi}{\tan\psi}
\approx
1-\frac{\psi^2}{3}-\frac{\psi^4}{45}.
$$

The flat-detector K-line coordinate is

$$
v_\kappa =
\frac{D P_\beta}{2\pi R}
\left[
\psi+
\frac{\psi}{\tan\psi}\frac{u}{D}
\right],
$$

where

```text
D = SDD
R = SID
```

and `P_beta` is the signed pitch defined above.

The K-line V coordinate is mapped to detector index as

$$
v_{idx} =
\frac{v_\kappa}{dv}
+
\frac{nDetV-1}{2}
+
detectorVCenterOffsetPix.
$$

## 7.2 G1 directional derivative

For flat detector data,

$$
G_1 =
\frac{D}
{\sqrt{D^2+u^2+v^2}}
\left[
g_\beta
+
\frac{D^2+u^2}{D}g_u
+
\frac{uv}{D}g_v
\right].
$$

The derivatives are evaluated by centered finite differences.

Important:

```text
dbeta = angleStep
```

must keep its sign.

Do **not** replace it with `abs(angleStep)` in the angular derivative.

## 7.3 G2

`G1` is interpolated onto the K-lines using the precomputed K-line detector V coordinate.

Out-of-detector K-line samples are set to zero. Zero padding does not reconstruct missing physical measurements.

## 7.4 G3 and Hilbert filtering

For the flat detector, the discrete Hilbert kernel is based on

$$
H(k) =
\frac{2}{\pi\,k\,du}
$$

for the corresponding odd offsets.

For equiangular data, the denominator uses

$$
\sin(k\,du).
$$

The filtered result is mapped from K-line coordinates back to the original detector grid through the inverse-`psi` lookup table.

For the equiangular implementation, the final filtered projection also includes the required `cos(alpha)` factor.

---

## 7.5 PI lines

For each voxel, Katsevich reconstruction integrates only over its PI interval

$$
[\beta_b,\beta_t].
$$

The source helix is

$$
z_s(\beta) =
zStart+h_\beta\beta.
$$

The PI-line equation is solved numerically.

Because PI geometry is periodic over one helical turn, the implementation precomputes a lookup table over one pitch.

### PI-LUT representation

Each LUT entry stores

$$
midRelative =
\frac{\beta_b+\beta_t}{2}
-
phase
$$

and the positive half-width

$$
\delta =
\frac{\beta_t-\beta_b}{2}.
$$

For an arbitrary voxel z position,

$$
\beta_z =
\frac{z-zStart}{h_\beta}.
$$

After wrapping `beta_z` into one `2*pi` phase and interpolating neighboring LUT entries,

$$
\beta_{mid} =
\beta_z+midRelative,
$$

$$
\beta_b =
\beta_{mid}-\delta,
$$

$$
\beta_t =
\beta_{mid}+\delta.
$$

The LUT construction and lookup must use the same signed `h_beta`.

---

## 7.6 Katsevich backprojection

PI endpoints are converted to view indices with the signed angular step:

$$
i_b =
\frac{\beta_b}{\Delta\beta},
\qquad
i_t =
\frac{\beta_t}{\Delta\beta}.
$$

The integration interval is

$$
view_b =
\left\lceil
\min(i_b,i_t)
\right\rceil,
$$

$$
view_t =
\left\lfloor
\max(i_b,i_t)
\right\rfloor.
$$

The implementation excludes the first and last views because the angular derivative uses centered differences.

For each valid view,

$$
f(\mathbf{x})
\mathrel{+}=
\frac{q}{2\pi\,den}
|\Delta\beta|.
$$

---

## 7.7 CPU implementation

Main class:

```cpp
CpuKatsevichRecon
```

Important functions:

```text
SignedPitchPerBeta / HelixSlopePerBeta
calculate_kLines / calculate_kLines_equal_angle
calculate_inverse_Psi_index
construct_hilbert_kernel
FilterProj
calculate_PI_line
build_PI_LUT
calculate_pi
BackProject
```

## 7.8 GPU implementation

Main class:

```cpp
GpuKatsevichRecon
```

Important kernels/functions:

```text
geom_filter          -> one thread per original projection pixel
cal_G2               -> one thread per (view, psi, u)
cal_G3_filter        -> one thread per (view, psi, u)
cal_filtedProj       -> one thread per detector pixel
build_PI_LUT_kernel  -> one thread per (phase, x, y)
BackProjKern         -> one thread per voxel
```

The GPU implementation is intended to reproduce the CPU mathematical pipeline while moving the expensive filtering, PI-LUT construction, and backprojection operations to CUDA.

---

# 8. Basic Usage

All four reconstruction classes use the same high-level interface:

```cpp
CpuFDKRecon
GpuFDKRecon
CpuKatsevichRecon
GpuKatsevichRecon
```

A typical setup is:

```cpp
CTGeometry geo{};

geo.SID = 1000.0f;      // mm
geo.SDD = 1500.0f;      // mm

geo.nDetU = ...;
geo.nDetV = ...;

geo.du = ...;            // flat: mm/pixel
                         // equiangular: rad/pixel
geo.dv = ...;            // mm/pixel

geo.nViews = ...;
geo.angleStep = ...;     // rad/view; may be negative

geo.pitch = ...;         // mm/turn; may be negative
geo.zStart = ...;        // source z at view 0

geo.detectorVCenterOffsetPix = ...;

geo.scan_type = 0;       // 0: flat/equidistant
                         // 1: equiangular

geo.nx = ...;
geo.ny = ...;
geo.nz = ...;

geo.dx = ...;            // mm
geo.dy = ...;            // mm
geo.dz = ...;            // mm

recon_para recp{};
recp.filter_name = "RamLak";  // used by FDK
recp.cuda_device = 0;         // used by GPU implementations
```

Prepare the projection vector using the layout

```text
[view][v][u]
```

and reconstruct:

```cpp
std::vector<float> proj = ...;
std::vector<float> vol;

// Choose one:
CpuFDKRecon recon(geo, recp);
// GpuFDKRecon recon(geo, recp);
// CpuKatsevichRecon recon(geo, recp);
// GpuKatsevichRecon recon(geo, recp);

recon.Reconstruct(proj, vol);
```

The output volume is stored as

```text
[z][y][x]
```

Before reconstruction, it is recommended to verify

```cpp
const size_t expected =
    static_cast<size_t>(geo.nViews)
    * geo.nDetV
    * geo.nDetU;

if (proj.size() != expected)
{
    throw std::invalid_argument(
        "Projection size does not match geometry");
}
```

---

# 9. Choosing an Algorithm

### FDK

Use FDK when:

- reconstruction speed is important,
- an approximate cone-beam reconstruction is acceptable,
- the helical cone angle and pitch are within a regime where FDK quality is sufficient,
- a simpler reconstruction path is preferred.

### Katsevich

Use Katsevich when:

- an exact helical cone-beam formulation is required,
- the acquisition geometry satisfies the assumptions of the algorithm,
- the required PI-line/K-line data are actually measured,
- additional computation and memory are acceptable.

A finite detector may truncate rays required by Katsevich. Geometry calibration or zero padding cannot reconstruct measurements that were not acquired.

---

# 10. CPU/GPU Validation

Do not validate CPU and GPU implementations only from the final reconstructed volume.

The recommended strategy is to compare intermediate stages.

### FDK

```text
CPU pre-weighted projection
        vs
GPU pre-weighted projection

CPU filtered projection
        vs
GPU filtered projection

CPU reconstructed volume
        vs
GPU reconstructed volume
```

### Katsevich

Compare:

```text
K-lines
inverse-psi LUT
G1
G2
G3
filtered projection
PI-LUT
final volume
```

Useful error metrics include:

```text
maximum absolute error
mean absolute error
relative L2 error
```

Small differences are expected from:

- CPU `std::sin/std::cos` versus CUDA `sinf/cosf`,
- different floating-point intermediate precision,
- different parallel execution order.

Large sign changes, spatial shifts, large zero regions, or order-of-magnitude differences should not be attributed to ordinary floating-point error.

---

# 11. Important Implementation Notes

- `angleStep` may be negative.
- `pitch` may be negative.
- The final angular integration uses `abs(angleStep)`.
- Katsevich angular derivatives use the **signed** `angleStep`.
- Katsevich `h_beta` must be derived from signed pitch.
- `detectorVCenterOffsetPix` is added when converting physical V to detector index and subtracted when converting detector index to physical V.
- For flat geometry, `du` is normally in mm/pixel.
- For equiangular geometry, `du` is in rad/pixel.
- `dv` remains a physical V spacing.
- `SID` and `SDD` should be represented as floating-point values.
- `SDD` is the source-to-detector-plane distance used by the ideal detector geometry, not a slanted distance to an arbitrarily shifted array center.
- Missing detector rays remain missing data.
- If detector offset varies with view, or if the detector is tilted, a single constant V offset is not sufficient.
- The current angle convention assumes

$$
\beta_i = i\,angleStep.
$$

If arbitrary acquisition start angles or explicitly reversed projection arrays are required, introduce an `angleStart` parameter consistently throughout the geometry.

---

# 12. Requirements

### CPU

- C++17-capable compiler
- Standard C++ threading support

### GPU

- NVIDIA CUDA-capable GPU
- NVIDIA CUDA Toolkit / NVCC

CUDA source files (`.cu`) must be compiled by NVCC and linked with the C++ reconstruction code.

Exact build commands depend on the build configuration supplied with the repository.

---

# 13. Performance Notes

### FDK

The current direct spatial filtering can become expensive for large detector widths. Possible optimizations include:

- FFT-based filtering,
- CUDA shared-memory filtering,
- CUDA texture objects for projection interpolation,
- caching geometry-independent filters.

### Katsevich

The main expensive operations are:

- direct Hilbert convolution,
- PI-line solution,
- backprojection.

The PI-LUT is particularly important because it replaces repeated per-voxel nonlinear PI-line solves with lookup and interpolation over one helical pitch.

Approximate PI-LUT memory usage is

$$
N_{\mathrm{bytes}} =
nx\cdot ny\cdot N_{\mathrm{phase}}\cdot 2\cdot \mathrm{sizeof}(\mathrm{float}).
$$

Using `float2(midRelative, delta)` is a natural representation on CUDA.

---


## Notes for Contributors

When modifying the reconstruction code, keep the following conventions synchronized between CPU and GPU:

1. coordinate system,
2. projection/volume memory layout,
3. pitch and rotation signs,
4. detector offset convention,
5. filter normalization,
6. interpolation rules,
7. PI-line/K-line definitions,
8. final angular integration.

For this project, many difficult CPU/GPU discrepancies come from inconsistent geometry conventions rather than from the reconstruction formula itself.
