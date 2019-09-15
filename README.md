# NeurISTTOK

NeurISTTOK is essentially a Python class which allows the reconstruction of plasma emissivity profiles for the tokamak ISTTOK. Such reconstructions are performed with a transposed convolutional neural network trained with synthetic data.

<p align="center">
  <img width="300" height="300" src="figures/capa.png">
</p>

### Prerequisites

RaySTTOK uses the heavy machinery implemented on the Raysetc Python package. Thus, the main prerequisite is:

```
Raysect
```
Other required packages include:
```
Numpy
Matplotlib
Scipy
Random
```

### Installing

The first step to successfuly use RaySTTOK is the installation of the package [Raysect](https://raysect.github.io/documentation/installation.html).
Then, simply place the file "raysttok.py" and the folder "resources" on the desired working directory.

## Getting Started

The simulation of a synthetic Gaussian emissivity profile can be done with:
```
raysttok = RaySTTOK(reflections=True, pixel_samples=100)

raysttok.place_plasma(shape='gaussian', emissiv_gain=1e3, mean=[0.05, -0.05], cov=[[5e-4, 0], [0, 5e-4]])

raysttok.simulate_rays()
raysttok.plot_detectors()
raysttok.show_plots()
```

On the other hand, the computation of ISTTOK's projection matrix can be performed with:
```
raysttok = RaySTTOK(reflections=True, pixel_samples=10)
raysttok.get_proj_matrix(pixel_side=15, out_file="proj_matrix1")
raysttok.plot_line_matrix(line=0, mat_file="proj_matrix1.npy")
raysttok.show_plots()
```
A careful description of the implemented methods and their potential can be found on [RaySSTOK Wiki](https://github.com/RVACardoso/RaySTTOK/wiki/RaySTTOK-Wiki)

## Authors

* **R. V. A. Cardoso**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
