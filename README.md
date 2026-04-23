# PointNet for CFD (Computational Fluid Dynamics)

![pic](./CFDpointNet.png)

**Point-Cloud Deep Learning for Prediction of Fluid Flow Fields on Irregular Geometries (Supervised Learning)**

**Authors:** Ali Kashefi (kashefi@stanford.edu) and Davis Rempe (drempe@stanford.edu)<br>
**Description:** Implementation of PointNet for *supervised learning* of computational mechanics on domains with irregular geometries <br>

**Citation** <br>
If you use the code, please cite the following journal paper: <br>
**[A point-cloud deep learning framework for prediction of fluid flow fields on irregular geometries](https://aip.scitation.org/doi/full/10.1063/5.0033376)**

    @article{kashefi2021PointNetCFD, 
      author = {Kashefi, Ali  and Rempe, Davis  and Guibas, Leonidas J.}, 
      title = {A point-cloud deep learning framework for prediction of fluid flow fields on irregular geometries},
      journal = {Physics of Fluids},
      volume = {33}, 
      number = {2}, 
      pages = {027104},
      year = {2021},
      doi = {10.1063/5.0033376}}

**Abstract** <br>
We present a novel deep learning framework for flow field predictions in irregular domains when the solution is a function of the geometry of either the domain or objects inside the domain. Grid vertices in a computational fluid dynamics (CFD) domain are viewed as point clouds and used as inputs to a neural network based on the PointNet architecture, which learns an end-to-end mapping between spatial positions and CFD quantities. Using our approach, (i) the network inherits desirable features of unstructured meshes (e.g., fine and coarse point spacing near the object surface and in the far field, respectively), which minimizes network training cost; (ii) object geometry is accurately represented through vertices located on object boundaries, which maintains boundary smoothness and allows the network to detect small changes between geometries and (iii) no data interpolation is utilized for creating training data; thus accuracy of the CFD data is preserved. None of these features are achievable by extant methods based on projecting scattered CFD data into Cartesian grids and then using regular convolutional neural networks. Incompressible laminar steady flow past a cylinder with various shapes for its cross section is considered. The mass and momentum of predicted fields are conserved. We test the generalizability of our network by predicting the flow around multiple objects as well as an airfoil, even though only single objects and no airfoils are observed during training. The network predicts the flow fields hundreds of times faster than our conventional CFD solver, while maintaining excellent to reasonable accuracy.

**Download the Full Data** <be>

You might use the following batch command to download the full dataset (a NumPy array, approximately 42MB in size).

```bash
wget https://web.stanford.edu/~kashefi/data/CFDdata.npy
```
**PointNet for CFD in PyTorch** <be>

The PyTorch version of the code can be downloaded from the following link:

[PyTorch Version](https://github.com/Ali-Stanford/KAN_PointNet_CFD/blob/main/others/PointNetMLP_alternative.py)

**PointNet for CFD using KAN (Kolmogorov-Arnold Networks)** <be>

Implementation of PointNet using KANs instead of MLPs can be found here:

[KAN PointNet for CFD](https://github.com/Ali-Stanford/KAN_PointNet_CFD/blob/main/others/PointNetKAN_alternative.py)

**Questions?** <br>
If you have any questions or need assistance, please do not hesitate to contact Ali Kashefi (kashefi@stanford.edu) via email. 

**About the Author** <br>
Please see the author's website: [Ali Kashefi](https://web.stanford.edu/~kashefi/) 
