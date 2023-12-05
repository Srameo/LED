# 噪声标定

## 先决条件

### 1. Rawpy

标定的代码使用了 rawpy 包，可根据此说明安装（https://pypi.org/project/rawpy/）

注意：如果无法在 Mac（m1/m2）上安装 rawpy，您可以参考这个issue（https://github.com/letmaik/rawpy/issues/171）

### 2. 数据

建议将待标定的数据按照不同的相机分别组织到各自的文件夹中。在每个相机文件夹内，应进一步根据不同的 ISO 值将数据分到几个子文件夹中。

## 数据拍摄与原理

我们分别使用了5种不同型号的相机分别拍摄了色卡（24个色块）和黑图。色卡如下图所示：

<p align="center">
  <img src='../.assets/calib/IMG_0483.jpg' alt='ICCV23_LED_LOGO' width='600px'/>
</p>

dark图像如下图所示：

<p align="center">
  <img src='../.assets/calib/IMG_0016.jpg' alt='ICCV23_LED_LOGO' width='600px'/>
</p>
其中色卡和黑图都选取了9个不同的ISO值，每个ISO值均拍摄了15张相同的图像。因此根据ELD[^1]中提到的公式:

$$\text{Var}(D) = K(K I)+\text{Var}\left(N_o\right)$$

相当于我们对每个ISO下的色卡图，都获取了24组不同的数据点，因此我们可以拟合出一条直线，并得到当前相机当前ISO下的K和信号无关噪声的方差，如下图：

<p align="center">
  <img src='../.assets/calib/1600_K1.png' alt='ICCV23_LED_LOGO' width='600px'/>
</p>

接着，我们根据拍摄的dark图像获取其他参数，首先，我们可以通过对整张图像求均值和方差得到Read Noise（假设服从TL分布或者高斯分布）与row噪声的和的分布参数；随后，我们可以分别获得当Read Noise服从TL分布或高斯分布这两种假设的参数。

最后，根据ELD[^1]中的公式：

$$\log (\sigma_{TL}) \mid \log (K) \sim \mathcal{N}(a_{TL} \log (K)+b_{TL}, \hat{\sigma}_{TL})$$

$$\log (\sigma_r) \mid \log (K) \sim \mathcal{N}(a_r \log (K)+b_r, \hat{\sigma}_r)$$

我们可以对每个相机在不同ISO下得到的K和分布参数作为数据点，拟合出两者之间的对数线性关系，例如下图：

<p align="center">
  <img src='../.assets/calib/log.png' alt='ICCV23_LED_LOGO' width='600px'/>
</p>

## 标定过程

对于标定过程，您可以通过直接按照主函数中给出的代码执行所有步骤一次完成，也可以根据您的需要分别执行每个步骤。这些步骤包括选择颜色块的位置、校准颜色块以获取 K 值、校准暗图像以获取其他参数，以及拟合 log(K) 和 log(方差)。

## 致谢

本代码参考了ELD和PMN的相关工作，在此向ELD[^1]和PMN[^2]的作者和贡献者致谢！

[^1]: 1https://arxiv.org/abs/2108.02158

[^2]: https://github.com/megvii-research/PMN/tree/TPAMI



