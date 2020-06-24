# Triplet Attention

<p float="left">
  <img src ="figures/triplet.png"  width="1000"/>
</p>

Authors - Diganta Misra <sup>1†</sup>, Trikay Nalamada <sup>1,2†</sup>, Ajay Uppili Arasanipalai <sup>1,3†</sup>, Qibin Hou <sup>4</sup>

1 - [Landskape](https://www.landskape.org/)     2. [IIT Guwahati](http://www.iitg.ac.in/)   3. [University of Illinois, Urbana Champaign](https://illinois.edu/)   4. [National University of Singapore](http://www.nus.edu.sg/)

† - Denotes Equal Contribution

<p float="left">
  <img src ="figures/comp.png"  width="1000"/>
</p>
<p>
    <em>Figure 1. (a). Squeeze Excitation Block. (b). Convolution Block Attention Module (CBAM) (Note - GMP denotes - Global Max Pooling). (c). Global Context (GC) block. (d). Triplet Attention (ours). </em>
</p>


<p float="left">
  <img src ="figures/grad.png"  width="1000"/>
</p>
<p>
    <em>Figure 2. GradCAM and GradCAM++ comparisons for ResNet-50 based on sample images from ImageNet dataset. </em>
</p>


## Pretrained Models:

### ImageNet:

|Model|Parameters|GFLOPs|Top-1 Error|Top-5 Error|Weights|
|:---:|:---:|:---:|:---:|:---:|:---:|
|ResNet-18 + Triplet Attention (k = 3)|11.69 M|1.823|**29.67%**|**10.42%**|[Google Drive](https://drive.google.com/file/d/1p3_s2kA5NFWqCtp4zvZdc91_2kEQhbGD/view?usp=sharing)|
|ResNet-50 + Triplet Attention (k = 7)|25.56 M|4.169|**22.52%**|**6.326%**|[Google Drive](https://drive.google.com/open?id=1ptKswHzVmULGbE3DuX6vMCjEbqwUvGiG)|
|ResNet-50 + Triplet Attention (k = 3)|25.56 M|4.131|**23.88%**|**6.938%**|[Google Drive](https://drive.google.com/open?id=1W6aDE6wVNY9NwgcM7WMx_vRhG2-ZiMur)|
|MobileNet v2 + Triplet Attention (k = 3)|3.506 M|0.322|**27.38%**|**9.23%**|[Google Drive](https://drive.google.com/file/d/1KIlqPBNLHh4qkdxyojb5gQhM5iB9b61_/view?usp=sharing)|
|MobileNet v2 + Triplet Attention (k = 7)|3.51 M||**28.01%**|**9.516%**|[Google Drive](https://drive.google.com/file/d/14iNMa7ygtTwsULsAydKoQuTkJ288hfKs/view?usp=sharing)|

### MS-COCO:

#### Object Detection:

|Backbone|Detectors|AP|AP<sub>50</sub>|AP<sub>75</sub>|AP<sub>S</sub>|AP<sub>M</sub>|AP<sub>L</sub>|Weights|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|ResNet-50 + Triplet Attention (k = 7)|Faster R-CNN|**39.2**|**60.8**|**42.3**|**23.3**|**42.5**|**50.3**|[Google Drive](https://drive.google.com/file/d/1Wq_B-C9lU9oaVGD3AT_rgFcvJ1Wtrupl/view?usp=sharing)|
|ResNet-50 + Triplet Attention (k = 7)|RetinaNet|**38.2**|**58.5**|**40.4**|**23.4**|**42.1**|**48.7**|[Google Drive](https://drive.google.com/file/d/1Wo-l_84xxuRwB2EMBJUxCLw5mhc8aAgI/view?usp=sharing)|
|ResNet-50 + Triplet Attention (k = 7)|Mask RCNN|**39.8**|**61.6**|**42.8**|**24.3**|**42.9**|**51.3**|[Google Drive](https://drive.google.com/file/d/18hFlWdziAsK-FB_GWJk3iBRrtdEJK7lf/view?usp=sharing)|

#### Instance Segmentation

|Backbone|Detectors|AP|AP<sub>50</sub>|AP<sub>75</sub>|AP<sub>S</sub>|AP<sub>M</sub>|AP<sub>L</sub>|Weights|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|ResNet-50 + Triplet Attention (k = 7)|Mask RCNN|**35.8**|**57.8**|**38.1**|**18**|**38.1**|**50.7**|[Google Drive](https://drive.google.com/file/d/18hFlWdziAsK-FB_GWJk3iBRrtdEJK7lf/view?usp=sharing)|
