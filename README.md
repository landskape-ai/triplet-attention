# PCAM - Permuted Convolution Attention Module

Authors - Diganta Misra <sup>1†</sup>, Trikay Nalamada <sup>1,2†</sup>, Ajay Uppili Arasanipalai <sup>1,3†</sup>

1 - [Landskape](https://www.landskape.org/)     2. [IIT Guwahati](http://www.iitg.ac.in/)   3. [University of Illinois, Urbana Champaign](https://illinois.edu/)

† - Denotes Equal Contribution

<div style="text-align:center"><img src ="figures/pbam.png"  width="1000"/></div>
<p>
    <em>Figure 1. The Proposed Permuted Convolution Attention Module (PCAM)</em>
</p>

<div style="text-align:center"><img src ="figures/spatial.png"  width="1000"/></div>
<p>
    <em>Figure 2. Spatial Gate of PCAM</em>
</p>

<p float="left">
  <img src ="figures/pcam(k=7).png"  width="350"/>
</p>
<p>
    <em>Figure 3. Training Curve of ResNet-50 + PCAM (k = 7) on ImageNet Classification. </em>
</p>


<p float="left">
  <img src ="figures/Vanilla_maps_small.png"  width="300"/>
   &emsp;
    &emsp;
    &emsp;
  <img src ="figures/PCAM_maps_small.png"  width="300"/>
</p>
<p>
    <em>Figure 4. Feature Maps comparison for pretrained Vanilla ResNet-50 (left) and pretrained ResNet-50 + PCAM (k = 7) (right). </em>
</p>


## Pretrained Models:

|Model|Parameters|GFLOPs|Top-1 Error|Top-5 Error|Weights|
|:---:|:---:|:---:|:---:|:---:|:---:|
|ResNet-50 + PCAM (k = 7)|25.56 M|4.169|**22.662%**|**6.478%**|[Google Drive](https://drive.google.com/file/d/1wjQgkdqkUhnk_USq9e_fDwy62s64_Sq-/view)|
|ResNet-50 + PCAM (k = 7)<sup>*</sup>|25.56 M|4.169|**22.52%**|**6.326%**|[Google Drive](https://drive.google.com/file/d/1UX3q8gaherNJMhsd20vsgPjLXUIbZdKS/view?usp=sharing)|

<sup>*</sup> Model was trained for 98 epochs. 
