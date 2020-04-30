# PCAM - Permuted Convolution Attention Module

Authors - Diganta Misra <sup>1†</sup>, Trikay Nalamada <sup>1,2†</sup>, Ajay Uppili Arasanipalai <sup>1,3†</sup>

1 - [Landskape](https://www.landskape.org/)     2. [IIT Guwahati](https://rose.ntu.edu.sg/Pages/Home.aspx)   3. [University of Illinois, Urbana Champaign](https://illinois.edu/)

† - Denotes Equal Contribution

<div style="text-align:center"><img src ="figures/pbam.png"  width="1000"/></div>
<p>
    <em>Figure 1. The Proposed Permuted Convolution Attention Module (PCAM)</em>
</p>

<div style="text-align:center"><img src ="figures/spatial.png"  width="1000"/></div>
<p>
    <em>Figure 2. Spatial Gate of PCAM</em>
</p>

## Pretrained Models:

|Model|Parameters|GFLOPs|Top-1 Error|Top-5 Error|Weights|
|:---:|:---:|:---:|:---:|:---:|:---:|
|ResNet-50 + PCAM (k = 7)|25.56 M|4.169|**22.662%**|**6.478%**|[Google Drive](https://drive.google.com/file/d/1wjQgkdqkUhnk_USq9e_fDwy62s64_Sq-/view)|
|ResNet-50 + PCAM (k = 3)<sup>*</sup>|25.56 M|4.131|**22.54%**|**6.324%**||

<sup>*</sup> Model was trained for 98 epochs. 
