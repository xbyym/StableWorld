<p align="center">
  <h1 align="center">✨ <b>StableWorld</b></h1>
  <h3 align="center">Towards Stable and Consistent Long Interactive Video Generation</h3>
</p>

<p align="center">
  <a href="https://github.com/xbyym" target="_blank">Ying Yang<sup>1</sup></a>,
  <a href="https://scholar.google.com/citations?user=FkkaUgwAAAAJ&hl=en" target="_blank">Zhengyao Lv<sup>1,2</sup></a>,
  <a href="https://tianlinn.com/" target="_blank">Tianlin Pan<sup>1,3</sup></a>,
  <a href="https://haofanwang.github.io/" target="_blank">Haofan Wang<sup>4</sup></a>,
  <a href="https://binxinyang.github.io/" target="_blank">Binxin Yang<sup>5</sup></a>,
  <a href="https://openreview.net/profile?id=~Hubery_Yin1" target="_blank">Hubery Yin<sup>6</sup></a>,
  <a href="https://scholar.google.com/citations?user=WDJL3gYAAAAJ&hl=zh-CN" target="_blank">Chen Li<sup>6</sup></a>,
  <a href="https://liuziwei7.github.io/" target="_blank">Ziwei Liu<sup>6</sup></a>,
  <a href="https://chenyangsi.top/" target="_blank">Chenyang Si<sup>1</sup></a>
</p>


<p align="center">
  <sup>1</sup>PRLab, Nanjing University &nbsp;&nbsp;·&nbsp;&nbsp;
  <sup>2</sup>The University of Hong Kong &nbsp;&nbsp;·&nbsp;&nbsp;
  <sup>3</sup>University of Chinese Academy of Sciences &nbsp;&nbsp;·&nbsp;&nbsp;
  <sup>4</sup>LibLib.ai &nbsp;&nbsp;·&nbsp;&nbsp;
  <sup>5</sup>WeChat, Tencent Inc. &nbsp;&nbsp;·&nbsp;&nbsp;
  <sup>6</sup>Nanyang Technological University
</p>


<!-- 如果你有 equal contribution / corresponding author，就打开这一段 -->
<!--
<p align="center">
  <i>* Equal Contribution &nbsp;&nbsp;&nbsp; † Corresponding Author</i>
</p>
-->

<p align="center">
  <a href="https://arxiv.org/abs/2601.15281">
    <img src="https://img.shields.io/badge/arXiv-Paper-red?style=flat&logo=arxiv" height="22">
  </a>
  <a href="https://sd-world.github.io/">
    <img src="https://img.shields.io/badge/Project-Page-blue?style=flat&logo=google-chrome" height="22">
  </a>
  <a href="https://github.com/Sd-World">
    <img src="https://img.shields.io/badge/Code-GitHub-black?style=flat&logo=github" height="22">
  </a>
</p>

---

## 🎥 Video Demo

<div align="center">

[![Video](https://github.com/xbyym/StableWorld/blob/main/file/new_demo.png?raw=1)](https://youtu.be/R5XXI-oAKVU)


</div>


> If the embedded YouTube player is blocked in your region/network, please visit the project page for mirrored videos.

---

## 🚀 Release

- [x] Paper released  
- [x] Project page released  
- [x] Demo videos released  
- [ ] Code coming soon  

---

## 🔥 Why Interactive Video Generation Becomes Unstable?

Long-horizon interactive video generation often suffers from **spatial drift** and **scene collapse**.  
We find that a major source of instability is **error accumulation, especially within the same scene**: small drifts accumulate under the same viewpoint, eventually leading to the collapse of the entire scene.

**This perspective is different from the commonly discussed error accumulation caused by the train–inference mismatch, and we find that scene collapse is largely driven by this factor.**

As illustrated below:

![Error accumulation under a fixed viewpoint](https://github.com/xbyym/StableWorld/blob/main/show1_01.png?raw=true)

Small drifts may seem negligible at first, but when repeatedly accumulated under the same viewpoint, they gradually grow and eventually lead to severe scene collapse.

**StableWorld** addresses this issue at the root by **continuously filtering out degraded frames** while **retaining geometrically consistent ones**, preventing drift from compounding over time.

**When the scene changes extremely rapidly (e.g., multiple visual changes per second), errors do not have sufficient time to accumulate, resulting in much less frequent scene collapse.**


---

## 🧩 The StableWorld Framework

**StableWorld** is a simple yet effective **Dynamic Frame Eviction Mechanism** that is **model-agnostic** and can be plugged into different interactive generation frameworks (e.g., Matrix-Game, Open-Oasis, Hunyuan-GameCraft) to improve **stability**, **temporal consistency**, and **generalization**.

<div align="center">

<img src="https://raw.githubusercontent.com/Sd-World/Sd-World.github.io/main/overnew_01.png" width="95%"/>

</div>

---

## 🎬 Visual Results

We provide extensive interactive demonstrations across multiple frameworks:

- **Matrix-Game 2.0**
- **Open-Oasis**
- **Hunyuan-GameCraft**
- **Ultra-long video generation (thousands of frames)**
- **Self-Forcing (autoregressive video)**

Please see the project page for full videos and side-by-side comparisons:  
👉 https://sd-world.github.io/

---

## 📚 Citation

If you find this work helpful, please consider citing:

```bibtex
@article{stableworld2026,
  title={StableWorld: Towards Stable and Consistent Long Interactive Video Generation},
  author={Ying Yang and Zhengyao Lv and Tianlin Pan and Haofan Wang and Binxin Yang and Hubery Yin and Chen Li and Ziwei Liu and Chenyang Si},
  journal={arXiv preprint arXiv:2601.15281},
  year={2026}
}
