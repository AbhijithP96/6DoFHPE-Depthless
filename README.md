# 📌 6DoFHPE – Extending with Depth Prediction (DPE)

This project builds upon the existing 6DoFHPE pipeline, which estimates the 6 Degrees of Freedom Head Pose using RGB-D images. To eliminate the dependency on explicit depth input, this extension integrates a Depth Prediction Estimator (DPE) that generates depth maps directly from RGB images. The remaining pipeline remains unchanged, using the predicted depth in place of the actual depth to estimate the translation vector.

This implementation is part of the methodology developed during my Master’s Thesis, focused on enhancing the practicality and scalability of head pose estimation systems by removing the need for specialized depth cameras.

## Acknowledgement

This project builds upon prior research in 6DoF Head Pose Estimation (6DoFHPE) and Monocular Depth Estimation. I would like to thank the authors and contributors of the following works, which served as a foundation or reference in this implementation:

```
@article{algabri2024real,
  title={Real-time 6DoF full-range markerless head pose estimation},
  author={Algabri, Redhwan and Shin, Hyunsoo and Lee, Sungon},
  journal={Expert Systems with Applications},
  volume={239},
  pages={122293},
  year={2024},
  publisher={Elsevier}
}
```

```
@misc{Metric3D,
  author =       {Yin, Wei and Hu, Mu},
  title =        {OpenMetric3D: An Open Toolbox for Monocular Depth Estimation},
  howpublished = {\url{https://github.com/YvanYin/Metric3D}},
  year =         {2024}
}
```