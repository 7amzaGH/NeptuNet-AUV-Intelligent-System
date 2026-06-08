# Limitations

This document summarizes the current limitations of NeptuNet. These limitations are intentionally stated to clarify the validated scope of the project and avoid overclaiming.

## 1. No Closed-Loop AUV Control

NeptuNet focuses on perception-side inspection.

The framework estimates visual and geometric information such as pipeline orientation, bubble activity, plume mask, probable leak source, and propagation direction. However, it does not implement or validate closed-loop control of an underwater vehicle.

The current work does not include:

* motion control,
* path tracking,
* station keeping,
* trajectory planning,
* vehicle dynamics,
* thruster control,
* closed-loop navigation validation.

## 2. No Full Underwater Autonomy Stack

NeptuNet should not be interpreted as a complete autonomous underwater robot system.

The current framework does not implement:

* SLAM,
* acoustic localization,
* global navigation,
* mission planning,
* underwater communication,
* multi-robot coordination,
* real-time robotic middleware integration.

It is a perception framework intended to support future underwater robotic inspection systems.

## 3. Monocular Vision Dependency

The current perception modules operate primarily on monocular underwater imagery.

This makes the framework lightweight and practical, but it also limits direct 3D interpretation. Image-plane source estimation and direction estimation should not be interpreted as full 3D leak localization.

Future work should combine visual perception with:

* depth sensors,
* sonar,
* acoustic positioning,
* stereo vision,
* inertial navigation,
* environmental sensors.

## 4. Dataset Scale and Environmental Diversity

The project uses multiple underwater datasets for pipeline perception, bubble analysis, and leak segmentation. However, underwater environments are highly diverse.

The current datasets do not fully cover all possible real-world conditions, such as:

* extreme turbidity,
* very low illumination,
* strong currents,
* heavy occlusion,
* diverse seabed types,
* different camera sensors,
* different pipeline materials,
* long-duration inspection missions.

Broader dataset expansion is needed for stronger deployment confidence.

## 5. Cross-Domain Generalization Remains Challenging

NeptuNet includes cross-domain evaluation, especially for pipeline perception and gas plume segmentation. However, underwater domain shift remains a major challenge.

Differences between controlled tank data, ROV footage, real offshore footage, camera quality, lighting, and water conditions can still affect performance.

Future work should investigate:

* domain adaptation,
* self-supervised learning,
* synthetic-to-real transfer,
* continual learning,
* uncertainty estimation.

## 6. Embedded Deployment Is Benchmark-Oriented

The project includes deployment-oriented benchmarking using ONNX, INT8 quantization, and Qualcomm RB3 Gen 2 NPU results.

However, the current validation does not represent a complete real-time onboard AUV deployment with all sensors, software, power constraints, and mission logic integrated together.

The embedded results should be understood as deployment feasibility evidence, not as full field deployment validation.

## 7. Bubble Analysis Depends on Visible Bubble Activity

The bubble-based early-warning module assumes that leak-related bubble activity is visible in the camera field of view.

Performance may degrade when:

* bubbles are outside the camera view,
* bubbles are visually weak,
* turbidity hides small bubbles,
* background particles resemble bubbles,
* plume structure is disrupted by currents,
* the camera is too far from the leak source.

## 8. Leak Source Estimation Is Image-Plane Based

The leak confirmation module estimates a probable source position from the visible plume mask. This is useful for visual inspection and operator support, but it is not equivalent to physical 3D source localization.

Accurate real-world source localization would require additional sensor information, camera calibration, depth estimation, and vehicle pose information.

## 9. Framework Integration Is Conceptual and Analytical

NeptuNet connects the three modules through a resource-aware architecture. However, the current main repository is a project hub and framework-level documentation repository.

Detailed code, datasets, demos, and experiments are maintained in the module repositories.

A complete runtime orchestrator integrating all modules into one onboard software stack remains future work.

## Summary

NeptuNet provides a strong perception-side foundation for underwater gas pipeline inspection, but its current contribution is not full underwater autonomy.

The validated contribution is:

* lightweight visual perception,
* geometry extraction,
* explainable bubble monitoring,
* gas plume segmentation,
* deployment-oriented benchmarking,
* and hierarchical resource-aware framework design.

Future work should focus on field deployment, multi-sensor fusion, closed-loop robotic integration, and broader underwater validation.
