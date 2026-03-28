# Mobile Edge Offloading

Source code for the Master's thesis **"Measuring Energy Consumption for Mobile Edge Offloading"**.

This repository investigates the energy-saving potential of computation offloading in mobile networks under different scenarios. It contains practical implementations for local, and edge execution as well as the implementation of the proposed scheduling algorithms and the baseline used for evaluation.

## Repository Overview

The repository is organized into four main subprojects:

### `offload-code`
Core implementation for the offloading system.

This project allows devices to take the role of either:
- **local device**
- **edge device**
- **cloud device** (if applicable in your setup)

The services can be started with Docker Compose from the corresponding directory.

### `conti`
Adaptation of `offload-code` for the **NVIDIA Jetson Nano**.

This subproject is intended for hardware-specific execution and deployment on embedded edge hardware.

### `Project`
Android application written in **Kotlin**.

This project enables an **Android smartphone** to act as the **local device** in the offloading experiments.

### `algos`
Implementation of the **proposed scheduling algorithms** and the **baseline approach** used for comparison in the thesis evaluation.

---

## Project Structure

```text
mobile-edge-offloading/
├── offload-code/   # Core offloading framework
├── conti/          # Jetson Nano adaptation
├── Project/        # Android/Kotlin client
├── algos/          # Scheduling algorithms and baseline
├── LICENSE
└── README.md
