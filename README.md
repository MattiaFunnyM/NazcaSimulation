# Nazca to View - MEEP Photonics Simulations

An ensemble of functions and scripts designed to harness the power of the MEEP (MIT Electromagnetic Equation Propagation) library in photonics environments.

## Purpose

This repository provides tools and utilities for performing advanced photonics simulations, including:

- **Mode Calculation**: Compute electromagnetic modes in optical structures
- **Overlap Efficiency**: Calculate coupling efficiency between modes
- **2D/3D Propagation**: Simulate light propagation through complex photonic structures
- **General Photonics Simulation**: Build and analyze various optical devices and waveguides

## Repository Structure

### Library

The core simulation libraries located in the `Library/` directory:

- **Simulation2D.py**: 2D electromagnetic simulation module (incomplete, not fully tested)
- **Simulation3D.py**: 3D electromagnetic simulation module (incomplete, not fully tested)

These libraries provide foundational classes and functions for setting up and running MEEP simulations. **Note**: These libraries are still under development and have not been fully tested across all use cases.

### Example Scripts

Three demonstration scripts in increasing order of complexity:

1. **2D Simulations.py**: Introduction to 2D waveguide simulations
2. **1.WaveguideDispersionGraph.py**: Dispersion analysis and waveguide characterization
3. **2.TaperOptimization.py**: Advanced optimization of tapered structures



### Installation

1. Clone the repository
2. Install MEEP (follow [MEEP installation guide](https://meep.readthedocs.io/))


### Improvements & Feedback

Suggestions for improvements are welcome! If you have ideas for enhancing the libraries, optimizing simulations, or adding new features, please don't hesitate to contact the maintainer.

