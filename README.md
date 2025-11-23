# 📦 Box-Embedding-Unboxed
<div align="center">

<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/PyTorch-2.9.0-EE4C2C?style=flat&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/Matplotlib-3.10.7-11557c?style=flat&logo=python&logoColor=white"/>
<br>

[![English](https://img.shields.io/badge/Language-English-blue?style=for-the-badge)](README.md)
[![Korean](https://img.shields.io/badge/Language-Korean-red?style=for-the-badge)](README_KR.md)

</div>

> **Visualizing Geometric Reasoning & Topological Constraints in 2D Space**

## 📑 Table of Contents
- [1. Introduction](#1-Introduction)
- [2. Key Features](#2-key-features)
- [3. Experiment & Analysis (Core)](#3-experiment--analysis-trouble-shooting)
- [4. Conclusion](#4-conclusion)
- [5. How to Run](#5-how-to-run)

## 1. Introduction
**"Why Box embedding?"**
![final gif](./gifs/main.gif)

In KG(Knowledge Graph), representing a knowledge(entity) as a point-vector in the embedding space makes hard to answer queries over the KG. It lacks the ability to represent Hierarchy and Uncertainty of knowledge. 

For example, think of answering complex queries involving sets of entities(e.g., *“where did Canadian citizens with
Turing Award graduate?”*) where each entities are represented as point-vector in the embedding space, It is geometrically counter-intuitive to model such a set as a single point vector. Box Embeddings address this by mapping entities to hyper-rectangles (Boxes), where the box's volume naturally represents the set of answer entities. [(Ren et al., 2020)](https://arxiv.org/pdf/2002.05969)

This project visualizes how model train the Hierarchy and Uncentainty of knowledge using geometric properties of Box embedding[(Vilnis et al., 2018)](https://arxiv.org/pdf/2109.04997)

## 2. Key Features
* **Geometric Reasoning:** Implement intersection and containment of knowledge in 2D space using box
* **Custom Synthetic Dataset:** Tree-structure dataset consist of Mutual-Exclusive 3 domain(Science, Art, Business)
* **Optimization:**
    * **Volume Regularization:**  Prevent boxes being bigger without constraint
    * **Aspect Ratio Regularization:** Prevent boxes making Orthogonal Overlap

## 3. Experiment & Analysis (Trouble Shooting)
![wrong](./gifs/wrong_case.gif)

The main point of this project is solving Topological trap with "Data-centric" method and dealing with several optimization problem

### 3.1. The Limitation of 2D Space (Blocking)
* **Problem 1(Topological Trap):** 2D planes(which I choose willfully to visualize box embedding) have fewer detours compared to higher dimensions, and a topological trap was observed where the child could not reach the parent due to an obstacle intervening between the parent and child.(e.g., `Jazz` failed to getting into `Music` stably due to bothering form negative samples)

* **Problem 2(Nesting Bottleneck):** Additionally, while the child-parent relationship is learned, the grandparent-grandchild relationship fails to be learned, which also caused a problem where the child box and parent box were not contained within the grandparent box(e.g., `Marketing` including `Brandinig` and `Ads` is far apart from `Business`) This is because the boxes, trained in a limited 2D space, did not receive a strong enough incentive to go to the grandparent box.

### 3.2.Boxes play tricks 
* **Problem (Anisotropy):** The loss resulting from the boxes' negative samples is measured by the size of the overlap area between the negative samples; this led to an observation where the boxes unnaturally stretched in either the horizontal or vertical direction, and the negative samples overlapped in a cross-like pattern.(`CS` is too long vertically and make few loss with `Finance`, `Branding`... unrelated entities)

### 3.3 Other optimization 
![collapse](./gifs/collapse.gif)

* **Problem (Mode Collapse & Volumes Explosion):** A phenomenon was observed where, when training boxes using only positive samples, boxes belonging to different domains were not distinguished, and the boxes reduced the loss simply by growing larger. Therefore, volume loss and negative samples were added.


### 3.4 Solution to problems
The problems are organically intertwined, meaning there is no single perfect solution for any one problem, but the main ideas used to address issue are as follows

* Append Grandparent-Grandchild data : We added the grandparent-child relationship to the existing data, which previously only included parent-child relationships, This created a strong force allowing the whole family to overcome obstacles and gather, as the child box is pulled towards both the parent and the grandparent box, Additionally, it solved the problem where the child box was not contained within the grandparent box, even when the child box overlapped the parent box and the parent box overlapped the grandparent box. -> **Topological Trap & Nesting Bottleneck**

* Use negative sampling : By adding negative samples, the model was trained to distinguish between different domains. -> **Mode Collapse**

* Add loss to anisotrophy : Loss was designed to increase if the box shape was biased toward either the horizontal or vertical direction, which prevented the boxes from expanding in unnecessary directions, Furthermore, a box shape closer to a square helped reduce the frequency of topological trap occurrence and guided the model to learn in the correct direction. -> **Topological Trap & Anisotrophy**

* Add loss to big volume : By adding a loss term to the box volume (or: size), this prevented the boxes from growing without limits during training. -> **Volume Explosion**


## 4. Conclusion
**"Better Data > Better Model Architecture"**

Through this project, visualizing the learning dynamics of Box Embeddings in a constrained 2D space provided critical insights into the interplay between data quality and model optimization.

Initially, attempts were made to resolve the Topological Trap (Blocking) and Nesting Bottleneck solely by tuning hyperparameters such as learning rate and margin. However, these efforts only shifted the local minima rather than solving the fundamental geometric constraints.

The breakthrough came from a Data-Centric approach: injecting Transitive Closure (Grandparent-Grandchild relations) into the dataset. By explicitly providing the model with "direct flight" connections (Grandchild-to-Grandparent), the model could bypass topological obstacles that were insurmountable in 2D space.This demonstrated that structural quality of data often outweighs complex model architecture adjustments, especially in geometrically constrained environments.

This project goes beyond simple implementation; it serves as a visual proof of concepts for : 
* **Geometric Reasoning**: Validating how logical entailment ($A \subset B$) translates to geometric containment. 
* **Optimization Dynamics**: Identifying and resolving mode collapse, anisotropy, and topological traps using targeted regularization techniques.

Ultimately, Unboxing the "Box" revealed that building a robust AI model requires not just minimizing loss, but deeply understanding the geometry of the latent space and the integrity of the data it learns from.

## 5. How to Run
```bash
# Clone the repository
git clone https://github.com/shw707/Box-Embedding-Unboxed.git
cd Box-Embedding-Unboxed

# Install dependencies
pip install -r requirements.txt
```
### 💻  Usage
```
python main.py --mode [MODE] --filename [NAME] --seed [INT]
```
| Argument | Type | Description | Options |
| :--- | :--- | :--- | :--- |
| **`--mode`** | `str` | **(Required)** Experiment scenario | `fine`, `anisotropy`, `collapse` |
| `--filename` | `str` | Output GIF filename (default: built-in filename) | Any string |
| `--seed` | `int` | Random seed for reproducibility | Any integer (e.g., `42`) |
### 🧪 3. Scenarios (Demo)
✅ Case 1: Success Model (Optimal Settings) Visualize how Transitive Closure and Regularization solve topological traps.
```
# Quick Start (Best Result)
python main.py --mode fine --filename my_success_box --seed 42
```


⚠️ Case 2: Anisotropy (Cross-shape Artifacts) Reproduce the "Cross-shape" artifacts caused by orthogonal overlap (without Aspect Ratio Reg).
```
python main.py --mode anisotropy --filename artifact_case --seed 42
```
❌ Case 3: Mode Collapse (Failure) Reproduce the "Clustering" phenomenon where distinct domains fail to separate (without Negative Sampling).
```
python main.py --mode collapse --filename collapse_case --seed 42
```

## 6.Limitaions

## 📚 References

This project implements the core concepts of Box Embeddings based on the following papers.

**1. Box Embeddings (Foundational Concept)**
* Vilnis, L., Li, X., Murty, S., & McCallum, A. (2018). **Probabilistic Embedding of Knowledge Graphs with Box Lattice Measures**. *ACL 2018*.
    * *Implemented:* Box containment as logical entailment, intersection operation.

**2. Query2Box (Parameterization & Logic)**
* Ren, H., Hu, W., & Leskovec, J. (2020). **Query2Box: Reasoning over Knowledge Graphs in Vector Space Using Box Embeddings**. *ICLR 2020*.
    * *Adopted:* Center-Offset parameterization, geometric reasoning logic.


## 🤝 Acknowledgement
This project was developed with the assistance of **Google Gemini**, which provided insights into geometric interpretation and code optimization strategies.

