# 📦 Box-Embedding-Unboxed
> **Visualizing Geometric Reasoning & Topological Constraints in 2D Space**


*(여기에 가장 잘 나온 최종 성공 GIF 경로를 넣으세요)*

## 1. Introduction
**"Why Box embedding?"**

In KG(Knowledge Graph), representing a knowledge(entity) as a point-vector in the embedding space makes hard to answer queries over the KG. It lacks the ability to represent Hierarchy and Uncertainty of knowledge. 

For example, think of answering complex queries involving sets of entities(e.g., *“where did Canadian citizens with
Turing Award graduate?”*) where each entities are represented as point-vector in the embedding space, It would be hard to imagine how answer point-vector of the query should be like. [(Ren et al., 2020)](https://arxiv.org/pdf/2002.05969)

This project visualizes how model train the Hierarchy and Uncentainty of knowledge using geometric properties of Box embedding[(Vilnis et al., 208)](https://arxiv.org/pdf/2109.04997)


## 2. Key Features
* **Geometric Reasoning:** Implement intersection and containment of knowledge in 2D space using box
* **Custom Synthetic Dataset:** Tree-structure dataset consist of Mutual-Exclusive 3 domain(Science, Art, Business)
* **Optimization:**
    * **Volume Regularization:** : Prevent boxes being bigger without constraint
    * **Aspect Ratio Regularization:** Prvent boxes making Orthogonal Overlap

## 3. Experiment & Analysis (Trouble Shooting)
The main point of this project is solving Topological trap with "Data-centric" method and dealing with several optimization problem

### 3.1. The Limitation of 2D Space (Blocking)
* **Problem:** 2D plane(which I choose willfully to visualize box embedding) has unsufficient bypass compare to high-dimension. Thus, If there is an obstacle between the parent and the child
 2차원 평면은 고차원에 비해 우회로가 부족합니다. 이로 인해 직계 부모(`Jazz` → `Music`) 관계는 학습되지만, 조상(`Jazz` → `Art`) 관계는 중간의 장애물(Negative Samples)에 가로막혀 수렴하지 못하는 **Blocking** 현상이 발생했습니다.
* **Solution (Data-Centric Approach):**
    * 모델의 파라미터를 튜닝하는 대신, **Transitive Closure (이행적 폐포)** 알고리즘을 적용했습니다.
    * 데이터셋에 `(Grandchild, IsA, Grandparent)` 관계를 명시적으로 주입하여, 모델이 중간 장애물을 뛰어넘어 수렴하도록 유도했습니다.

### 3.2 Greedy box problem : Add loss to big box
### 3.3.Boxes play tricks :  Anisotropy (비등방성)
* **Observation:** 특정 박스들이 세로 혹은 가로로 길게 늘어지는 현상 관측.
* **Analysis:** 이는 모델이 Negative Constraint가 없는 방향(Null Space)으로 박스를 확장하여 Loss를 줄이려는 기하학적 최적화 과정임을 확인했습니다.
### 3.4 Sometimes, Hate is useful : The necessity of negative sampling

## 4. Conclusion
**"Better Data > Better Model"**
초기에는 Learning Rate나 Margin 튜닝에 집중했으나, 근본적인 해결책은 **데이터의 구조적 결함(Transitivity 부족)을 보완**하는 것이었습니다. 이를 통해 AI 모델링에서 아키텍처만큼이나 **데이터의 품질과 구조(Data Quality)**가 성능에 결정적임을 확인했습니다.

"Optimization Strategy: To handle the sparsity of hierarchical data (frequent root nodes vs. rare leaf nodes), I utilized the Adam optimizer, which adapts the learning rate for each embedding parameter individually, preventing frequent nodes from oscillating while ensuring rare nodes converge effectively."

## 5. Tech Stack
* **Language:** Python 3.10
* **Framework:** PyTorch
* **Visualization:** Matplotlib, ImageIO



## 6. How to Run
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train & Visualize
python main.py --epochs 3000 --lr 0.005