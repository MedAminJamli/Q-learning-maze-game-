# 🤖 Q-Learning Maze Explorer — Interactive Reinforcement Learning Visualizer

[![Python 3.x](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Tkinter](https://img.shields.io/badge/GUI-Tkinter%20(Built--in)-blue?style=flat)](#python-desktop-application)
[![Vanilla JS](https://img.shields.io/badge/Web-HTML5%20%2F%20Canvas%20%2F%20JS-F7DF1E?style=flat&logo=javascript&logoColor=black)](#web-application)
[![Dependencies](https://img.shields.io/badge/Dependencies-Zero%20(Pure%20Standard%20Libraries)-brightgreen?style=flat)](#quick-start)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An interactive, educational reinforcement learning visualizer built for teaching **Q-Learning** and the **Bellman Equation** to college students and machine learning beginners. Features dual implementations: a pure-Python desktop GUI (`tkinter`) and a responsive web application (`HTML5 Canvas` + `Vanilla JS`) — both completely standalone with **zero external dependencies**.

---

## 📌 Repository "About" Description (Copy & Paste for GitHub)

```text
Interactive 2D Q-Learning visualizer built for teaching Reinforcement Learning. Features live Bellman equation updates, dynamic Q-table evolution, policy arrow overlays, and convergence charts. Available in pure Python (Tkinter) and Vanilla JS/HTML5 Canvas with zero dependencies.
```

### Recommended GitHub Topics (Tags)
`reinforcement-learning` • `q-learning` • `machine-learning` • `bellman-equation` • `maze-solver` • `educational` • `python` • `tkinter` • `html5-canvas` • `artificial-intelligence` • `interactive-visualization`

---

## 🎯 Educational Motivation & Learning Objectives

When students learn tabular Q-learning, formulas like the Bellman update often feel abstract. This project brings the mathematics to life by letting students watch an agent navigate an intermediate-complexity maze in real time.

Students can visually observe:
1. **Exploration vs. Exploitation ($\varepsilon$-Greedy Policy)**: Watch the agent transition from chaotic wandering ($\varepsilon \approx 1.0$) to confident, optimal navigation ($\varepsilon \to 0.01$).
2. **Step-by-Step Bellman Updating**: Inspect the numerical substitution into:
   $$Q(s, a) \leftarrow Q(s, a) + \alpha \Big[ R + \gamma \max_{a'} Q(s', a') - Q(s, a) \Big]$$
   including Temporal Difference (TD) target, TD error, and updated Q-value for each move.
3. **Live Q-Table Evolution**: Track state-action values ($Q(s, a)$) for all 45 walkable cells across all 4 discrete actions (`Up`, `Down`, `Left`, `Right`).
4. **Policy Arrow Projection**: See the derived greedy policy $\pi(s) = \arg\max_a Q(s, a)$ emerge dynamically directly on the maze grid.
5. **Reward Shaping & Potential**: Learn how directional reward heuristics prevent exploration starvation and guide the agent through deep dead-ends.

---

## 🗺️ Intermediate Maze Architecture ($11 \times 10$)

The environment is designed to balance intermediate complexity with fast in-class convergence (typically learns within 25–40 episodes):

```text
        C0  C1  C2  C3  C4  C5  C6  C7  C8  C9  C10
Row 0:  [█] [█] [█] [█] [█] [█] [█] [█] [█] [█] [█]
Row 1:  [█]  S   ·   ·  [█]  ·   ·   ·   ·   ·  [█]  <-- Dead-end branch at (1, 9)
Row 2:  [█]  ·  [█]  ·  [█]  ·  [█] [█] [█]  ·  [█]
Row 3:  [█]  ·  [█]  ·   ·   ·  [█]  ·   ·   ·  [█]  <-- Decision junction (3, 5)
Row 4:  [█]  ·  [█] [█] [█]  ·  [█]  ·  [█] [█] [█]
Row 5:  [█]  ·   ·   ·  [█]  ·   ·   ·  [█]  ·  [█]  <-- Dead-end trap at (5, 9)
Row 6:  [█] [█] [█]  ·  [█] [█] [█]  ·  [█]  ·  [█]
Row 7:  [█]  ·   ·   ·   ·   ·  [█]  ·   ·   ·  [█]  <-- Long lower bypass corridor
Row 8:  [█]  ·  [█] [█] [█]  ·   ·   ·  [█]  G  [█]  <-- Exit Goal G at (8, 9)
Row 9:  [█] [█] [█] [█] [█] [█] [█] [█] [█] [█] [█]
```

- **Grid Size**: 11 columns $\times$ 10 rows (45 walkable states, 65 wall cells).
- **Start ($S$)**: `(1, 1)` (Row 1, Column 1).
- **Exit Goal ($G$)**: `(8, 9)` (Row 8, Column 9).
- **Optimal Shortest Path**: Exactly 15 steps.
- **Key Challenges**:
  - Deep dead ends at `(1, 9)`, `(5, 9)`, and `(7, 1)` penalize naive wandering.
  - Multiple alternative routes challenge the discount factor $\gamma$.

---

## ⚙️ Reinforcement Learning Mechanics

### 1. Reward Function (Potential-Based Shaping)
To guide the agent intuitively while penalizing backtracking and collisions, rewards are shaped via shortest-path distance $D(s)$ (computed via Breadth-First Search from Exit):

$$\text{Reward}(s, a) = \begin{cases}
+100.0 & \text{if } s_{\text{next}} = \text{Goal} \\
+1.0 & \text{if } D(s_{\text{next}}) < D(s) \quad (\text{moving closer to goal}) \\
-1.0 & \text{if } D(s_{\text{next}}) \ge D(s) \quad (\text{backtracking or detour}) \\
-5.0 & \text{if wall collision} \quad (s_{\text{next}} = s)
\end{cases}$$

### 2. Wall Collision Treatment
- Wall-hitting actions are initialized to $-5.00$.
- Wall collision states evaluate $\max_{a'} Q(s_{\text{wall}}, a') = 0.0$, preventing artificial value inflation and guaranteeing that greedy policy arrows never point into obstacles.

### 3. Conditional $\varepsilon$-Decay
- Epsilon decays ($\varepsilon \leftarrow \varepsilon \times \varepsilon_{\text{decay}}$) **only when the agent successfully reaches the exit**.
- If the agent times out (max steps per episode reached), $\varepsilon$ stays high to prevent premature exploitation of sub-optimal paths.

---

## 🚀 Quick Start

No third-party packages (like `pygame`, `gymnasium`, or `numpy`) are required. Everything runs out-of-the-box using standard runtime libraries.

### Option A: Python Desktop App (Tkinter)

```bash
# Clone the repository
git clone https://github.com/your-username/q-learning-maze-explorer.git
cd q-learning-maze-explorer

# Launch the visualizer with Python 3
python q_learning_maze.py
```

### Option B: Web App (HTML5 Canvas + Vanilla JS)

Simply double-click `index.html` to open it in your browser, or serve it locally:

```bash
# Optional: run a local lightweight HTTP server
python -m http.server 8000
```
Then visit `http://localhost:8000` in any modern web browser (Chrome, Edge, Firefox, Safari).

---

## 🎛️ Interactive Controls & Hyperparameters

Both the web and desktop interfaces provide real-time hyperparameter controls:

| Parameter | Default | Range | Pedagogical Purpose |
| :--- | :---: | :---: | :--- |
| **Learning Rate ($\alpha$)** | `0.25` | `0.01 – 1.0` | Controls how aggressively new experience overrides old estimates. |
| **Discount Factor ($\gamma$)** | `0.90` | `0.10 – 0.99` | Weights future rewards vs. immediate rewards; high values encourage long-term planning. |
| **Exploration Rate ($\varepsilon$)** | `1.00` | `0.00 – 1.00` | Probability of taking a random exploratory action vs. greedy exploitation. |
| **Epsilon Decay Rate** | `0.95` | `0.80 – 0.99` | Multiplier applied to $\varepsilon$ after each successful episode. |
| **Animation Speed** | Variable | `1 – 200 ms` | Slow down to analyze single Bellman steps, or speed up to observe fast convergence. |

---

## 📁 Repository Structure

```text
├── index.html            # Web-based interactive visualizer (Canvas, CSS Grid, Vanilla JS)
├── q_learning_maze.py    # Python 3 desktop visualizer (Tkinter, ttk, Canvas)
└── README.md             # Complete project documentation and teaching guide
```

---

## 🎓 Classroom Demo Suggestions for Instructors

1. **Demonstrating Pure Exploration**: Set $\varepsilon = 1.0$ and disable decay ($\text{decay} = 1.0$). Show students how the agent wanders randomly without ever converging on an efficient path.
2. **Demonstrating Premature Exploitation**: Set $\varepsilon = 0.0$ from episode 1. Show how the agent gets stuck in a local loop or refuses to explore the maze.
3. **The Power of $\varepsilon$-Decay**: Start with defaults ($\varepsilon = 1.0, \text{decay} = 0.95$). Watch the first 5 episodes look chaotic, followed by a dramatic transition around Episode 15–20 where the robot takes clean, purposeful lines straight to the exit.
4. **Analyzing the Bellman Box**: Pause after a step and ask students to calculate the new Q-value by hand using the on-screen formula and current values, then click **Step** to verify their calculation.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details. Open-source and free for educational and research use.
