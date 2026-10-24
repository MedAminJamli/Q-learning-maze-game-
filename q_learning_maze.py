"""
Q-Learning Maze Visualizer for Students
Built with Python 3 and Tkinter (standard library, zero external dependencies).
"""

import tkinter as tk
from tkinter import ttk
from collections import deque
import random
import math

# =====================================================================
# 1. ENVIRONMENT CONFIGURATION (Intermediate 11x10 Maze)
# =====================================================================
# 0: Open Path, 1: Wall, 2: Start, 3: Exit Goal
MAZE_LAYOUT = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 2, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
    [1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 0, 0, 1, 3, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

ROWS = len(MAZE_LAYOUT)
COLS = len(MAZE_LAYOUT[0])
CELL_SIZE = 46

# Action mappings: 0=Up, 1=Down, 2=Left, 3=Right
ACTIONS = [0, 1, 2, 3]
ACTION_NAMES = ["↑ Up", "↓ Down", "← Left", "→ Right"]
ACTION_SYMBOLS = ["↑", "↓", "←", "→"]
DELTA = [(-1, 0), (1, 0), (0, -1), (0, 1)]

START_POS = (1, 1)
EXIT_POS = (8, 9)
for r in range(ROWS):
    for c in range(COLS):
        if MAZE_LAYOUT[r][c] == 2:
            START_POS = (r, c)
        elif MAZE_LAYOUT[r][c] == 3:
            EXIT_POS = (r, c)

WALKABLE_STATES = [
    (r, c) for r in range(ROWS) for c in range(COLS) if MAZE_LAYOUT[r][c] != 1
]

# Shortest path distance via BFS from every cell to exit
DIST_TO_EXIT = {EXIT_POS: 0}
q = deque([EXIT_POS])
while q:
    cr, cc = q.popleft()
    for dr, dc in DELTA:
        nr, nc = cr + dr, cc + dc
        if 0 <= nr < ROWS and 0 <= nc < COLS and MAZE_LAYOUT[nr][nc] != 1 and (nr, nc) not in DIST_TO_EXIT:
            DIST_TO_EXIT[(nr, nc)] = DIST_TO_EXIT[(cr, cc)] + 1
            q.append((nr, nc))

# Clean & Intuitive Rewards:
REWARD_GOAL = 100.0     # Goal reached
REWARD_GOOD_DIR = 1.0   # Closer to exit
REWARD_BAD_DIR = -1.0   # Backtracking / moving away
REWARD_WALL = -5.0      # Wall collision punishment
MAX_STEPS_PER_EPISODE = 600


# =====================================================================
# 2. Q-LEARNING AGENT
# =====================================================================
class QLearningAgent:
    def __init__(self, alpha=0.25, gamma=0.95, epsilon=1.0, decay=0.88, epsilon_min=0.05):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay = decay
        self.epsilon_min = epsilon_min
        self.q_table = {}
        self.reset_q()

    def reset_q(self):
        self.q_table = {}
        for r, c in WALKABLE_STATES:
            q_list = []
            for dr, dc in DELTA:
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= ROWS or nc < 0 or nc >= COLS or MAZE_LAYOUT[nr][nc] == 1:
                    q_list.append(-5.0)  # Wall action initialized to penalty
                else:
                    q_list.append(0.0)   # Open path initialized to 0.0
            self.q_table[(r, c)] = q_list

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(ACTIONS), True
        q_vals = self.q_table[state]
        max_q = max(q_vals)
        best_actions = [i for i, v in enumerate(q_vals) if math.isclose(v, max_q, abs_tol=1e-4)]
        return random.choice(best_actions), False

    def update(self, state, action, reward, next_state, is_terminal, is_wall):
        old_q = self.q_table[state][action]
        # Walls have no next state (0.0). Terminal goal has no next actions (0.0).
        max_next_q = 0.0 if (is_terminal or is_wall) else max(self.q_table[next_state])
        td_target = reward + self.gamma * max_next_q
        td_error = td_target - old_q
        new_q = old_q + self.alpha * td_error
        self.q_table[state][action] = new_q
        return old_q, max_next_q, new_q

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.decay)


# =====================================================================
# 3. TKINTER GUI APPLICATION
# =====================================================================
class QLearningApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Q-Learning Maze Visualizer (Reinforcement Learning)")
        self.geometry("1280x780")
        self.minsize(1100, 700)
        self.configure(bg="#090d16")

        self.agent = QLearningAgent()
        self.robot_pos = START_POS
        self.episode = 0
        self.current_step = 0
        self.episode_reward = 0.0
        self.best_reward = -float("inf")
        self.wins = 0
        self.is_running = False
        self.show_policy = False
        self.reward_history = []
        self.after_id = None

        self._build_ui()
        self._draw_maze()
        self._update_q_table_view()
        self._update_status_bar("Ready. Exit: +100 | Good Direction: +1 | Backtrack: -1 | Wall: -5")

    def _build_ui(self):
        header = tk.Frame(self, bg="#0f172a", pady=10, padx=20)
        header.pack(fill=tk.X)
        title = tk.Label(
            header,
            text="🤖 Q-Learning Interactive Maze Visualizer",
            font=("Segoe UI", 15, "bold"),
            fg="#38bdf8",
            bg="#0f172a",
        )
        title.pack(side=tk.LEFT)
        subtitle = tk.Label(
            header,
            text="Exit: +100 | Good Dir: +1 | Backtrack: -1 | Wall: -5",
            font=("Segoe UI", 10),
            fg="#94a3b8",
            bg="#0f172a",
        )
        subtitle.pack(side=tk.RIGHT)

        main_content = tk.Frame(self, bg="#090d16", padx=15, pady=10)
        main_content.pack(fill=tk.BOTH, expand=True)

        left_frame = tk.Frame(main_content, bg="#090d16", width=530)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 15))

        right_frame = tk.Frame(main_content, bg="#090d16")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Environment card
        canvas_card = tk.LabelFrame(
            left_frame, text=" 🏗️ Maze Grid World (11×10) ", font=("Segoe UI", 11, "bold"),
            fg="#38bdf8", bg="#0f172a", padx=8, pady=8, bd=1, relief="solid"
        )
        canvas_card.pack(fill=tk.BOTH)

        self.canvas = tk.Canvas(
            canvas_card,
            width=COLS * CELL_SIZE,
            height=ROWS * CELL_SIZE,
            bg="#090d16",
            highlightthickness=0,
            cursor="crosshair",
        )
        self.canvas.pack()
        self.canvas.bind("<Motion>", self._on_canvas_hover)
        self.canvas.bind("<Leave>", self._on_canvas_leave)

        # Legend
        legend_frame = tk.Frame(canvas_card, bg="#0f172a", pady=4)
        legend_frame.pack(fill=tk.X)
        items = [
            ("Start (1,1)", "#16a34a"),
            ("Exit (8,9)", "#eab308"),
            ("Good (+1)", "#22c55e"),
            ("Wall (-5)", "#ef4444"),
            ("Backtrack (-1)", "#f87171"),
        ]
        for label, color in items:
            dot = tk.Label(legend_frame, text="■", fg=color, bg="#0f172a", font=("Segoe UI", 10))
            dot.pack(side=tk.LEFT, padx=(5, 1))
            lbl = tk.Label(legend_frame, text=label, fg="#94a3b8", bg="#0f172a", font=("Segoe UI", 8))
            lbl.pack(side=tk.LEFT, padx=(0, 5))

        # Bellman Equation Update
        formula_card = tk.LabelFrame(
            left_frame, text=" 📐 Real-Time Bellman Equation Update ", font=("Segoe UI", 10, "bold"),
            fg="#38bdf8", bg="#0f172a", padx=8, pady=6, bd=1, relief="solid"
        )
        formula_card.pack(fill=tk.X, pady=(10, 0))

        self.formula_text = tk.Label(
            formula_card,
            text="Q(s, a) ← Q(s, a) + α [ r + γ max Q(s', a') - Q(s, a) ]",
            font=("Consolas", 9, "bold"),
            fg="#4ade80",
            bg="#090d16",
            padx=8,
            pady=6,
            justify=tk.LEFT,
        )
        self.formula_text.pack(fill=tk.X)

        # Reward Trajectory Graph
        graph_card = tk.LabelFrame(
            left_frame, text=" 📈 Reward Learning Trajectory ", font=("Segoe UI", 10, "bold"),
            fg="#38bdf8", bg="#0f172a", padx=6, pady=4, bd=1, relief="solid"
        )
        graph_card.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        self.graph_canvas = tk.Canvas(
            graph_card, height=120, bg="#090d16", highlightthickness=0
        )
        self.graph_canvas.pack(fill=tk.BOTH, expand=True)

        # Controls
        ctrl_card = tk.LabelFrame(
            right_frame, text=" 🎮 Controls & Hyperparameters ", font=("Segoe UI", 11, "bold"),
            fg="#38bdf8", bg="#0f172a", padx=10, pady=8, bd=1, relief="solid"
        )
        ctrl_card.pack(fill=tk.X)

        btn_row = tk.Frame(ctrl_card, bg="#0f172a")
        btn_row.pack(fill=tk.X, pady=(0, 8))

        self.btn_run = tk.Button(
            btn_row, text="▶ Start Training", font=("Segoe UI", 9, "bold"),
            bg="#2563eb", fg="white", activebackground="#1d4ed8", activeforeground="white",
            relief="flat", padx=10, pady=4, command=self.toggle_training, cursor="hand2"
        )
        self.btn_run.pack(side=tk.LEFT, padx=3)

        self.btn_step = tk.Button(
            btn_row, text="⏭ Step Action", font=("Segoe UI", 9),
            bg="#1e293b", fg="white", activebackground="#334155", activeforeground="white",
            relief="flat", padx=8, pady=4, command=self.step_action, cursor="hand2"
        )
        self.btn_step.pack(side=tk.LEFT, padx=3)

        self.btn_fast = tk.Button(
            btn_row, text="⚡ Fast 10 Eps", font=("Segoe UI", 9),
            bg="#0891b2", fg="white", activebackground="#0e7490", activeforeground="white",
            relief="flat", padx=8, pady=4, command=lambda: self.fast_episodes(10), cursor="hand2"
        )
        self.btn_fast.pack(side=tk.LEFT, padx=3)

        self.btn_policy = tk.Button(
            btn_row, text="🗺️ Show Policy", font=("Segoe UI", 9),
            bg="#059669", fg="white", activebackground="#047857", activeforeground="white",
            relief="flat", padx=8, pady=4, command=self.toggle_policy_overlay, cursor="hand2"
        )
        self.btn_policy.pack(side=tk.LEFT, padx=3)

        self.btn_reset = tk.Button(
            btn_row, text="🔄 Reset All", font=("Segoe UI", 9),
            bg="#dc2626", fg="white", activebackground="#b91c1c", activeforeground="white",
            relief="flat", padx=8, pady=4, command=self.reset_all, cursor="hand2"
        )
        self.btn_reset.pack(side=tk.RIGHT, padx=3)

        # Sliders
        sliders_frame = tk.Frame(ctrl_card, bg="#0f172a")
        sliders_frame.pack(fill=tk.X)

        tk.Label(sliders_frame, text="Speed:", fg="#94a3b8", bg="#0f172a", font=("Segoe UI", 9)).grid(row=0, column=0, sticky="w", pady=2)
        self.speed_var = tk.IntVar(value=12)
        self.speed_scale = tk.Scale(
            sliders_frame, from_=1, to=25, orient=tk.HORIZONTAL, variable=self.speed_var,
            bg="#0f172a", fg="#38bdf8", highlightthickness=0, troughcolor="#334155", length=130
        )
        self.speed_scale.grid(row=0, column=1, sticky="w", padx=6)

        tk.Label(sliders_frame, text="ε Exploration:", fg="#94a3b8", bg="#0f172a", font=("Segoe UI", 9)).grid(row=0, column=2, sticky="w", pady=2, padx=(10, 0))
        self.eps_var = tk.DoubleVar(value=1.0)
        self.eps_scale = tk.Scale(
            sliders_frame, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL, variable=self.eps_var,
            bg="#0f172a", fg="#38bdf8", highlightthickness=0, troughcolor="#334155", length=130,
            command=self._on_eps_change
        )
        self.eps_scale.grid(row=0, column=3, sticky="w", padx=6)

        tk.Label(sliders_frame, text="α (Learning Rate):", fg="#94a3b8", bg="#0f172a", font=("Segoe UI", 9)).grid(row=1, column=0, sticky="w", pady=2)
        self.alpha_var = tk.DoubleVar(value=0.25)
        self.alpha_scale = tk.Scale(
            sliders_frame, from_=0.01, to=1.0, resolution=0.01, orient=tk.HORIZONTAL, variable=self.alpha_var,
            bg="#0f172a", fg="#38bdf8", highlightthickness=0, troughcolor="#334155", length=130,
            command=self._on_alpha_change
        )
        self.alpha_scale.grid(row=1, column=1, sticky="w", padx=6)

        tk.Label(sliders_frame, text="γ (Discount):", fg="#94a3b8", bg="#0f172a", font=("Segoe UI", 9)).grid(row=1, column=2, sticky="w", pady=2, padx=(10, 0))
        self.gamma_var = tk.DoubleVar(value=0.95)
        self.gamma_scale = tk.Scale(
            sliders_frame, from_=0.1, to=0.99, resolution=0.01, orient=tk.HORIZONTAL, variable=self.gamma_var,
            bg="#0f172a", fg="#38bdf8", highlightthickness=0, troughcolor="#334155", length=130,
            command=self._on_gamma_change
        )
        self.gamma_scale.grid(row=1, column=3, sticky="w", padx=6)

        # Statistics Dashboard
        stats_frame = tk.Frame(right_frame, bg="#0f172a", pady=6, padx=8)
        stats_frame.pack(fill=tk.X, pady=(10, 0))

        self.lbl_ep = tk.Label(stats_frame, text="Episodes: 0", font=("Consolas", 10, "bold"), fg="#38bdf8", bg="#0f172a")
        self.lbl_ep.pack(side=tk.LEFT, padx=6)

        self.lbl_wins = tk.Label(stats_frame, text="Wins: 0", font=("Consolas", 10, "bold"), fg="#4ade80", bg="#0f172a")
        self.lbl_wins.pack(side=tk.LEFT, padx=6)

        self.lbl_reward = tk.Label(stats_frame, text="Reward: 0", font=("Consolas", 10), fg="#f8fafc", bg="#0f172a")
        self.lbl_reward.pack(side=tk.LEFT, padx=6)

        self.lbl_best = tk.Label(stats_frame, text="Best: —", font=("Consolas", 10), fg="#fbbf24", bg="#0f172a")
        self.lbl_best.pack(side=tk.LEFT, padx=6)

        self.lbl_mode = tk.Label(stats_frame, text="Action: —", font=("Consolas", 10), fg="#a78bfa", bg="#0f172a")
        self.lbl_mode.pack(side=tk.RIGHT, padx=6)

        # Live Treeview Q-Table
        table_card = tk.LabelFrame(
            right_frame, text=" 📋 Up-To-Date Q-Table (State-Action Values) ", font=("Segoe UI", 11, "bold"),
            fg="#38bdf8", bg="#0f172a", padx=6, pady=6, bd=1, relief="solid"
        )
        table_card.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Treeview",
            background="#090d16",
            foreground="#f8fafc",
            fieldbackground="#090d16",
            rowheight=20,
            font=("Consolas", 9),
        )
        style.configure(
            "Treeview.Heading",
            background="#1e293b",
            foreground="#38bdf8",
            font=("Segoe UI", 9, "bold"),
            relief="flat",
        )
        style.map("Treeview", background=[("selected", "#1e3a5f")])

        cols = ("state", "up", "down", "left", "right", "best")
        self.tree = ttk.Treeview(table_card, columns=cols, show="headings", height=13)
        self.tree.heading("state", text="State (r, c)")
        self.tree.heading("up", text="↑ Up")
        self.tree.heading("down", text="↓ Down")
        self.tree.heading("left", text="← Left")
        self.tree.heading("right", text="→ Right")
        self.tree.heading("best", text="★ Policy")

        self.tree.column("state", width=95, anchor="center")
        self.tree.column("up", width=85, anchor="e")
        self.tree.column("down", width=85, anchor="e")
        self.tree.column("left", width=85, anchor="e")
        self.tree.column("right", width=85, anchor="e")
        self.tree.column("best", width=75, anchor="center")

        scrollbar = ttk.Scrollbar(table_card, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        for s in WALKABLE_STATES:
            self.tree.insert("", tk.END, iid=f"{s[0]}_{s[1]}", values=(f"({s[0]}, {s[1]})", "0.00", "0.00", "0.00", "0.00", "—"))

        self.status_bar = tk.Label(
            self, text="", font=("Segoe UI", 9), fg="#94a3b8", bg="#020617", anchor="w", padx=15, pady=4
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # -----------------------------------------------------------------
    # Maze Canvas Drawing
    # -----------------------------------------------------------------
    def _draw_maze(self):
        self.canvas.delete("all")

        for r in range(ROWS):
            for c in range(COLS):
                x1, y1 = c * CELL_SIZE, r * CELL_SIZE
                x2, y2 = x1 + CELL_SIZE, y1 + CELL_SIZE
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                cell = MAZE_LAYOUT[r][c]

                if cell == 1:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="#1e293b", outline="#334155", width=1)
                    continue

                self.canvas.create_rectangle(x1, y1, x2, y2, fill="#090d16", outline="#1e293b", width=1)
                q_vals = self.agent.q_table.get((r, c), [0, 0, 0, 0])

                is_wall_a = [
                    r - 1 < 0 or MAZE_LAYOUT[r - 1][c] == 1,
                    r + 1 >= ROWS or MAZE_LAYOUT[r + 1][c] == 1,
                    c - 1 < 0 or MAZE_LAYOUT[r][c - 1] == 1,
                    c + 1 >= COLS or MAZE_LAYOUT[r][c + 1] == 1,
                ]

                # Draw Q-value Triangles
                self._draw_q_triangle(x1, y1, x2, y1, cx, cy, q_vals[0], is_wall_a[0])
                self._draw_q_triangle(x1, y2, x2, y2, cx, cy, q_vals[1], is_wall_a[1])
                self._draw_q_triangle(x1, y1, x1, y2, cx, cy, q_vals[2], is_wall_a[2])
                self._draw_q_triangle(x2, y1, x2, y2, cx, cy, q_vals[3], is_wall_a[3])

                # Direction Arrow: only for valid open path
                best_a = -1
                best_val = -float("inf")
                for a in range(4):
                    if not is_wall_a[a] and q_vals[a] > best_val:
                        best_val = q_vals[a]
                        best_a = a

                if best_a != -1 and best_val > -4.9:
                    self.canvas.create_text(
                        cx, cy, text=ACTION_SYMBOLS[best_a], font=("Consolas", 12, "bold"), fill="#ffffff"
                    )

                if (r, c) == START_POS:
                    self.canvas.create_rectangle(x1 + 2, y1 + 2, x2 - 2, y2 - 2, outline="#22c55e", width=2)
                    self.canvas.create_text(cx, y1 + 9, text="START", font=("Segoe UI", 6, "bold"), fill="#4ade80")
                elif (r, c) == EXIT_POS:
                    self.canvas.create_rectangle(x1 + 2, y1 + 2, x2 - 2, y2 - 2, outline="#eab308", fill="#ca8a04", width=2)
                    self.canvas.create_text(cx, cy, text="🏁", font=("Segoe UI", 14))

        if self.show_policy:
            self._draw_greedy_path()

        # Robot
        ar, ac = self.robot_pos
        rx1 = ac * CELL_SIZE + 6
        ry1 = ar * CELL_SIZE + 6
        rx2 = (ac + 1) * CELL_SIZE - 6
        ry2 = (ar + 1) * CELL_SIZE - 6
        self.canvas.create_oval(rx1, ry1, rx2, ry2, fill="#38bdf8", outline="#e0f2fe", width=2)
        rcx = (rx1 + rx2) / 2
        rcy = (ry1 + ry2) / 2
        self.canvas.create_oval(rcx - 4, rcy - 2, rcx - 1, rcy + 1, fill="#090d16", outline="")
        self.canvas.create_oval(rcx + 1, rcy - 2, rcx + 4, rcy + 1, fill="#090d16", outline="")

    def _draw_q_triangle(self, px1, py1, px2, py2, cx, cy, val, is_wall):
        if is_wall:
            color = "#7f1d1d"  # Dark red for walls
        elif abs(val) < 1e-4:
            color = "#1e293b"
        elif val > 0:
            intensity = min(1.0, val / 50.0)
            g = int(80 + 175 * intensity)
            color = f"#{20:02x}{g:02x}{40:02x}"
        else:
            intensity = min(1.0, abs(val) / 5.0)
            r = int(90 + 160 * intensity)
            color = f"#{r:02x}{30:02x}{30:02x}"
        self.canvas.create_polygon(px1, py1, px2, py2, cx, cy, fill=color, outline="#111827", width=1)

    def _draw_greedy_path(self):
        curr = START_POS
        visited = set()
        path = [curr]
        for _ in range(50):
            if curr == EXIT_POS or curr in visited:
                break
            visited.add(curr)
            q_vals = self.agent.q_table.get(curr, [0, 0, 0, 0])

            best_a = -1
            best_val = -float("inf")
            for a in range(4):
                dr, dc = DELTA[a]
                nr, nc = curr[0] + dr, curr[1] + dc
                if 0 <= nr < ROWS and 0 <= nc < COLS and MAZE_LAYOUT[nr][nc] != 1:
                    if q_vals[a] > best_val:
                        best_val = q_vals[a]
                        best_a = a

            if best_a == -1:
                break
            curr = (curr[0] + DELTA[best_a][0], curr[1] + DELTA[best_a][1])
            path.append(curr)

        if len(path) > 1:
            coords = []
            for r, c in path:
                coords.extend([c * CELL_SIZE + CELL_SIZE // 2, r * CELL_SIZE + CELL_SIZE // 2])
            self.canvas.create_line(coords, fill="#fbbf24", width=3, dash=(4, 2), capstyle=tk.ROUND)

    # -----------------------------------------------------------------
    # Simulation Logic
    # -----------------------------------------------------------------
    def step_action(self):
        state = self.robot_pos
        action, explored = self.agent.choose_action(state)

        dr, dc = DELTA[action]
        nr, nc = state[0] + dr, state[1] + dc

        is_wall = False
        is_terminal = False

        if nr < 0 or nr >= ROWS or nc < 0 or nc >= COLS or MAZE_LAYOUT[nr][nc] == 1:
            next_state = state
            reward = REWARD_WALL
            is_wall = True
            dir_feedback = "Wall Crash (-5)"
        elif (nr, nc) == EXIT_POS:
            next_state = (nr, nc)
            reward = REWARD_GOAL
            is_terminal = True
            dir_feedback = "Exit Goal Reached (+100)"
        else:
            next_state = (nr, nc)
            curr_dist = DIST_TO_EXIT[state]
            next_dist = DIST_TO_EXIT[next_state]
            if next_dist < curr_dist:
                reward = REWARD_GOOD_DIR  # +1.0
                dir_feedback = "Good Direction (+1)"
            else:
                reward = REWARD_BAD_DIR   # -1.0
                dir_feedback = "Backtrack (-1)"

        old_q, max_next, new_q = self.agent.update(state, action, reward, next_state, is_terminal, is_wall)
        self.robot_pos = next_state
        self.current_step += 1
        self.episode_reward += reward

        act_name = ACTION_NAMES[action]
        r_str = f"+{reward:.0f}" if reward > 0 else f"{reward:.0f}"
        self.formula_text.config(
            text=(
                f"Q({state}, {act_name}) ← {old_q:.2f} + {self.agent.alpha:.2f} × "
                f"[{r_str} + {self.agent.gamma:.2f}×({max_next:.2f}) - {old_q:.2f}] = {new_q:.2f}"
            )
        )
        mode_str = "Explore" if explored else "Exploit"
        self.lbl_mode.config(text=f"Action: {ACTION_SYMBOLS[action]} [{mode_str}] | {dir_feedback}")

        episode_done = is_terminal or (self.current_step >= MAX_STEPS_PER_EPISODE)
        if episode_done:
            self.episode += 1
            if is_terminal:
                self.wins += 1
                self.agent.decay_epsilon()
                self.eps_var.set(round(self.agent.epsilon, 2))
                self._update_status_bar(f"🎉 GOAL REACHED! Episode {self.episode} finished with Reward: {self.episode_reward:.0f}")
            else:
                self._update_status_bar(f"Episode {self.episode} timed out.")

            if self.episode_reward > self.best_reward:
                self.best_reward = self.episode_reward

            self.reward_history.append(self.episode_reward)
            self.robot_pos = START_POS
            self.current_step = 0
            self.episode_reward = 0.0

        self._update_dashboard_labels()
        self._draw_maze()
        self._update_single_q_table_row(state)
        if episode_done:
            self._draw_reward_graph()

        return episode_done

    def toggle_training(self):
        self.is_running = not self.is_running
        if self.is_running:
            self.btn_run.config(text="⏸ Pause", bg="#eab308", activebackground="#ca8a04")
            self._update_status_bar("Training loop active...")
            self._run_loop()
        else:
            self.btn_run.config(text="▶ Start Training", bg="#2563eb", activebackground="#1d4ed8")
            self._update_status_bar("Training paused.")
            if self.after_id:
                self.after_cancel(self.after_id)

    def _run_loop(self):
        if not self.is_running:
            return

        self.step_action()
        sp = self.speed_var.get()
        delay = max(5, int(220 - sp * 8.5))
        self.after_id = self.after(delay, self._run_loop)

    def fast_episodes(self, n=10):
        was_running = self.is_running
        self.is_running = False
        if self.after_id:
            self.after_cancel(self.after_id)

        for _ in range(n):
            self.robot_pos = START_POS
            self.current_step = 0
            self.episode_reward = 0.0
            done = False
            while not done:
                state = self.robot_pos
                action, _ = self.agent.choose_action(state)
                dr, dc = DELTA[action]
                nr, nc = state[0] + dr, state[1] + dc

                is_wall = False
                is_terminal = False

                if nr < 0 or nr >= ROWS or nc < 0 or nc >= COLS or MAZE_LAYOUT[nr][nc] == 1:
                    next_state = state
                    reward = REWARD_WALL
                    is_wall = True
                elif (nr, nc) == EXIT_POS:
                    next_state = (nr, nc)
                    reward = REWARD_GOAL
                    is_terminal = True
                else:
                    next_state = (nr, nc)
                    if DIST_TO_EXIT[next_state] < DIST_TO_EXIT[state]:
                        reward = REWARD_GOOD_DIR
                    else:
                        reward = REWARD_BAD_DIR

                self.agent.update(state, action, reward, next_state, is_terminal, is_wall)
                self.robot_pos = next_state
                self.current_step += 1
                self.episode_reward += reward

                if is_terminal or (self.current_step >= MAX_STEPS_PER_EPISODE):
                    done = True
                    self.episode += 1
                    if is_terminal:
                        self.wins += 1
                        self.agent.decay_epsilon()
                    if self.episode_reward > self.best_reward:
                        self.best_reward = self.episode_reward
                    self.reward_history.append(self.episode_reward)

        self.robot_pos = START_POS
        self.current_step = 0
        self.episode_reward = 0.0
        self.eps_var.set(round(self.agent.epsilon, 2))

        self._update_dashboard_labels()
        self._update_q_table_view()
        self._draw_maze()
        self._draw_reward_graph()
        self._update_status_bar(f"Completed {n} fast episodes! Total wins: {self.wins}.")

        if was_running:
            self.toggle_training()

    def toggle_policy_overlay(self):
        self.show_policy = not self.show_policy
        self.btn_policy.config(
            bg="#047857" if self.show_policy else "#059669",
            text="🗺️ Hide Policy" if self.show_policy else "🗺️ Show Policy",
        )
        self._draw_maze()

    def reset_all(self):
        self.is_running = False
        if self.after_id:
            self.after_cancel(self.after_id)
        self.btn_run.config(text="▶ Start Training", bg="#2563eb")
        self.agent.reset_q()
        self.agent.epsilon = 1.0
        self.eps_var.set(1.0)
        self.robot_pos = START_POS
        self.episode = 0
        self.current_step = 0
        self.episode_reward = 0.0
        self.best_reward = -float("inf")
        self.wins = 0
        self.reward_history.clear()
        self.formula_text.config(text="Q(s, a) ← Q(s, a) + α [ r + γ max Q(s', a') - Q(s, a) ]")
        self.lbl_mode.config(text="Action: —")
        self._update_dashboard_labels()
        self._update_q_table_view()
        self._draw_maze()
        self._draw_reward_graph()
        self._update_status_bar("Simulation reset.")

    def _update_dashboard_labels(self):
        self.lbl_ep.config(text=f"Episodes: {self.episode}")
        self.lbl_wins.config(text=f"Wins: {self.wins}")
        self.lbl_reward.config(text=f"Reward: {self.episode_reward:.0f}")
        self.lbl_reward.config(fg="#4ade80" if self.episode_reward >= 0 else "#f87171")
        best_str = f"{self.best_reward:.0f}" if self.best_reward != -float("inf") else "—"
        self.lbl_best.config(text=f"Best: {best_str}")

    def _update_single_q_table_row(self, state):
        iid = f"{state[0]}_{state[1]}"
        q_vals = self.agent.q_table[state]

        best_a = -1
        best_val = -float("inf")
        for a in range(4):
            nr, nc = state[0] + DELTA[a][0], state[1] + DELTA[a][1]
            if 0 <= nr < ROWS and 0 <= nc < COLS and MAZE_LAYOUT[nr][nc] != 1:
                if q_vals[a] > best_val:
                    best_val = q_vals[a]
                    best_a = a

        best_str = ACTION_NAMES[best_a] if (best_a != -1 and best_val > -4.9) else "—"

        self.tree.item(
            iid,
            values=(
                f"({state[0]}, {state[1]})",
                f"{q_vals[0]:.2f}",
                f"{q_vals[1]:.2f}",
                f"{q_vals[2]:.2f}",
                f"{q_vals[3]:.2f}",
                best_str,
            ),
        )
        self.tree.selection_set(iid)

    def _update_q_table_view(self):
        for state in WALKABLE_STATES:
            iid = f"{state[0]}_{state[1]}"
            q_vals = self.agent.q_table[state]

            best_a = -1
            best_val = -float("inf")
            for a in range(4):
                nr, nc = state[0] + DELTA[a][0], state[1] + DELTA[a][1]
                if 0 <= nr < ROWS and 0 <= nc < COLS and MAZE_LAYOUT[nr][nc] != 1:
                    if q_vals[a] > best_val:
                        best_val = q_vals[a]
                        best_a = a

            best_str = ACTION_NAMES[best_a] if (best_a != -1 and best_val > -4.9) else "—"

            self.tree.item(
                iid,
                values=(
                    f"({state[0]}, {state[1]})",
                    f"{q_vals[0]:.2f}",
                    f"{q_vals[1]:.2f}",
                    f"{q_vals[2]:.2f}",
                    f"{q_vals[3]:.2f}",
                    best_str,
                ),
            )

    def _draw_reward_graph(self):
        self.graph_canvas.delete("all")
        w = self.graph_canvas.winfo_width()
        h = self.graph_canvas.winfo_height()
        if w < 10 or h < 10:
            w, h = 420, 120

        if len(self.reward_history) < 2:
            self.graph_canvas.create_text(
                w / 2, h / 2, text="Reward curve will appear after 2+ episodes", fill="#64748b", font=("Segoe UI", 9)
            )
            return

        pts = self.reward_history[-80:]
        min_v = min(pts)
        max_v = max(pts)
        val_range = max_v - min_v if not math.isclose(max_v, min_v) else 1.0

        pad = 20
        plot_w = w - 2 * pad
        plot_h = h - 2 * pad

        # Zero reference line
        if min_v <= 0 <= max_v:
            zero_y = h - pad - ((0 - min_v) / val_range) * plot_h
            self.graph_canvas.create_line(pad, zero_y, w - pad, zero_y, fill="#334155", dash=(3, 3))

        coords = []
        for i, val in enumerate(pts):
            x = pad + (i / (len(pts) - 1)) * plot_w
            y = h - pad - ((val - min_v) / val_range) * plot_h
            coords.extend([x, y])

        self.graph_canvas.create_line(coords, fill="#38bdf8", width=2, smooth=True)
        self.graph_canvas.create_text(pad + 15, pad - 8, text=f"{max_v:.0f}", fill="#94a3b8", font=("Consolas", 8))
        self.graph_canvas.create_text(pad + 15, h - pad + 8, text=f"{min_v:.0f}", fill="#94a3b8", font=("Consolas", 8))

    def _on_canvas_hover(self, event):
        col = event.x // CELL_SIZE
        row = event.y // CELL_SIZE
        if 0 <= row < ROWS and 0 <= col < COLS and MAZE_LAYOUT[row][col] != 1:
            q = self.agent.q_table.get((row, col), [0, 0, 0, 0])
            self._update_status_bar(
                f"Cell ({row}, {col}) → ↑{q[0]:.1f}, ↓{q[1]:.1f}, ←{q[2]:.1f}, →{q[3]:.1f}"
            )

    def _on_canvas_leave(self, event):
        self._update_status_bar("Hover over any cell to inspect its 4 Q-values.")

    def _on_eps_change(self, val):
        self.agent.epsilon = float(val)

    def _on_alpha_change(self, val):
        self.agent.alpha = float(val)

    def _on_gamma_change(self, val):
        self.agent.gamma = float(val)

    def _update_status_bar(self, msg):
        self.status_bar.config(text=f" 💡 {msg}")


if __name__ == "__main__":
    app = QLearningApp()
    app.mainloop()
