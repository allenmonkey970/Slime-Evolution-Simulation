import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from collections import defaultdict

class Tree:
    def __init__(self, x, y):
        self.apple_count = 10  # Each tree produces 10 apples per turn
        self.x = x  # Spatial position
        self.y = y

    def produce_apples(self):
        self.apple_count = 10  # Reset apple production each turn

    def reproduce(self, world_size, spread=0.08):
        """Trees can reproduce by spawning a new tree nearby (if within bounds)."""
        angle = random.uniform(0, 2 * np.pi)
        distance = random.uniform(0.03, spread)
        new_x = self.x + np.cos(angle) * distance
        new_y = self.y + np.sin(angle) * distance
        # world bounds
        new_x = max(0, min(world_size[0], new_x))
        new_y = max(0, min(world_size[1], new_y))
        return Tree(new_x, new_y)

class Slime:
    def __init__(self, gene_type, x, y):
        self.gene_type = gene_type  # "greedy" or "minimal"
        self.apples = 0
        self.needs_to_search = True
        self.x = x  # Spatial position
        self.y = y
        self.age = 0

    def collect_apples(self, trees):
        """Slime collects apples from trees"""
        if not self.needs_to_search:
            # Greedy slime that collected 10 apples previously doesn't need to search
            self.needs_to_search = True  # Will need to search next turn
            return True

        # Sort trees by distance to slime
        sorted_trees = sorted(trees, key=lambda t: ((t.x - self.x) ** 2 + (t.y - self.y) ** 2) ** 0.5)

        for tree in sorted_trees:
            if tree.apple_count >= 5:  # Need at least 5 apples
                # Move slime closer to the tree
                self.x = 0.8 * self.x + 0.2 * tree.x
                self.y = 0.8 * self.y + 0.2 * tree.y

                # Collect based on gene type
                if self.gene_type == "greedy" and tree.apple_count >= 10:
                    tree.apple_count -= 10
                    self.apples += 10
                    self.needs_to_search = False  # Won't need to search next turn
                    return True
                elif self.gene_type == "minimal":
                    tree.apple_count -= 5
                    self.apples += 5
                    self.needs_to_search = True  # Will always need to search
                    return True

        # Couldn't find enough apples - move randomly
        self.x += random.uniform(-0.05, 0.05)
        self.y += random.uniform(-0.05, 0.05)
        # Keep within boundaries
        self.x = max(0, min(1, self.x))
        self.y = max(0, min(1, self.y))
        return False

    def survive(self):
        """Consume apples for survival"""
        if self.apples >= 5:  # Survival costs 5 apples
            self.apples -= 5
            self.age += 1
            return True
        else:
            return False

    def reproduce(self, partner):
        """Reproduce with another slime"""
        # Determine gene type with 50% chance from each parent
        if random.random() < 0.5:
            child_gene = self.gene_type
        else:
            child_gene = partner.gene_type

        # Child appears between parents
        child_x = (self.x + partner.x) / 2 + random.uniform(-0.05, 0.05)
        child_y = (self.y + partner.y) / 2 + random.uniform(-0.05, 0.05)

        # Keep within boundaries
        child_x = max(0, min(1, child_x))
        child_y = max(0, min(1, child_y))

        return Slime(child_gene, child_x, child_y)

class Simulation:
    def __init__(self, num_trees, initial_slimes, world_size=(1, 1), max_trees=None, tree_repro_chance=0.03):
        self.world_size = world_size
        self.max_trees = max_trees  # None means infinite trees allowed
        self.tree_repro_chance = tree_repro_chance  # Chance per tree per turn to reproduce

        # Create trees at random positions
        self.trees = [Tree(random.random(), random.random()) for _ in range(num_trees)]

        self.slimes = []
        # Create initial population with equal distribution of genes at random positions
        half = initial_slimes // 2
        for _ in range(half):
            self.slimes.append(Slime("greedy", random.random(), random.random()))
        for _ in range(initial_slimes - half):
            self.slimes.append(Slime("minimal", random.random(), random.random()))

        # For tracking statistics
        self.turn = 0
        self.stats = {
            "greedy": [half],
            "minimal": [initial_slimes - half],
            "total": [initial_slimes],
            "apple_count": [sum(tree.apple_count for tree in self.trees)],
            "tree_count": [len(self.trees)],
            "positions": [],  # For storing positions of each slime at each turn
            "tree_positions": [(tree.x, tree.y) for tree in self.trees]
        }
        # Save initial positions
        self.update_positions()

    def update_positions(self):
        """Update the position records for visualization"""
        greedy_pos = [(slime.x, slime.y) for slime in self.slimes if slime.gene_type == "greedy"]
        minimal_pos = [(slime.x, slime.y) for slime in self.slimes if slime.gene_type == "minimal"]
        tree_pos = [(tree.x, tree.y) for tree in self.trees]
        self.stats["positions"].append({
            "greedy": greedy_pos,
            "minimal": minimal_pos,
            "trees": tree_pos
        })

    def run_turn(self):
        """Execute one turn of the simulation"""
        self.turn += 1

        # --- Tree reproduction ---
        new_trees = []
        # If max_trees is None, allow infinite trees
        if self.max_trees is None or len(self.trees) < self.max_trees:
            for tree in self.trees:
                if random.random() < self.tree_repro_chance:
                    if self.max_trees is None or len(self.trees) + len(new_trees) < self.max_trees:
                        new_tree = tree.reproduce(self.world_size)
                        new_trees.append(new_tree)
        self.trees.extend(new_trees)

        # Trees produce new apples
        for tree in self.trees:
            tree.produce_apples()

        # Slimes collect apples
        random.shuffle(self.slimes)  # Randomize slime order
        survivors = []

        for slime in self.slimes:
            if slime.collect_apples(self.trees) and slime.survive():
                survivors.append(slime)

        # Reproduction phase
        new_generation = []

        # Only reproduce if we have at least 2 slimes
        if len(survivors) >= 2:
            # Pair slimes randomly for reproduction
            random.shuffle(survivors)
            for i in range(0, len(survivors) - 1, 2):
                parent1 = survivors[i]
                parent2 = survivors[i + 1]

                # Each pair produces one child
                child = parent1.reproduce(parent2)
                new_generation.append(child)

        # Update slime population with survivors and their offspring
        self.slimes = survivors + new_generation

        # Update statistics
        greedy_count = sum(1 for slime in self.slimes if slime.gene_type == "greedy")
        minimal_count = len(self.slimes) - greedy_count
        apple_count = sum(tree.apple_count for tree in self.trees)

        self.stats["greedy"].append(greedy_count)
        self.stats["minimal"].append(minimal_count)
        self.stats["total"].append(len(self.slimes))
        self.stats["apple_count"].append(apple_count)
        self.stats["tree_count"].append(len(self.trees))
        self.stats["tree_positions"] = [(tree.x, tree.y) for tree in self.trees]

        # Update positions
        self.update_positions()

    def run_simulation(self, num_turns):
        """Run the simulation for a specified number of turns"""
        for _ in range(num_turns):
            self.run_turn()
            # Stop if extinction
            if len(self.slimes) == 0:
                print(f"Extinction after {self.turn} turns!")
                break

        return self.stats

    def plot_population_graph(self):
        """Plot the population trends over time"""
        turns = range(len(self.stats["greedy"]))

        plt.figure(figsize=(12, 8))

        # Plot population counts
        plt.subplot(3, 1, 1)
        plt.plot(turns, self.stats["greedy"], 'r-', linewidth=2.5, label='Greedy Slimes')
        plt.plot(turns, self.stats["minimal"], 'b-', linewidth=2.5, label='Minimal Slimes')
        plt.plot(turns, self.stats["total"], 'g--', linewidth=2, label='Total Population')
        plt.plot(turns, self.stats["apple_count"], 'y-.', linewidth=1.5, label='Total Apples')
        plt.plot(turns, self.stats["tree_count"], 'darkgreen', linestyle=':', linewidth=2, label='Tree Count')

        plt.xlabel('Turn')
        plt.ylabel('Count')
        plt.title('Slime Population and Resource Dynamics')
        plt.legend()
        plt.grid(True)

        # Plot population percentages
        plt.subplot(3, 1, 2)
        total_pop = np.array(self.stats["total"])
        # Avoid division by zero
        total_pop[total_pop == 0] = 1
        greedy_pct = 100 * np.array(self.stats["greedy"]) / total_pop
        minimal_pct = 100 * np.array(self.stats["minimal"]) / total_pop

        plt.stackplot(turns, greedy_pct, minimal_pct,
                      labels=['Greedy Gene %', 'Minimal Gene %'],
                      colors=['#ff9999', '#9999ff'])
        plt.xlabel('Turn')
        plt.ylabel('Percentage of Population')
        plt.title('Gene Distribution Over Time')
        plt.legend()
        plt.grid(True)

        # Plot number of trees
        plt.subplot(3, 1, 3)
        plt.plot(turns, self.stats["tree_count"], color='darkgreen', label='Tree Count', linewidth=2)
        plt.xlabel('Turn')
        plt.ylabel('Trees')
        plt.title('Number of Trees Over Time')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig('slime_population_over_time.png', dpi=300)
        plt.show()

    def create_animation(self, interval=200, frames=None):
        """Create animation of the simulation - always saves both MP4 (if possible) and GIF"""
        if frames is None:
            frames = len(self.stats["positions"])

        fig, ax = plt.subplots(figsize=(10, 8))

        # Initialize empty scatter plots for slimes and trees
        greedy_scatter = ax.scatter([], [], color='red', s=50, alpha=0.7, label='Greedy Slimes')
        minimal_scatter = ax.scatter([], [], color='blue', s=50, alpha=0.7, label='Minimal Slimes')
        trees_scatter = ax.scatter([], [], color='green', s=100, marker='^', label='Trees')

        # Text for displaying turn and counts
        turn_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('Slime Evolution Simulation')
        ax.legend(loc='upper right')

        def update(frame):
            # Update slime and tree positions
            if frame < len(self.stats["positions"]):
                pos = self.stats["positions"][frame]

                if pos["greedy"]:
                    greedy_x, greedy_y = zip(*pos["greedy"]) if pos["greedy"] else ([], [])
                    greedy_scatter.set_offsets(np.column_stack([greedy_x, greedy_y]))
                else:
                    greedy_scatter.set_offsets(np.array([[], []]).T)

                if pos["minimal"]:
                    minimal_x, minimal_y = zip(*pos["minimal"]) if pos["minimal"] else ([], [])
                    minimal_scatter.set_offsets(np.column_stack([minimal_x, minimal_y]))
                else:
                    minimal_scatter.set_offsets(np.array([[], []]).T)

                if pos["trees"]:
                    tree_x, tree_y = zip(*pos["trees"])
                    trees_scatter.set_offsets(np.column_stack([tree_x, tree_y]))

                # Update counts text
                turn_text.set_text(f'Turn: {frame}\n'
                                   f'Greedy: {self.stats["greedy"][frame]}\n'
                                   f'Minimal: {self.stats["minimal"][frame]}\n'
                                   f'Total: {self.stats["total"][frame]}\n'
                                   f'Trees: {self.stats["tree_count"][frame]}\n'
                                   f'Apples: {self.stats["apple_count"][frame]}')

            return greedy_scatter, minimal_scatter, trees_scatter, turn_text

        ani = FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)
        plt.tight_layout()

        print("Creating GIF animation...")
        try:
            # Use pillow to create GIF
            ani.save('slime_simulation.gif', writer='pillow', fps=8, dpi=80)
            print("✓ Animation saved as 'slime_simulation.gif'")
        except Exception as e:
            print(f"✗ Could not save GIF: {str(e)}")

        print("Trying to create MP4 video... ")
        try:
            ani.save('slime_simulation.mp4', writer='ffmpeg', fps=10, dpi=150)
            print("✓ Animation also saved as 'slime_simulation.mp4'")
        except Exception as e:
            print(f"✗ MP4 creation skipped: {str(e)}")

        plt.show()
        return ani

    def final_stats_visualization(self):
        """Create a comprehensive final statistics visualization"""
        plt.figure(figsize=(15, 12))

        # Population history
        plt.subplot(2, 2, 1)
        turns = range(len(self.stats["greedy"]))
        plt.plot(turns, self.stats["greedy"], 'r-', label='Greedy')
        plt.plot(turns, self.stats["minimal"], 'b-', label='Minimal')
        plt.plot(turns, self.stats["total"], 'g--', label='Total')
        plt.xlabel('Turn')
        plt.ylabel('Population')
        plt.title('Population History')
        plt.grid(True)
        plt.legend()

        # Final spatial distribution
        plt.subplot(2, 2, 2)
        final_pos = self.stats["positions"][-1]
        if final_pos["greedy"]:
            greedy_x, greedy_y = zip(*final_pos["greedy"])
            plt.scatter(greedy_x, greedy_y, color='red', s=50, alpha=0.7, label='Greedy')
        if final_pos["minimal"]:
            minimal_x, minimal_y = zip(*final_pos["minimal"])
            plt.scatter(minimal_x, minimal_y, color='blue', s=50, alpha=0.7, label='Minimal')
        if final_pos["trees"]:
            tree_x, tree_y = zip(*final_pos["trees"])
            plt.scatter(tree_x, tree_y, color='green', s=100, marker='^', label='Trees')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Final Spatial Distribution')
        plt.grid(True)
        plt.legend()

        # Gene percentage over time
        plt.subplot(2, 2, 3)
        total_pop = np.array(self.stats["total"])
        total_pop[total_pop == 0] = 1  # Avoid division by zero
        greedy_pct = np.array(self.stats["greedy"]) / total_pop
        minimal_pct = np.array(self.stats["minimal"]) / total_pop

        plt.stackplot(turns, [greedy_pct, minimal_pct],
                      labels=['Greedy', 'Minimal'],
                      colors=['#ff9999', '#9999ff'])
        plt.xlabel('Turn')
        plt.ylabel('Proportion')
        plt.title('Gene Distribution Over Time')
        plt.grid(True)
        plt.legend()

        # Resource availability over time
        plt.subplot(2, 2, 4)
        plt.plot(turns, self.stats["apple_count"], color='red', linewidth=2, label='Apples')
        plt.plot(turns, self.stats["tree_count"], color='darkgreen', linewidth=2, label='Trees')
        plt.xlabel('Turn')
        plt.ylabel('Count')
        plt.title('Resource Availability')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig('slime_final_stats.png', dpi=300)
        plt.show()


# Example usage
if __name__ == "__main__":
    # Parameters
    NUM_TREES = 30
    INITIAL_SLIMES = 50
    NUM_TURNS = 100

    # Tree reproduction options
    MAX_TREES = None           # Infinite trees allowed
    TREE_REPRO_CHANCE = 0.03   # Probability for each tree to reproduce per turn

    # Run the simulation
    print("Starting simulation...")
    sim = Simulation(NUM_TREES, INITIAL_SLIMES, max_trees=MAX_TREES, tree_repro_chance=TREE_REPRO_CHANCE)
    stats = sim.run_simulation(NUM_TURNS)

    print("Generating visualizations...")
    # Static visualizations
    sim.plot_population_graph()
    sim.final_stats_visualization()

    # visualization (animation)
    print("Creating animation...")
    animation = sim.create_animation()

    # Print final statistics
    print(f"\nFinal Results after {sim.turn} turns:")
    print(f"Total population: {stats['total'][-1]} slimes")
    if stats['total'][-1] > 0:
        print(f"Greedy slimes: {stats['greedy'][-1]} ({stats['greedy'][-1] / stats['total'][-1] * 100:.1f}%)")
        print(f"Minimal slimes: {stats['minimal'][-1]} ({stats['minimal'][-1] / stats['total'][-1] * 100:.1f}%)")

    if stats['greedy'][-1] > stats['minimal'][-1]:
        print("\nConclusion: The greedy gene strategy was more successful!")
    elif stats['minimal'][-1] > stats['greedy'][-1]:
        print("\nConclusion: The minimal gene strategy was more successful!")
    else:
        print("\nConclusion: Both gene strategies performed equally well!")

    print("\n Final resource availability:")
    final_tree_count = stats['tree_count'][-1]
    apples_produced_last_turn = final_tree_count * 10
    print(f"\nFinal number of trees: {final_tree_count}")
    print(f"Apples produced in the last turn: {apples_produced_last_turn}")