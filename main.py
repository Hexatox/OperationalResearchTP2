
import sys
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict


# ============================================================================
# UTILITY FUNCTIONS FOR UI/UX
# ============================================================================

class Colors:
    """ANSI color codes for beautiful console output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def print_header(text: str):
    """Print a styled header"""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'=' * 80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text.center(80)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'=' * 80}{Colors.END}\n")


def print_subheader(text: str):
    """Print a styled subheader"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'-' * 80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'-' * 80}{Colors.END}")


def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_info(text: str):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")


def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


def print_table(headers: List[str], rows: List[List], title: Optional[str] = None):
    """Print a formatted table"""
    if title:
        print(f"\n{Colors.BOLD}{title}{Colors.END}")
    
    # Calculate column widths
    col_widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    
    # Print header
    header_line = "│ " + " │ ".join(
        str(h).ljust(w) for h, w in zip(headers, col_widths)
    ) + " │"
    separator = "├" + "┼".join("─" * (w + 2) for w in col_widths) + "┤"
    top_border = "┌" + "┬".join("─" * (w + 2) for w in col_widths) + "┐"
    bottom_border = "└" + "┴".join("─" * (w + 2) for w in col_widths) + "┘"
    
    print(top_border)
    print(header_line)
    print(separator)
    
    # Print rows
    for row in rows:
        row_line = "│ " + " │ ".join(
            str(cell).ljust(w) for cell, w in zip(row, col_widths)
        ) + " │"
        print(row_line)
    
    print(bottom_border)


# ============================================================================
# PROBLEM 1: RESOURCE ALLOCATION (DYNAMIC PROGRAMMING)
# ============================================================================

class ResourceAllocation:
    """
    Solves the Resource Allocation Problem using Dynamic Programming.
    
    Problem: Given B total resources and n activities with profit functions,
    allocate resources to maximize total profit.
    """
    
    def __init__(self, total_resources: int, n_activities: int, profit_tables: List[List[int]]):
        """
        Initialize the resource allocation problem.
        
        Args:
            total_resources: Total amount of resources available (B)
            n_activities: Number of activities (n)
            profit_tables: List of profit tables, one per activity
                          profit_tables[j][x] = profit from activity j with x resources
        """
        self.B = total_resources
        self.n = n_activities
        self.profit_tables = profit_tables
        self.memo = {}
        self.decisions = {}
        self.computation_steps = []
    
    def get_profit(self, activity: int, resources: int) -> int:
        """Get profit for a given activity and resource amount"""
        if resources >= len(self.profit_tables[activity]):
            return self.profit_tables[activity][-1]
        return self.profit_tables[activity][resources]
    
    def solve(self) -> Tuple[int, List[int]]:
        """
        Solve the resource allocation problem using dynamic programming.
        
        Returns:
            Tuple of (maximum_profit, optimal_allocation)
        """
        print_header("SOLVING RESOURCE ALLOCATION PROBLEM")
        print_info(f"Total Resources: {self.B}")
        print_info(f"Number of Activities: {self.n}")
        
        # Display profit tables
        self._display_profit_tables()
        
        # Solve using DP
        max_profit = self._dp(0, self.B)
        
        # Reconstruct solution
        allocation = self._reconstruct_solution()
        
        # Display results
        self._display_memoization_table()
        self._display_computation_steps()
        self._display_final_solution(allocation, max_profit)
        
        return max_profit, allocation
    
    def _dp(self, activity: int, remaining_resources: int) -> int:
        """
        Dynamic programming recursive function with memoization.
        
        Args:
            activity: Current activity index (0 to n-1)
            remaining_resources: Resources still available
        
        Returns:
            Maximum profit achievable from this state
        """
        # Base case: no more activities
        if activity >= self.n:
            return 0
        
        # Check memo
        state = (activity, remaining_resources)
        if state in self.memo:
            return self.memo[state]
        
        # Try all possible resource allocations for current activity
        max_profit = 0
        best_allocation = 0
        
        step_info = {
            'state': state,
            'attempts': []
        }
        
        for x in range(remaining_resources + 1):
            current_profit = self.get_profit(activity, x)
            future_profit = self._dp(activity + 1, remaining_resources - x)
            total_profit = current_profit + future_profit
            
            step_info['attempts'].append({
                'allocation': x,
                'current_profit': current_profit,
                'future_profit': future_profit,
                'total_profit': total_profit
            })
            
            if total_profit > max_profit:
                max_profit = total_profit
                best_allocation = x
        
        step_info['best_allocation'] = best_allocation
        step_info['max_profit'] = max_profit
        self.computation_steps.append(step_info)
        
        # Memoize
        self.memo[state] = max_profit
        self.decisions[state] = best_allocation
        
        return max_profit
    
    def _reconstruct_solution(self) -> List[int]:
        """Reconstruct the optimal allocation from decisions"""
        allocation = []
        remaining = self.B
        
        for activity in range(self.n):
            state = (activity, remaining)
            allocated = self.decisions.get(state, 0)
            allocation.append(allocated)
            remaining -= allocated
        
        return allocation
    
    def _display_profit_tables(self):
        """Display the profit tables for all activities"""
        print_subheader("Input: Profit Tables")
        
        for j, table in enumerate(self.profit_tables):
            headers = ["Resources"] + [str(x) for x in range(len(table))]
            rows = [[f"Activity {j+1}"] + table]
            print(f"\n{Colors.BOLD}Activity {j+1} Profit Function:{Colors.END}")
            print("  ", end="")
            for x, profit in enumerate(table):
                print(f"c_{j+1}({x}) = {profit}  ", end="")
            print()
    
    def _display_memoization_table(self):
        """Display the memoization table"""
        print_subheader("Dynamic Programming Memoization Table")
        
        # Group by activity
        for activity in range(self.n):
            print(f"\n{Colors.BOLD}Activity {activity + 1}:{Colors.END}")
            states = [(a, r, v) for (a, r), v in self.memo.items() if a == activity]
            states.sort(key=lambda x: x[1], reverse=True)
            
            headers = ["Remaining Resources", "Max Profit", "Best Allocation"]
            rows = []
            for a, r, v in states:
                best_alloc = self.decisions.get((a, r), 0)
                rows.append([r, v, best_alloc])
            
            if rows:
                print_table(headers, rows, None)
    
    def _display_computation_steps(self):
        """Display selected computation steps for clarity"""
        print_subheader("Key Computation Steps (Sample)")
        
        # Show a few interesting steps
        steps_to_show = min(5, len(self.computation_steps))
        shown_steps = self.computation_steps[:steps_to_show]
        
        for i, step in enumerate(shown_steps, 1):
            activity, remaining = step['state']
            print(f"\n{Colors.CYAN}Step {i}: Activity {activity + 1}, Resources Available = {remaining}{Colors.END}")
            
            # Show top 3 attempts
            sorted_attempts = sorted(step['attempts'], key=lambda x: x['total_profit'], reverse=True)[:3]
            
            for attempt in sorted_attempts:
                alloc = attempt['allocation']
                curr = attempt['current_profit']
                future = attempt['future_profit']
                total = attempt['total_profit']
                
                marker = "→" if alloc == step['best_allocation'] else " "
                print(f"  {marker} Allocate {alloc} resources: {curr} + {future} = {Colors.GREEN}{total}{Colors.END}")
            
            print(f"  {Colors.BOLD}Decision: Allocate {step['best_allocation']} resources (Max profit: {step['max_profit']}){Colors.END}")
    
    def _display_final_solution(self, allocation: List[int], max_profit: int):
        """Display the final optimal solution"""
        print_subheader("OPTIMAL SOLUTION")
        
        headers = ["Activity", "Resources Allocated", "Profit Earned"]
        rows = []
        
        for i, allocated in enumerate(allocation):
            profit = self.get_profit(i, allocated)
            rows.append([f"Activity {i+1}", allocated, profit])
        
        rows.append(["TOTAL", sum(allocation), max_profit])
        
        print_table(headers, rows, None)
        
        print_success(f"Maximum Total Profit: {Colors.BOLD}{max_profit}{Colors.END}")
        print_success(f"Optimal Allocation: {allocation}")


# ============================================================================
# PROBLEM 2: TRAVELING SALESMAN PROBLEM (HELD-KARP ALGORITHM)
# ============================================================================

class TravelingSalesman:
    """
    Solves the Traveling Salesman Problem using the Held-Karp algorithm.
    
    Uses dynamic programming with subsets to find the shortest Hamiltonian cycle.
    """
    
    def __init__(self, n_cities: int, distance_matrix: List[List[int]]):
        """
        Initialize the TSP solver.
        
        Args:
            n_cities: Number of cities
            distance_matrix: n x n matrix of distances between cities
        """
        self.n = n_cities
        self.dist = distance_matrix
        self.memo = {}
        self.parent = {}
        self.computation_steps = []
    
    def solve(self) -> Tuple[int, List[int]]:
        """
        Solve TSP using Held-Karp algorithm.
        
        Returns:
            Tuple of (minimum_cost, optimal_tour)
        """
        print_header("SOLVING TRAVELING SALESMAN PROBLEM (HELD-KARP ALGORITHM)")
        print_info(f"Number of Cities: {self.n}")
        
        # Display distance matrix
        self._display_distance_matrix()
        
        # Solve using DP
        min_cost = self._held_karp()
        
        # Reconstruct tour
        tour = self._reconstruct_tour()
        
        # Display results
        self._display_subset_computations()
        self._display_final_solution(tour, min_cost)
        
        return min_cost, tour
    
    def _held_karp(self) -> int:
        """
        Held-Karp algorithm implementation.
        
        State: (current_city, visited_set)
        visited_set is represented as a frozenset for hashing
        """
        # Start from city 0
        start_city = 0
        all_cities = set(range(self.n))
        
        # Base case: direct paths from start to each city
        for city in range(1, self.n):
            visited = frozenset([start_city, city])
            self.memo[(city, visited)] = self.dist[start_city][city]
            self.parent[(city, visited)] = start_city
        
        # Build up subsets of increasing size
        for subset_size in range(3, self.n + 1):
            self._process_subsets_of_size(subset_size, start_city)
        
        # Find minimum cost to complete the tour
        min_cost_val: float = float('inf')
        last_city = -1
        
        all_visited = frozenset(all_cities)
        
        for city in range(1, self.n):
            cost = self.memo.get((city, all_visited), float('inf'))
            total_cost = cost + self.dist[city][start_city]
            
            if total_cost < min_cost_val:
                min_cost_val = total_cost
                last_city = city
        
        # Store the final return to start
        self.final_last_city = last_city
        
        return int(min_cost_val)
    
    def _process_subsets_of_size(self, size: int, start_city: int):
        """Process all subsets of given size"""
        from itertools import combinations
        
        cities = list(range(1, self.n))
        
        for subset in combinations(cities, size - 1):
            visited = frozenset([start_city] + list(subset))
            
            # Try ending at each city in the subset
            for last_city in subset:
                self._compute_state(last_city, visited, start_city)
    
    def _compute_state(self, last_city: int, visited: frozenset, start_city: int):
        """Compute optimal cost for a given state"""
        min_cost = float('inf')
        best_prev = -1
        
        step_info = {
            'last_city': last_city,
            'visited': visited,
            'attempts': []
        }
        
        # Try all possible previous cities
        prev_cities = [c for c in visited if c != last_city and c != start_city]
        
        for prev_city in prev_cities:
            prev_visited = visited - {last_city}
            prev_cost = self.memo.get((prev_city, prev_visited), float('inf'))
            
            if prev_cost == float('inf'):
                continue
            
            total_cost = prev_cost + self.dist[prev_city][last_city]
            
            step_info['attempts'].append({
                'prev_city': prev_city,
                'prev_cost': prev_cost,
                'edge_cost': self.dist[prev_city][last_city],
                'total_cost': total_cost
            })
            
            if total_cost < min_cost:
                min_cost = total_cost
                best_prev = prev_city
        
        if min_cost != float('inf'):
            step_info['best_prev'] = best_prev
            step_info['min_cost'] = min_cost
            self.memo[(last_city, visited)] = min_cost
            self.parent[(last_city, visited)] = best_prev
            
            # Store only interesting steps (to avoid overwhelming output)
            if len(visited) <= 4 or len(self.computation_steps) < 10:
                self.computation_steps.append(step_info)
    
    def _reconstruct_tour(self) -> List[int]:
        """Reconstruct the optimal tour from parent pointers"""
        tour = []
        start_city = 0
        
        current_city = self.final_last_city
        visited = frozenset(range(self.n))
        
        # Backtrack through parent pointers
        while len(visited) > 1:
            tour.append(current_city)
            prev_city = self.parent.get((current_city, visited), -1)
            
            if prev_city == -1:
                break
            
            visited = visited - {current_city}
            current_city = prev_city
        
        # Add start city at beginning and end
        tour.reverse()
        tour = [start_city] + tour + [start_city]
        
        return tour
    
    def _display_distance_matrix(self):
        """Display the distance matrix"""
        print_subheader("Input: Distance Matrix")
        
        headers = ["From/To"] + [f"City {i}" for i in range(self.n)]
        rows = []
        
        for i in range(self.n):
            row = [f"City {i}"] + self.dist[i]
            rows.append(row)
        
        print_table(headers, rows, None)
    
    def _display_subset_computations(self):
        """Display subset computation steps"""
        print_subheader("Key Subset Computations (Sample)")
        
        steps_to_show = min(8, len(self.computation_steps))
        shown_steps = self.computation_steps[:steps_to_show]
        
        for i, step in enumerate(shown_steps, 1):
            visited_list = sorted(list(step['visited']))
            print(f"\n{Colors.CYAN}Step {i}: Ending at City {step['last_city']}, Visited = {visited_list}{Colors.END}")
            
            for attempt in step['attempts'][:3]:  # Show top 3
                prev = attempt['prev_city']
                prev_cost = attempt['prev_cost']
                edge_cost = attempt['edge_cost']
                total = attempt['total_cost']
                
                marker = "→" if prev == step.get('best_prev') else " "
                print(f"  {marker} From City {prev}: {prev_cost} + {edge_cost} = {total}")
            
            if 'min_cost' in step:
                print(f"  {Colors.BOLD}Decision: Come from City {step['best_prev']} (Cost: {step['min_cost']}){Colors.END}")
    
    def _display_final_solution(self, tour: List[int], min_cost: int):
        """Display the final optimal tour"""
        print_subheader("OPTIMAL TOUR")
        
        print(f"\n{Colors.BOLD}Path:{Colors.END}")
        path_str = " → ".join([f"City {c}" for c in tour])
        print(f"  {path_str}")
        
        print(f"\n{Colors.BOLD}Detailed Path with Costs:{Colors.END}")
        headers = ["From", "To", "Distance"]
        rows = []
        
        total_distance = 0
        for i in range(len(tour) - 1):
            from_city = tour[i]
            to_city = tour[i + 1]
            distance = self.dist[from_city][to_city]
            total_distance += distance
            rows.append([f"City {from_city}", f"City {to_city}", distance])
        
        print_table(headers, rows, None)
        
        print_success(f"Minimum Tour Cost: {Colors.BOLD}{min_cost}{Colors.END}")
        print_success(f"Optimal Tour: {tour}")


# ============================================================================
# USER INTERACTION & MAIN PROGRAM
# ============================================================================

def get_positive_integer(prompt: str, min_val: int = 1) -> int:
    """Get a positive integer from user with validation"""
    while True:
        try:
            value = int(input(prompt))
            if value >= min_val:
                return value
            else:
                print_warning(f"Please enter a value >= {min_val}")
        except ValueError:
            print_warning("Please enter a valid integer")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            sys.exit(0)


def get_resource_allocation_input():
    """Get input for resource allocation problem"""
    print_header("RESOURCE ALLOCATION PROBLEM - INPUT")
    
    B = get_positive_integer("Enter total resources (B): ")
    n = get_positive_integer("Enter number of activities (n): ")
    
    profit_tables = []
    
    for j in range(n):
        print(f"\n{Colors.BOLD}Activity {j+1}:{Colors.END}")
        print(f"Enter profit values for allocating 0 to {B} resources")
        print(f"(Enter {B+1} values separated by spaces)")
        
        while True:
            try:
                line = input(f"Profits for Activity {j+1}: ")
                profits = list(map(int, line.strip().split()))
                
                if len(profits) != B + 1:
                    print_warning(f"Expected {B+1} values, got {len(profits)}. Try again.")
                    continue
                
                profit_tables.append(profits)
                break
            except ValueError:
                print_warning("Please enter valid integers separated by spaces")
            except KeyboardInterrupt:
                print("\n\nExiting...")
                sys.exit(0)
    
    return B, n, profit_tables


def get_tsp_input():
    """Get input for TSP problem"""
    print_header("TRAVELING SALESMAN PROBLEM - INPUT")
    
    n = get_positive_integer("Enter number of cities: ", min_val=2)
    
    print(f"\n{Colors.BOLD}Enter distance matrix ({n}x{n}):{Colors.END}")
    print("Enter each row on a separate line, with distances separated by spaces")
    print("Use 0 for diagonal elements (distance from city to itself)")
    
    distance_matrix = []
    
    for i in range(n):
        while True:
            try:
                line = input(f"Row {i} (distances from City {i}): ")
                distances = list(map(int, line.strip().split()))
                
                if len(distances) != n:
                    print_warning(f"Expected {n} values, got {len(distances)}. Try again.")
                    continue
                
                distance_matrix.append(distances)
                break
            except ValueError:
                print_warning("Please enter valid integers separated by spaces")
            except KeyboardInterrupt:
                print("\n\nExiting...")
                sys.exit(0)
    
    return n, distance_matrix


def run_example_resource_allocation():
    """Run an example of resource allocation"""
    print_header("EXAMPLE: RESOURCE ALLOCATION")
    print_info("Running with predefined example data...")
    
    # Example: 3 activities, 5 total resources
    B = 5
    n = 3
    profit_tables = [
        [0, 3, 5, 6, 6, 6],  # Activity 1
        [0, 4, 6, 7, 8, 8],  # Activity 2
        [0, 2, 4, 6, 8, 9]   # Activity 3
    ]
    
    solver = ResourceAllocation(B, n, profit_tables)
    max_profit, allocation = solver.solve()
    
    return max_profit, allocation


def run_example_tsp():
    """Run an example of TSP"""
    print_header("EXAMPLE: TRAVELING SALESMAN PROBLEM")
    print_info("Running with predefined example data...")
    
    # Example: 4 cities
    n = 4
    distance_matrix = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]
    
    solver = TravelingSalesman(n, distance_matrix)
    min_cost, tour = solver.solve()
    
    return min_cost, tour


def main_menu():
    """Display main menu and handle user choice"""
    while True:
        print_header("DYNAMIC PROGRAMMING SOLVERS")
        print(f"{Colors.BOLD}Choose a problem to solve:{Colors.END}\n")
        print(f"  {Colors.CYAN}1.{Colors.END} Resource Allocation Problem")
        print(f"  {Colors.CYAN}2.{Colors.END} Traveling Salesman Problem (TSP)")
        print(f"  {Colors.CYAN}3.{Colors.END} Run Resource Allocation Example")
        print(f"  {Colors.CYAN}4.{Colors.END} Run TSP Example")
        print(f"  {Colors.CYAN}5.{Colors.END} Run Both Examples")
        print(f"  {Colors.CYAN}0.{Colors.END} Exit")
        
        try:
            choice = input(f"\n{Colors.BOLD}Enter your choice (0-5): {Colors.END}").strip()
            
            if choice == '1':
                B, n, profit_tables = get_resource_allocation_input()
                solver = ResourceAllocation(B, n, profit_tables)
                solver.solve()
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.END}")
            
            elif choice == '2':
                n, distance_matrix = get_tsp_input()
                solver = TravelingSalesman(n, distance_matrix)
                solver.solve()
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.END}")
            
            elif choice == '3':
                run_example_resource_allocation()
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.END}")
            
            elif choice == '4':
                run_example_tsp()
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.END}")
            
            elif choice == '5':
                run_example_resource_allocation()
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.END}")
                run_example_tsp()
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.END}")
            
            elif choice == '0':
                print(f"\n{Colors.GREEN}Thank you for using the DP Solvers! Goodbye!{Colors.END}\n")
                break
            
            else:
                print_warning("Invalid choice. Please enter a number between 0 and 5.")
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.END}")
        
        except KeyboardInterrupt:
            print(f"\n\n{Colors.GREEN}Thank you for using the DP Solvers! Goodbye!{Colors.END}\n")
            break
        except Exception as e:
            print_warning(f"An error occurred: {e}")
            input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.END}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.GREEN}Thank you |Dev by : Adnane Malki !{Colors.END}\n")
        sys.exit(0)
