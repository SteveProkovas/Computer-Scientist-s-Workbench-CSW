from typing import List, Tuple, Dict
import numpy as np

class KnapsackSolver:
    """
    Solves the 0/1 Knapsack problem using four equivalent representations:
    1. DFS (top-down) with state: (current_index, remaining_capacity)
    2. DP (bottom-up) with same state representation
    3. DFS (top-down) with state: (items_processed, total_capacity)
    4. DP (bottom-up) with classic DP formulation (items_processed, total_capacity)
    
    Mathematical Formulation:
    Maximize: Σ v_i·x_i
    Subject to: Σ w_i·x_i ≤ C, x_i ∈ {0,1}
    where:
        v_i = value of item i
        w_i = weight of item i
        C = total capacity
        x_i = decision variable (1 if taken, 0 otherwise)
    """
    
    def __init__(self, weights: List[int], values: List[int], capacity: int):
        self.weights = weights
        self.values = values
        self.capacity = capacity
        self.n = len(weights)
        
    # ===================================================================
    # Representation A: State = (current_index, remaining_capacity)
    # ===================================================================
    
    def dfs_a(self) -> int:
        """Top-down DFS: State represents current index and remaining capacity"""
        memo = {}
        
        def dfs(i: int, c: int) -> int:
            # Base case: end of items or capacity exhausted
            if i >= self.n or c <= 0:
                return 0
                
            # Check memoization
            if (i, c) in memo:
                return memo[(i, c)]
            
            # Skip current item
            skip = dfs(i + 1, c)
            
            # Take current item (if feasible)
            take = -10**9
            if c >= self.weights[i]:
                take = self.values[i] + dfs(i + 1, c - self.weights[i])
            
            # Optimal decision at this state
            memo[(i, c)] = max(skip, take)
            return memo[(i, c)]
        
        return dfs(0, self.capacity)
    
    def dp_a(self) -> int:
        """Bottom-up DP: Direct translation of DFS_A state representation"""
        # dp[i][c] = max value from items[i..n-1] with capacity c
        dp = np.full((self.n + 1, self.capacity + 1), -10**9)
        
        # Base cases: 0 value at end of items
        dp[self.n, :] = 0
        
        # Fill table backwards (from last item to first)
        for i in range(self.n - 1, -1, -1):
            for c in range(self.capacity + 1):
                skip = dp[i + 1][c]
                take = -10**9
                if c >= self.weights[i]:
                    take = self.values[i] + dp[i + 1][c - self.weights[i]]
                dp[i][c] = max(skip, take)
        
        return dp[0][self.capacity]
    
    # ===================================================================
    # Representation B: State = (items_processed, total_capacity)
    # ===================================================================
    
    def dfs_b(self) -> int:
        """Top-down DFS: State represents items processed and total capacity"""
        memo = {}
        
        def dfs(k: int, w: int) -> int:
            # Base case: no items processed
            if k == 0:
                return 0
                
            # Check memoization
            if (k, w) in memo:
                return memo[(k, w)]
            
            # Skip k-th item (items are 0-indexed)
            skip = dfs(k - 1, w)
            
            # Take k-th item (if feasible)
            take = -10**9
            if w >= self.weights[k - 1]:
                take = self.values[k - 1] + dfs(k - 1, w - self.weights[k - 1])
            
            # Optimal decision at this state
            memo[(k, w)] = max(skip, take)
            return memo[(k, w)]
        
        return dfs(self.n, self.capacity)
    
    def dp_b(self) -> int:
        """Bottom-up DP: Classic Knapsack formulation (items processed)"""
        # dp[k][w] = max value using first k items with capacity w
        dp = np.zeros((self.n + 1, self.capacity + 1))
        
        # Build table incrementally
        for k in range(1, self.n + 1):
            for w in range(self.capacity + 1):
                skip = dp[k - 1][w]
                take = -10**9
                if w >= self.weights[k - 1]:
                    take = self.values[k - 1] + dp[k - 1][w - self.weights[k - 1]]
                dp[k][w] = max(skip, take)
        
        return int(dp[self.n][self.capacity])
    
    # ===================================================================
    # Unified Solution Verification
    # ===================================================================
    
    def solve_all(self) -> Dict[str, int]:
        """Solve using all four methods and verify consistency"""
        results = {
            "DFS_A": self.dfs_a(),
            "DP_A": self.dp_a(),
            "DFS_B": self.dfs_b(),
            "DP_B": self.dp_b()
        }
        
        # Verify all solutions match
        unique_results = set(results.values())
        if len(unique_results) != 1:
            raise ValueError(f"Inconsistent results: {results}")
            
        return results


def run_test_suite():
    """Comprehensive test suite for Knapsack implementations"""
    test_cases = [
        # Basic case
        ([2, 3, 4, 5], [3, 4, 5, 6], 8, 10),
        # Exact fit
        ([1, 2, 3], [6, 10, 12], 5, 22),
        # No solution
        ([5, 6, 7], [10, 20, 30], 4, 0),
        # Zero capacity
        ([1, 2, 3], [4, 5, 6], 0, 0),
        # Zero items
        ([], [], 10, 0),
        # Heavy items
        ([10, 20, 30], [60, 100, 120], 50, 220),
        # Fractional temptation (but integer solution)
        ([3, 4, 5], [30, 50, 60], 8, 90),
        # Large values
        ([2, 4, 6], [100, 1000, 10000], 6, 10000),
        # Duplicate items
        ([3, 3, 3], [5, 5, 5], 6, 10),
    ]
    
    for i, (weights, values, capacity, expected) in enumerate(test_cases):
        print(f"\nTest Case #{i+1}:")
        print(f"Items: {list(zip(weights, values))}")
        print(f"Capacity: {capacity} | Expected: {expected}")
        
        solver = KnapsackSolver(weights, values, capacity)
        results = solver.solve_all()
        
        for method, value in results.items():
            print(f"{method}: {value} | {'PASS' if value == expected else 'FAIL'}")
        
        print(f"Consistency: {'ALL MATCH' if len(set(results.values())) == 1 else 'ERROR'}")


if __name__ == "__main__":
    run_test_suite()
