# 🚀 Quick Start Guide

## Running the Program

Simply run:
```bash
python main.py
```

## Menu Options

### Option 1: Resource Allocation Problem (Custom Input)
You'll be asked to enter:
- Total resources (B)
- Number of activities (n)
- Profit table for each activity

Example input:
```
Total resources (B): 5
Number of activities (n): 3

Activity 1 profits: 0 3 5 6 6 6
Activity 2 profits: 0 4 6 7 8 8
Activity 3 profits: 0 2 4 6 8 9
```

### Option 2: Traveling Salesman Problem (Custom Input)
You'll be asked to enter:
- Number of cities
- Distance matrix (n×n)

Example input:
```
Number of cities: 4

Row 0: 0 10 15 20
Row 1: 10 0 35 25
Row 2: 15 35 0 30
Row 3: 20 25 30 0
```

### Option 3: Run Resource Allocation Example
Runs a predefined example with 3 activities and 5 resources.

### Option 4: Run TSP Example
Runs a predefined example with 4 cities.

### Option 5: Run Both Examples
Runs both predefined examples sequentially.

## Expected Output Features

✅ **Beautiful Colored Output** (ANSI colors)
✅ **Formatted Tables** with borders
✅ **Step-by-step Computation Visualization**
✅ **Memoization Table Display**
✅ **Optimal Solution Breakdown**
✅ **Input Validation** with helpful error messages

## Example Output Screenshot

When you run the examples, you'll see:

1. **Headers** with centered text and borders
2. **Info messages** with icons (ℹ, ✓, ⚠)
3. **Tables** with box-drawing characters
4. **Color-coded decisions** (highlighted in cyan/green)
5. **Final optimal solution** clearly displayed

## Tips

- Press `Ctrl+C` at any time to exit gracefully
- Input validation ensures you enter correct data
- All examples are pre-configured for quick testing
- Read the detailed README.md for algorithm explanations

## What You'll Learn

📚 **Dynamic Programming Concepts:**
- State definition and transitions
- Memoization techniques
- Solution reconstruction

🎯 **Optimization Problems:**
- Resource allocation strategies
- Shortest path problems (TSP)

💻 **Clean Code Practices:**
- Object-oriented design
- User-friendly interfaces
- Professional output formatting

---

**Enjoy exploring dynamic programming! 🎉**
