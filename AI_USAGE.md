# AI Usage Log

This document records examples of AI-assisted development during the project. AI tools were used to accelerate development, understand RVV concepts, improve documentation, and debug implementation issues. All generated suggestions were reviewed, tested, and modified before integration.

---

## Entry 1 – Understanding RVV Strip Mining

### Question Asked

What's the difference between the L1 and L2 magnitude norm, and why implement both?

### AI Suggestion

Explained L1 (|Gx|+|Gy|) is faster with no floating point, while L2 (sqrt(Gx²+Gy²)) is the true Euclidean distance and gives cleaner edges but costs more computation due to the square root.

### What We Changed

Added a comment in our code explaining L1 ≥ L2 always holds mathematically, which we then used as a property-based unit test instead of needing a separate reference image.

### What We Learned

Choosing between L1 and L2 in embedded systems is a real engineering tradeoff between accuracy and computational cost — not just an arbitrary implementation detail.

---

## Entry 2 – Debugging Build and Toolchain Issues

### Question Asked

What do VLEN and LMUL mean in RVV, and how do they affect our code?

### AI Suggestion

Explained VLEN as the physical width of a vector register in bits (set by hardware/QEMU), and LMUL as how many registers are grouped together in software to hold wider data types.

### What We Changed

Fixed an LMUL mismatch where we tried to combine an m2 (int16) vector directly with an m4 (int32) vector without the required widening step in between.

### What We Learned

LMUL must match between operands used together in the same RVV operation, and widening operations always change the LMUL of the output (e.g. m2 → m4).

---

## Entry 3 – RVV Magnitude Optimization

### Question Asked

How can the gradient magnitude computation be vectorized using RVV intrinsics?

### AI Suggestion

The AI suggested loading multiple gradient values into vector registers, performing parallel arithmetic operations, and storing results back to memory.

### What We Changed

We adapted the approach to fit our implementation and verified correctness against the scalar reference output.

### What We Learned

We learned how data-level parallelism can significantly accelerate image-processing kernels.

---

## Entry 4 – Unit Testing Strategy

### Question Asked

What types of tests should be written for Gaussian blur, Sobel filtering, and magnitude computation?

### AI Suggestion

The AI recommended using synthetic images, constant-value images, edge cases, and known outputs to validate correctness.

### What We Changed

We created test cases covering multiple pipeline stages and compared outputs against expected results.

### What We Learned

We learned how systematic testing improves confidence in algorithm correctness before optimization.

---

## Entry 5 – README and Documentation

### Question Asked

How should a professional GitHub README be structured for an embedded systems project?

### AI Suggestion

The AI recommended including project objectives, pipeline descriptions, build instructions, testing procedures, optimization details, and performance results.

### What We Changed

We reorganized the documentation and added detailed explanations of the pipeline stages and project structure.

### What We Learned

We learned how good documentation improves project readability, maintainability, and reproducibility.

---

## Reflection

AI tools were used as a learning and productivity aid throughout the project. All suggestions were reviewed, tested, and modified when necessary. The team ensured that every piece of code included in the final submission was fully understood and could be explained during project discussion and evaluation.
