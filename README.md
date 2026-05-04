# microgradj

A Java reimplementation of [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd), built for educational purposes while following the *Neural Networks: from Zero to Hero* video series.

The goal is **not** to provide a production-ready autograd engine, but to understand — by reimplementing it — how reverse-mode automatic differentiation actually works under the hood, on simple scalar values.

## What this project is

`microgradj` is a tiny scalar-valued automatic differentiation engine. Each numeric value is wrapped in a `DiffScalarNode` that:

- carries its scalar `data`;
- accumulates a `grad` (gradient) during backpropagation;
- remembers the operation that produced it and its parent nodes in the computation graph.

From these primitives, the project will progressively build up the components of a neural network (neurons, layers, MLPs) following the same incremental approach as Karpathy's video.

## Project status

Implemented so far:

- `DiffScalarNode` — the core differentiable scalar node.
- Atomic operations: addition, multiplication, power (with constant exponent).
- Derived operations: subtraction, division, negation.
- Computation graph traversal (topological sort).
- Graph visualization to PNG via [`graphviz-java`](https://github.com/nidi3/graphviz-java).

Not yet implemented (coming next):

- `tanh` activation function.
- Backpropagation (`backward()` method).
- Neural network primitives (`Neuron`, `Layer`, `MLP`).
- Recurrent architectures.

## Requirements

- **Java 24** or compatible JDK
- **Maven 3.9+**
- A computer running macOS or Linux (the scripts use bash; on Windows use the IDE directly)

## How to run

### From the command line (recommended for quick demos)

A convenience script runs any test by its number:

```bash
./scripts/run.sh <test-number>
```

For example:

```bash
./scripts/run.sh 2
```

This compiles the project (if needed) and runs `Test2`, which produces a `graph.png` in the project root showing the computation graph.

If you pass a number that doesn't correspond to any test, the script lists the available tests:

```bash
./scripts/run.sh 99
# Error: test file not found: src/main/java/com/pierluigicovone/microgradj/examples/Test99.java
# Available tests:
#   Test1.java
#   Test2.java
```

### Cleaning up

To remove compiled artifacts and generated graph images:

```bash
./scripts/reset.sh
```

### From an IDE

Open the project as a Maven project. Each `TestN.java` file in `examples/` has a `main()` method and can be run directly (right-click → Run, or the green ▶ in the gutter).

## Why "atomic" and "derived" operations?

Following Karpathy's design, only a minimal set of operations is implemented as **atomic** (with their own forward and, in the future, backward logic):

- addition (`+`)
- multiplication (`*`)
- power with constant exponent (`^c`)

All other operations are **derived** from these:

- `neg(x) = x * (-1)`
- `sub(a, b) = a + neg(b)`
- `div(a, b) = a * b^(-1)`

This keeps the engine small, easier to reason about, and ensures correctness: if the three atomic operations are correct, all derived ones are correct by construction.

## Acknowledgments

This project is inspired by [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd) and his [Neural Networks: from Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) video series.
