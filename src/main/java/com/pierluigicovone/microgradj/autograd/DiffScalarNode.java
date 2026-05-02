package com.pierluigicovone.microgradj.autograd;

/**
 * Building Block of the automatic differentiation engine.
 *
 * Wraps a scalar value and tracks the operation that produced it,
 * along with references to its parent nodes in the computation graph.
 *
 * Each node id differentiable and knows how to propagate its gradient
 * backward to its parents, enabling reverse-mode automatic differentiation.
 */
public class DiffScalarNode {

    // --- FIELDS ---
    private final double data;      // <Is it really final???>
    private double grad;


    // --- METHODS ---

    /**
     * Takes as parameter in input the scalar value "data" (temporarily...).
     */
    public DiffScalarNode(double data) {
        // initialize fields
        this.data = data;
    }

    /**
     * Get the "data" value.
     */
    public double getData() {
        return data;
    }

    /**
     * Get the "gradient" value.
     */
    public double getGrad() {
        return grad;
    }

    /**
     * Set the gradient.
     */
    public void setGrad(double grad) {
        this.grad = grad;
    }


    // --- OVERRIDES ---

    @Override
    public String toString() {
        return String.format("DiffScalarNode(data=%s, grad=%s)", data,grad);
    }

}