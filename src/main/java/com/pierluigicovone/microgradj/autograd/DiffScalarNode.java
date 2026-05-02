package com.pierluigicovone.microgradj.autograd;

import java.util.Objects;

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
     * Assuming two instances of the DiffScalarNode class, a and b;
     * This method is the equivalent of:    a + b.
     */
    public DiffScalarNode add(DiffScalarNode other) {
        return new DiffScalarNode(data + other.data);
    }

    /**
     * Assuming an instance "a" of the DiffScalarNode class.
     * This method is the equivalent of:
     *                  a + constant,
     * where "constant" is any instance of number.
     */
    public DiffScalarNode add(Number other) {
        return new DiffScalarNode(data + other.doubleValue());
    }


    /**
     * Assuming two instances of the DiffScalarNode class, a and b;
     * This method is the equivalent of:    a - b.
     */
    public DiffScalarNode sub(DiffScalarNode other) {
        return new DiffScalarNode(data - other.data);
    }

    /**
     * Assuming an instance "a" of the DiffScalarNode class.
     * This method is the equivalent of:
     *                  a - constant,
     * where "constant" is any instance of number.
     */
    public DiffScalarNode sub(Number other) {
        return new DiffScalarNode(data - other.doubleValue());
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

    @Override
    public boolean equals(Object o) {

        // This method should be updated each time a new field is inserted.

        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;

        DiffScalarNode object = (DiffScalarNode) o;

        // We can use the "==" operator because they're double values.
        return data == object.data
                && grad == object.grad;
     }

    @Override
    public int hashCode() {
        return Objects.hash(data, grad);
    }

}