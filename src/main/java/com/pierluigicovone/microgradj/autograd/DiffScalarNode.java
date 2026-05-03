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

    // ----- FIELDS -----
    private final double data;      // <Is it really final???>
    private double grad;

    private final String operation;     // The op. that produced this node.





    // ----- METHODS -----

    /**
     * Constructor takes as parameter in input the scalar value "data"
     * and set the "operation" field as the default value of "".
     */
    public DiffScalarNode(double data) {
        this.data = data;
        this.operation = "";
    }

    /**
     * Constructor takes as parameter in input the scalar value "data" and the operation type.
     */
    public DiffScalarNode(double data, String operation) {

        this.data = data;
        this.operation = operation;

    }


    // --- Operations ---

    /**
     * Assuming two instances of the DiffScalarNode class, a and b;
     * This method is the equivalent of:    a + b.
     */
    public DiffScalarNode add(DiffScalarNode other) {
        return sum(data, other.data);
    }

    /**
     * This method is the equivalent of:        a + constant,
     * where "constant" is any instance of number.
     */
    public DiffScalarNode add(Number other) {
        return sum(data, other.doubleValue());
    }

    /**
     * Avoid the repetition of using "+" in both public "sum" methods.
     */
    private DiffScalarNode sum(double thisData, double otherData) {
        return new DiffScalarNode(thisData + otherData, "+");
    }


    /**
     * Assuming two instances of the DiffScalarNode class, a and b;
     * This method is the equivalent of:    a - b.
     */
    public DiffScalarNode sub(DiffScalarNode other) {
        return sub(data, other.data);
    }

    /**
     * This method is the equivalent of:        a - constant,
     * where "constant" is any instance of number.
     */
    public DiffScalarNode sub(Number other) {
        return sub(data, other.doubleValue());
    }

    /**
     * Avoid the repetition of using "-" in both public "sum" methods.
     */
    private DiffScalarNode sub(double thisData, double otherData) {
        return new DiffScalarNode(thisData + otherData, "-");
    }


    // --- Getters & Setters ---

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

    /**
     * Get the "operation".
     */
    public String getOperation() {
        return operation;
    }


    // --- Overrides ---

    @Override
    public String toString() {
        return String.format("DiffScalarNode(data=%s, grad=%s, op=%s)",
                data,grad, operation.isEmpty() ? "leaf" : operation );
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