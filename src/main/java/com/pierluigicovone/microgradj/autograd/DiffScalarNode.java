package com.pierluigicovone.microgradj.autograd;

import java.util.Objects;
import java.util.Set;
import java.util.List;

/**
 * Building Block of the automatic differentiation engine.
 *
 * Wraps a scalar value and tracks the operation that produced it,
 * along with references to its parent nodes in the computation graph.
 *
 * Each node is differentiable and knows how to propagate its gradient
 * backward to its parents, enabling reverse-mode automatic differentiation.
 */
public class DiffScalarNode {

    // ----- FIELDS -----
    private final double data;      // <Is it really final???>
    private double grad;

    // --- Graph Construction ---
    private final Set<DiffScalarNode> parents;      // Set is useful to walk through the graph
    private final String operation;                 // The op. that produced this node.


    // ----- METHODS -----

    /**
     * Private constructor that initialize fields.
     */
    private DiffScalarNode(double data, Set<DiffScalarNode> parents, String operation) {
        this.data = data;

        this.parents = Objects.requireNonNull(parents);
        this.operation = operation;

    }

    /**
     * Factory method to create leafs (that are nodes created by users).
     */
    public static DiffScalarNode leaf(double data) {
        // A leaf has parents neither operations from which it is created.
        return new DiffScalarNode( data, Set.of(), "");
    }

    /**
     * (Private) Factory method to create nodes from operations.
     * This method is static because of coherence with the public "leaf" method.
     */
    private static DiffScalarNode fromOperation(double data, Set<DiffScalarNode> parents, String operation) {
        return new DiffScalarNode( data, parents, operation);
    }


    // --- Atomic Operations ---
    //   a + b      (and: a + c)
    //   a * b      (and: a * c)
    //   a^c        (with c constant)

    // --- Addition
    /**
     * Add two "DiffScalarNode" instances.
     */
    public DiffScalarNode add(DiffScalarNode other) {
        return add( data,
                other.data,
                Set.copyOf( List.of(this, other) )
        );
    }

    /**
     * Add a "DiffScalarNode" instance with a scalar.
     */
    public DiffScalarNode add(Number other) {
        return add(data,
                other.doubleValue(),
                Set.of( this )
        );
    }

    /**
     * Avoid the repetition of using "+" in both public "add" methods.
     */
    private DiffScalarNode add(double thisData, double otherData, Set<DiffScalarNode> parents) {
        return DiffScalarNode.fromOperation(thisData + otherData, parents, "+");
    }

    // --- Multiplication
    /**
     * Multiply two "DiffScalarNode" instances.
     */
    public DiffScalarNode mul(DiffScalarNode other) {
        return mul(data,
                other.data,
                Set.copyOf( List.of(this, other) )
        );
    }

    /**
     * Multiply a "DiffScalarNode" instance with a scalar.
     */
    public DiffScalarNode mul(Number other) {
        return mul( data,
                other.doubleValue(),
                Set.of(this)
        );
    }

    /**
     * Avoid the repetition of the "*" in both "mul" operations.
     */
    private DiffScalarNode mul(double thisData, double otherData, Set<DiffScalarNode> parents) {
        return DiffScalarNode.fromOperation(thisData * otherData, parents, "*" );
    }

    // --- Exponential
    /**
     * Exponentiation given a constant.
     */
    public DiffScalarNode pow (Number other) {
        return DiffScalarNode.fromOperation(
                Math.pow(data, other.doubleValue()),
                Set.of(this),
                "^"
        );
    }


    // --- Derived Operations ----

    // --- Negative
    public DiffScalarNode neg() {
        return this.mul(-1);
    }

    // --- Subtraction
    /**
     * Subtract two "DiffScalarNode" instances.
     */
    public DiffScalarNode sub(DiffScalarNode other) {
        return this.add( other.neg() );
    }

    /**
     * Subtract a "DiffScalarNode" instance by a scalar.
     */
    public DiffScalarNode sub(Number other) {
        return this.add( -other.doubleValue() );
    }

    // --- Division
    /**
     * Divide two "DiffScalarNode" instances.
     */
    public DiffScalarNode div(DiffScalarNode other) {
        return this.mul( other.pow(-1) );
    }

    /**
     * Divide a "DiffScalarNode" instance by a scalar.
     */
    public DiffScalarNode div(Number other) {
        //      DiffScalarValue / number
        //      DiffScalarValue * number^(-1)
        //      DiffScalarValue * 1/number
        return this.mul( 1/other.doubleValue() );
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

    /**
     * Get parents.
     */
    public Set<DiffScalarNode> getParents() {
        return parents;
    }


    // --- Overrides ---

    @Override
    public String toString() {
        return String.format("DiffScalarNode(data=%s, grad=%s, op=%s)",
                data,grad, operation.isEmpty() ? "leaf" : operation );
    }

}