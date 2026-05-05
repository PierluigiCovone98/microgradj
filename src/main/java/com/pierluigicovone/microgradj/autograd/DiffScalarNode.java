package com.pierluigicovone.microgradj.autograd;

import java.security.DigestException;
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

    // --- Visualization Fields
    private final Double constant;              // Operations with constants.
    // private final String variableName;

    // ----- METHODS -----

    /**
     * Private constructor that initialize fields.
     */
    private DiffScalarNode(double data, Set<DiffScalarNode> parents, String operation, Double constant) {
        this.data = data;

        this.parents = Objects.requireNonNull(parents);
        this.operation = operation;

        // This is non-null iif the related operation involved a constant.
        this.constant = constant;

    }

    /**
     * Factory method to create leafs (that are nodes created by users).
     */
    public static DiffScalarNode leaf(double data) {
        // A leaf has parents neither operations from which it is created.
        return new DiffScalarNode( data, Set.of(), "", null);
    }

    /**
     * Factory method to create nodes from operations between two "DiffScalarNode" instances,
     * or between a "DiffScalarNode" instance and a scalar.
     * This method is static because of coherence with the public "leaf" method.
     */
    private static DiffScalarNode fromOperation(double data, Set<DiffScalarNode> parents, String operation, Double constant) {
        return new DiffScalarNode( data, parents, operation, constant);
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
                Set.copyOf( List.of(this, other) ),
                null
        );
    }

    /**
     * Add a "DiffScalarNode" instance with a scalar.
     */
    public DiffScalarNode add(Number other) {
        double otherData = other.doubleValue();

        return add(data,
                otherData,
                Set.of( this ),
                otherData
        );
    }

    /**
     * Avoid the repetition of using "+" in both public "add" methods.
     */
    private DiffScalarNode add(double thisData, double otherData, Set<DiffScalarNode> parents, Double constant) {
       return DiffScalarNode.fromOperation(thisData + otherData, parents, "+", constant);

    }

    // --- Multiplication
    /**
     * Multiply two "DiffScalarNode" instances.
     */
    public DiffScalarNode mul(DiffScalarNode other) {
        return mul(data,
                other.data,
                Set.copyOf( List.of(this, other) ),
                null
        );
    }

    /**
     * Multiply a "DiffScalarNode" instance with a scalar.
     */
    public DiffScalarNode mul(Number other) {
        double otherData = other.doubleValue();

        return mul( data,
                otherData,
                Set.of(this),
                otherData
        );
    }

    /**
     * Avoid the repetition of the "*" in both "mul" operations.
     */
    private DiffScalarNode mul(double thisData, double otherData, Set<DiffScalarNode> parents, Double constant) {
        return DiffScalarNode.fromOperation(thisData * otherData, parents, "*" , constant);
    }

    // --- Exponential
    /**
     * Exponentiation given a constant.
     */
    public DiffScalarNode pow (Number other) {
        double otherData = other.doubleValue();

        return DiffScalarNode.fromOperation(
                Math.pow(data, otherData),
                Set.of(this),
                "^",
                otherData
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
        // Redundant conversion to double of "other".
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

    /**
     * Get constant.
     */
    public Double getConstant() {
        return constant;
    }

    /**
     * Check if is there a constant.
     */
    public boolean hasConstant() {
        return constant != null;
    }


    // --- Overrides ---

    @Override
    public String toString() {
        return String.format("DiffScalarNode(data=%s, grad=%s, op=%s)",
                data,grad, operation.isEmpty() ? "leaf" : operation );
    }

}