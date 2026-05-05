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
    private final double data;
    private double grad;

    private BackwardOp backwardOp;

    // --- Graph Construction ---
    private final Set<DiffScalarNode> parents;      // Set is useful to walk through the graph
    private final String operation;                 // The op. that produced this node.

    // --- Visualization Fields
    private final Double constant;              // Operations with constants.
    private String variableName;                // To pretty format leafs, in the graph viz.


    // ----- METHODS -----

    /**
     * Private constructor that initialize fields.
     */
    private DiffScalarNode(double data, Set<DiffScalarNode> parents, String operation, Double constant, String variableName) {
        this.data = data;

        this.parents = Objects.requireNonNull(parents);
        this.operation = operation;

        // This is non-null iif the related operation involved a constant.
        this.constant = constant;

        this.variableName = variableName;

        this.backwardOp = BackwardOp.NO_OP;       // By default, do nothing;
    }

    /**
     * Factory method to create leafs (that are nodes created by users).
     */
    public static DiffScalarNode leaf(double data, String variableName) {
        return new DiffScalarNode( data, Set.of(), "", null, variableName);
    }

    /**
     * Factory method to create nodes from operations between two "DiffScalarNode" instances,
     * or between a "DiffScalarNode" instance and a scalar.
     * This method is static because of coherence with the public "leaf" method.
     */
    private static DiffScalarNode fromOperation(double data, Set<DiffScalarNode> parents,
                                                String operation,
                                                Double constant) {
        return new DiffScalarNode( data, parents, operation, constant, "");
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

        // First create the "output" node
        DiffScalarNode out = DiffScalarNode.fromOperation(
                data + other.data,
                Set.copyOf( List.of(this, other) ),
                "+",
                null);

        // Then define the backpropagation function.
        BackwardOp op = () -> {
                this.grad += out.grad;
                other.grad += out.grad;
        };

        // Modify the default behaviour.
        out.backwardOp = op;

        return out;
    }

    /**
     * Add a "DiffScalarNode" instance with a scalar.
     */
    public DiffScalarNode add(Number other) {

        // Taking the double value
        double otherData = other.doubleValue();

        // Creating the output node
        DiffScalarNode out = DiffScalarNode.fromOperation(data + otherData, Set.of( this ), "+", otherData);

        // Define the backpropagation function.
        BackwardOp op = () -> {
            this.grad += out.grad;
        };

        // Modify the default behaviour.
        out.backwardOp = op;

        return out;
    }


    // --- Multiplication
    /**
     * Multiply two "DiffScalarNode" instances.
     */
    public DiffScalarNode mul(DiffScalarNode other) {

        DiffScalarNode out = DiffScalarNode.fromOperation(
                data * other.data,
                Set.copyOf( List.of( this, other ) ),
                "*" ,
                null);

        // Define the backpropagation function.
        BackwardOp op = () -> {
            this.grad += other.data * out.grad;
            other.grad += this.data * out.grad;
        };

        // Modify the default behaviour.
        out.backwardOp = op;

        return out;
    }

    /**
     * Multiply a "DiffScalarNode" instance with a scalar.
     */
    public DiffScalarNode mul(Number other) {

        // Using the double value
        double otherData = other.doubleValue();

        DiffScalarNode out = DiffScalarNode.fromOperation(data * otherData, Set.of( this ) , "*" , otherData);

        // Define the backpropagation function.
        BackwardOp op = () -> {
            this.grad += otherData * out.grad;
        };

        // Modify the default behaviour.
        out.backwardOp = op;

        return out;
    }



    // --- Exponential
    /**
     * Exponentiation given a constant.
     */
    public DiffScalarNode pow (Number other) {

        double otherData = other.doubleValue();

        DiffScalarNode out =  DiffScalarNode.fromOperation(
                Math.pow(data, otherData),
                Set.of(this),
                "^",
                otherData
        );

        // Define the backpropagation function.
        BackwardOp op = () -> {
            this.grad +=  ( otherData * Math.pow(data, otherData-1) ) * out.grad ;
        };

        // Modify the default behaviour.
        out.backwardOp = op;

        return out;
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

    /**
     * Get the variable name.
     */
    public String getVariableName() {
        return variableName;
    }

    /**
     * Modify the name of the node on which is invoked.
     */
    public DiffScalarNode withName(String newName) {
        this.variableName = newName;
        return this;  // Return the same object
    }

    // --- Overrides ---

    @Override
    public String toString() {
        return String.format("DiffScalarNode(data=%s, grad=%s, op=%s)",
                data,grad, operation.isEmpty() ? "leaf" : operation );
    }

}