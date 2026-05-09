package com.pierluigicovone.microgradj.autograd;

import java.util.Objects;
import java.util.Set;
import java.util.List;
import java.util.function.Function;

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

        /*
          Let's use an example:
            We have that
                (i) c = a + b,
            and "a" and "b" are tyne nodes in a bigger graph.

            Let's say that the "output node" is L;
            and what we want is to express values
                "a.grad" and "b.grad"
            as derivatives respect to "L". To be more precise:
                - a.grad = dL / da
                - b.grad = dL / db

            But as we can see, "L" is not "directly connected" to
            both nodes (I mean: at least there is the node "c"
            between them); so:
                how can we express "dL / da" (or "dL / db") if
                they are not directly connected?

            Here the "chain role" is what helps us (see the definition).

            Now, when we want to calculate "a.grad" and "b.grad", of course
            we already know the value of "c.grad" (because we started to calculate
            the gradient of nodes from the last node "L",  of our graph).

            And here we are:
                - a.grad = (dL / dc) * (dc / da)
                - b.grad = (dL / dc) * (dc / db)

            And, because from (i) we can state that
                - (dc / da) = 1
                - (dc / db) = 1
            we say that:
                - a.grad = (dL / dc) * 1
                - b.grad = (dL / dc) * 1

            With the "+" operation, the local derivative is always 1.
            This means that:
                - a.grad = (dL / dc)
                - b.grad = (dL / dc)

            The usage of "+=" is because we could also have
                (ii) c = a + a;
            In that case we have:
                - a.grad = 2 * (dL / dc)
            that, translated in "coding", it means:
                - this.grad += (dL / dc)
                - other.grad += (dL / dc)
            where "this" and "other" are the same instance.
         */
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

        // Define how calculate the gradient for "parent" nodes,
        // knowing the gradient of the child "output".
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

        /*
          Let's use an example:
            We have that
                (i) c = a * b,
            and "a" and "b" are tyne nodes in a bigger graph.

            Let's say that the "output node" is L;
            and what we want is to express values
                "a.grad" and "b.grad"
            as derivatives respect to "L". To be more precise:
                - a.grad = dL / da
                - b.grad = dL / db

            But as we can see, "L" is not "directly connected" to
            both nodes (I mean: at least there is the node "c"
            between them); so:
                how can we express "dL / da" (or "dL / db") if
                they are not directly connected?

            Here the "chain role" is what helps us (see the definition).

            Now, when we want to calculate "a.grad" and "b.grad", of course
            we already know the value of "c.grad" (because we started to calculate
            the gradient of nodes from the last node "L",  of our graph).

            And here we are:
                - a.grad = (dL / dc) * (dc / da)
                - b.grad = (dL / dc) * (dc / db)

            And, because from (i) we can state that
                - (dc / da) = b
                - (dc / db) = a
            we say that:
                - a.grad = (dL / dc) * b
                - b.grad = (dL / dc) * a

            In terms of coding, we translate it as:
                - a.grad += (dL / dc) * b.data
                - b.grad += (dL / dc) * a.data

            The usage of "+=" is the same specified for the "+" case.
        */
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

        // Define how calculate the gradient for "parents" node,
        // knowing the gradient of the child "output".
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

        // Define how calculate the gradient for "parents" node,
        // knowing the gradient of the child "output".
        BackwardOp op = () -> {
            this.grad +=  ( otherData * Math.pow(data, otherData-1) ) * out.grad ;
        };

        // Modify the default behaviour.
        out.backwardOp = op;

        return out;
    }

    // --- tanh

    /**
     * This is a "non-atomic" implementation of the non-linear tanh function;
     * what does it mean?
     * If we look at the "tanh" definition
     *      tanh := ( e^x - e^(-x) ) / ( e^x + e^(-x) )
     * we could "build" piece by piece each component.
     * For example, we could write
     *      tanh := ( a - b ) / ( a + b ),
     * where
     *      DiffScalarNode a = e^x,
     *      DiffScalarNode b = e^(-x)
     *
     * But we are not interested in "how atomic the function is";
     * what we want is that we know how to locally derive it.
     * So: we pre-compute the value "data" using non-DiffScalarNode
     * derived (or atomic) function.
     */
    public DiffScalarNode tanh() {

        // Given a certain value "x", compute "tanh(x)".
        Function<Double, Double> computeTanh = x -> {

            double e1 =  Math.exp(x);   // e^x
            double e2 = Math.exp(-x);   // e^(-x)

            // tanh := ( e^x - e^(-x) ) / ( e^x + e^(-x) )
            return ( e1 - e2 ) / ( e1 + e2 );
        };

        // Forward pass: compute "tanh( this.data )".
        double tanhValue = computeTanh.apply( this.data );

        // Create the output value.
        DiffScalarNode out = DiffScalarNode.fromOperation(
                tanhValue,
                Set.of(this),
                "tanh",
                null
        );

        /*
          We have:
                o = tanh(n)
          This means that, for "backpropagation purposes" we want to calculate
          the derivate of "o" respect to "n";
          this means that we want to set the value of:
                n.grad
          But, terms of coding, this means that we have to set
                this.grad = do / dn
                          = [...]
                          = 1 - tanh(n)^2
         */
        BackwardOp op = () -> {
            // Here also "+" is ok.
            this.grad +=  1 - Math.pow(tanhValue, 2);
        };

        // Setting the "op" as the "backwardOp" field
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