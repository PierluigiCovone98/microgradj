package com.pierluigicovone.microgradj.autograd;

/**
 * We use this functional interface as a placeholder,
 * such that each node can have a field containing its own function
 * for backpropagation.
 */
@FunctionalInterface
public interface BackwardOp {

    // Empty implementation of the "BackwardOp".
    BackwardOp NO_OP = () -> {};

    void propagate();
}
