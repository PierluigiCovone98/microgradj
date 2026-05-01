package com.pierluigicovone.microgradj.autograd;

/**
 *
 */
public class GradScalar {

    // FIELDS
    private final double data;    // 0.0 as default.        <Is that really final?>

    private double grad;    // 0.0 as default
    // private String op;


    // METHODS

    /**
     * Constructor
     * @param data: the double vale of this "node".
     */
    public GradScalar(double data) {

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
        // define a new implementation of the "toString" method.
        return "";
    }


}
