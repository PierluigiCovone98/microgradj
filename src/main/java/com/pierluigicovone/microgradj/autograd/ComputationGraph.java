package com.pierluigicovone.microgradj.autograd;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Utility class for traversing computation graphs built from {@link DiffScalarNode} instances.
 * Used by the autograd engine and the visualization layer.
 */
public final class ComputationGraph {

    /**
     * Private constructor to avoid initializations from the outside.
     */
    private ComputationGraph() {
        // Empty
    }

    /**
     * Given a {@link DiffScalarNode} node, retrieve the graph in its topological order.
     * The parameter "root" will correspond to the "output".
     *
     * This is a recursive method; it should not be a problem for the dimensions
     * we are going to deal.
     */
    public static List<DiffScalarNode> topologicalSort(DiffScalarNode root) {
        // Prepare the result and the set of visited.
        List<DiffScalarNode> result = new ArrayList<>();
        Set<DiffScalarNode> visited = new HashSet<>();

        buildTopo(root, visited, result);

        return result;
    }

    // Depth first walk
    private static void buildTopo(DiffScalarNode node, Set<DiffScalarNode> visited, List<DiffScalarNode> result) {

        // Check if we've already visited the node.
        if ( visited.contains(node) ) return;

        // We mark as visited.
        visited.add(node);

        // If no parents are present, it is a leaf;
        // drop the cycle and add it directly.
        for (DiffScalarNode parent : node.getParents()) {
            buildTopo(parent, visited, result);
        }

        // When returning, we add the node to the result.
        result.add(node);
    }

}
