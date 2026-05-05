package com.pierluigicovone.microgradj.viz;

import com.pierluigicovone.microgradj.autograd.ComputationGraph;
import com.pierluigicovone.microgradj.autograd.DiffScalarNode;

import guru.nidi.graphviz.attribute.Font;
import guru.nidi.graphviz.attribute.Label;
import guru.nidi.graphviz.attribute.Rank;
import guru.nidi.graphviz.attribute.Shape;
import guru.nidi.graphviz.engine.Format;
import guru.nidi.graphviz.engine.Graphviz;
import guru.nidi.graphviz.model.Factory;
import guru.nidi.graphviz.model.MutableGraph;
import guru.nidi.graphviz.model.MutableNode;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Renders a computation graph (built from {@link DiffScalarNode}) into a visual diagram,
 * styled similarly to the diagrams shown in Andrej Karpathy's micrograd video.
 *
 * Each value node is rendered as a record-style box with two cells:
 * one for the data value and one for the gradient.
 * Each operation is rendered as a small circle with the operation symbol.
 *
 * The output is a PNG image, produced via the graphviz-java library.
 */
public final class GraphRenderer {

    private GraphRenderer() {}

    public static void renderToFile(DiffScalarNode root, String outputPath) throws IOException {
        MutableGraph g = buildGraph(root);
        Graphviz.fromGraph(g).render(Format.PNG).toFile(new File(outputPath));
    }

    private static MutableGraph buildGraph(DiffScalarNode root) {
        MutableGraph g = Factory.mutGraph("ComputationGraph")
                .setDirected(true)
                .graphAttrs().add(Rank.dir(Rank.RankDir.LEFT_TO_RIGHT))
                .nodeAttrs().add(Font.name("Helvetica"));

        List<DiffScalarNode> nodes = ComputationGraph.topologicalSort(root);
        Map<DiffScalarNode, MutableNode> valueNodes = new HashMap<>();

        // Pass 1: value boxes (record-style, two cells)
        for (DiffScalarNode n : nodes) {
            String id = "val_" + System.identityHashCode(n);
            MutableNode valueBox = Factory.mutNode(id)
                    .add(Shape.RECORD)
                    .add(Label.of(formatValueRecord(n)));
            valueNodes.put(n, valueBox);
            g.add(valueBox);
        }

        // Pass 2: for each non-leaf node, create the "operation circle"
        for (DiffScalarNode n : nodes) {
            if (n.getParents().isEmpty()) continue;

            String opId = "op_" + System.identityHashCode(n);
            MutableNode opCircle = Factory.mutNode(opId)
                    .add(Shape.CIRCLE)
                    .add(Label.of(n.getOperation()));
            g.add(opCircle);

            // Parents → op
            for (DiffScalarNode parent : n.getParents()) {
                valueNodes.get(parent).addLink(opCircle);
            }

            // If there's a constant, render it as a small node and link it too
            if (n.hasConstant()) {
                String constId = "const_" + System.identityHashCode(n);
                MutableNode constNode = Factory.mutNode(constId)
                        .add(Shape.ELLIPSE)
                        .add(Font.size(10))
                        .add(Label.of(String.format(Locale.US, "%.4f", n.getConstant())));
                g.add(constNode);
                constNode.addLink(opCircle);
            }

            // Op → this value
            opCircle.addLink(valueNodes.get(n));
        }

        return g;
    }

    /**
     * Builds the record-style label for a value box.
     * Format: "{ data = 2.0000 | grad = 0.0000 }".
     */
    private static String formatValueRecord(DiffScalarNode n) {
        String dataCell = String.format(Locale.US, "data %.4f", n.getData());
        String gradCell = String.format(Locale.US, "grad %.4f", n.getGrad());

        if (n.getVariableName().isEmpty()) {
            return String.format("{ %s | %s }", dataCell, gradCell);
        } else {
            return String.format("{ %s | %s | %s }", n.getVariableName(), dataCell, gradCell);
        }
    }
}