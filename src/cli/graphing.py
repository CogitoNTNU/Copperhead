"""Draw a neural-network schematic using Graphviz."""

import json
import subprocess
from pathlib import Path

from model_driven_nn.common import positive_int, validate_mapping
from model_driven_nn.parser import Parser
from model_driven_nn.types import Config, YamlMapping
from model_driven_nn.validator import Validator


def graph_yaml(
    source: str | Path,
    destination: str | Path,
    config: Config | None = None,
    *,
    max_neurons: int = 8,
) -> Path:
    positive_int(max_neurons, "max_neurons")

    architecture = Validator().validate_architecture(
        Parser().parse(source, config=config)
    )

    destination = Path(destination)
    output_format = destination.suffix.lower().removeprefix(".")

    if output_format not in {"dot", "svg", "png", "pdf"}:
        raise ValueError("Destination must end in .dot, .svg, .png, or .pdf")

    def quote(value: str) -> str:
        return json.dumps(value, ensure_ascii=False)

    lines = [
        "digraph Network {",
        "  rankdir=LR;",
        (
            f"  graph [label={quote(architecture.name)}, "
            'labelloc=t, fontname="Helvetica", '
            "nodesep=0.15, ranksep=0.85, splines=line];"
        ),
        (
            "  node [shape=circle, fixedsize=true, width=0.22, "
            'label="", style=filled, fillcolor="#DBEAFE", '
            'color="#3B82F6"];'
        ),
        ('  edge [color="#94A3B880", penwidth=0.6, arrowsize=0.4];'),
    ]
    column_count = 0

    # None represents an ellipsis among the visible neurons.
    type Neurons = list[str | None]

    def column(
        title: str,
        dim: int,
        color: str = "#DBEAFE",
        *,
        template: Neurons | None = None,
    ) -> Neurons:
        """Create a column, optionally preserving an incoming layout."""
        nonlocal column_count

        prefix = f"c{column_count}"
        column_count += 1

        if template is None:
            visible = min(dim, max_neurons)
            slots = [False] * visible
            if dim > max_neurons:
                slots.insert(visible // 2, True)
        else:
            slots = [item is None for item in template]

        label_id = f"{prefix}_label"
        label = quote(f"{title}\n{dim} features")

        lines.append(f"  subgraph column_{prefix} {{")
        lines.append("    rank=same;")
        lines.append(
            f"    {label_id} [shape=plaintext, fixedsize=false, "
            f'style="", fontname="Helvetica", label={label}];'
        )

        ordered = [label_id]
        neurons: Neurons = []

        for index, is_ellipsis in enumerate(slots):
            node_id = f"{prefix}_n{index}"
            ordered.append(node_id)

            if is_ellipsis:
                lines.append(
                    f"    {node_id} [shape=plaintext, fixedsize=false, "
                    f'style="", label={quote("⋮")}];'
                )
                neurons.append(None)
            else:
                lines.append(f'    {node_id} [fillcolor="{color}"];')
                neurons.append(node_id)

        lines.append(f"    {' -> '.join(ordered)} [style=invis, weight=100];")
        lines.append("  }")
        return neurons

    def connect(
        sources: Neurons,
        targets: Neurons,
        *,
        dense: bool,
    ) -> None:
        if dense:
            pairs = (
                (source, target)
                for source in sources
                for target in targets
                if source is not None and target is not None
            )
        else:
            pairs = (
                (source, target)
                for source, target in zip(sources, targets, strict=True)
                if source is not None and target is not None
            )

        for source, target in pairs:
            lines.append(f"  {source} -> {target};")

    def visit(
        spec: YamlMapping,
        incoming: Neurons,
        input_dim: int,
    ) -> tuple[Neurons, int]:
        component_type = spec["type"]

        if component_type == "linear":
            output_dim = positive_int(
                spec.get("output_dim"),
                "Linear output_dim",
            )
            outgoing = column("Linear", output_dim)
            connect(incoming, outgoing, dense=True)
            return outgoing, output_dim

        if component_type == "relu":
            outgoing = column(
                "ReLU",
                input_dim,
                "#DCFCE7",
                template=incoming,
            )
            connect(incoming, outgoing, dense=False)
            return outgoing, input_dim

        if component_type == "sequential":
            children = spec["layers"]
            if not isinstance(children, list):
                raise TypeError("Sequential layers must be a list")

            for child in children:
                incoming, input_dim = visit(
                    validate_mapping(child, "Layer"),
                    incoming,
                    input_dim,
                )

            return incoming, input_dim

        if component_type == "concat":
            paths = spec["paths"]
            if not isinstance(paths, list):
                raise TypeError("Concat paths must be a list")

            branches = [
                visit(
                    validate_mapping(path, "Path"),
                    incoming,
                    input_dim,
                )
                for path in paths
            ]

            # Concatenation preserves the branch features and their order.
            # No additional computational neurons are introduced.
            return (
                [neuron for branch_neurons, _ in branches for neuron in branch_neurons],
                sum(dim for _, dim in branches),
            )

        raise ValueError(f"Unsupported component type: {component_type!r}")

    inputs = column(
        "Input",
        architecture.input_dim,
        "#FEF3C7",
    )
    outputs, output_dim = visit(
        architecture.model,
        inputs,
        architecture.input_dim,
    )

    if architecture.output_dim is not None and output_dim != architecture.output_dim:
        raise ValueError(
            f"Architecture declares output_dim={architecture.output_dim}, "
            f"but graph inferred {output_dim} features"
        )

    # Highlight the final neurons and label them without adding a layer.
    lines.append(
        "  output_label [shape=plaintext, fixedsize=false, "
        f'style="", label={quote(f"Output: {output_dim} features")}, '
        'fontname="Helvetica"];'
    )

    for neuron in outputs:
        if neuron is not None:
            lines.append(f'  {neuron} [fillcolor="#FCE7F3", color="#DB2777"];')
            lines.append(f"  {neuron} -> output_label [style=invis];")

    lines.append("}")
    dot_source = "\n".join(lines)

    if output_format == "dot":
        destination.write_text(dot_source, encoding="utf-8")
    else:
        try:
            subprocess.run(
                ["dot", f"-T{output_format}", "-o", str(destination)],
                input=dot_source,
                encoding="utf-8",
                capture_output=True,
                check=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Graphviz's 'dot' executable is required to render graphs. "
                "Install Graphviz, or use a .dot destination."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Graphviz rendering failed: {exc.stderr.strip()}"
            ) from exc

    return destination
