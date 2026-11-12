from coppercore.model_driven_nn.components import Concat, Linear, ReLU, Sequential
from coppercore.model_driven_nn.construction.loader import Loader
from coppercore.model_driven_nn.network import Network


def test_simple_linear_architecture_roundtrip(tmp_path):
    model = Sequential([Linear(10, 5)])
    net = Network("simple_linear", model, 10, 5)

    yaml_path = tmp_path / "simple_linear.yaml"
    net.save_architecture(yaml_path)

    restored = Loader().load(yaml_path)

    assert restored.name == net.name
    assert restored.input_dim == net.input_dim
    assert restored.output_dim == net.output_dim
    assert restored.number() == net.number()
    assert len(restored.params()) == len(net.params())
    for p_orig, p_rest in zip(net.params(), restored.params()):
        assert p_orig.data.shape == p_rest.data.shape


def test_multiple_linear_relu_architecture_roundtrip(tmp_path):
    model = Sequential(
        [
            Linear(20, 15),
            ReLU(),
            Linear(15, 10),
            ReLU(),
            Linear(10, 2),
        ]
    )
    net = Network("mlp", model, 20, 2)

    yaml_path = tmp_path / "mlp.yaml"
    net.save_architecture(yaml_path)

    restored = Loader().load(yaml_path)

    assert restored.name == net.name
    assert restored.input_dim == 20
    assert restored.output_dim == 2
    assert restored.number() == net.number()
    assert len(restored.params()) == len(net.params())


def test_nested_sequential_architecture_roundtrip(tmp_path):
    model = Sequential(
        [
            Linear(32, 16),
            ReLU(),
            Sequential(
                [
                    Linear(16, 8),
                    ReLU(),
                    Linear(8, 4),
                ]
            ),
            ReLU(),
            Linear(4, 2),
        ]
    )
    net = Network("nested_seq", model, 32, 2)

    yaml_path = tmp_path / "nested_seq.yaml"
    net.save_architecture(yaml_path)

    restored = Loader().load(yaml_path)

    assert restored.name == net.name
    assert restored.input_dim == 32
    assert restored.output_dim == 2
    assert restored.number() == net.number()
    assert len(restored.params()) == len(net.params())


def test_concat_branches_architecture_roundtrip(tmp_path):
    branch1 = Sequential([Linear(10, 4), ReLU()])
    branch2 = Sequential([Linear(10, 6), ReLU()])
    concat = Concat([branch1, branch2])
    model = Sequential([concat, Linear(10, 3)])
    net = Network("concat_net", model, 10, 3)

    yaml_path = tmp_path / "concat_net.yaml"
    net.save_architecture(yaml_path)

    restored = Loader().load(yaml_path)

    assert restored.name == net.name
    assert restored.input_dim == 10
    assert restored.output_dim == 3
    assert restored.number() == net.number()
    assert len(restored.params()) == len(net.params())


def test_nested_sequential_inside_concat_architecture_roundtrip(tmp_path):
    branch1 = Sequential(
        [
            Linear(12, 8),
            ReLU(),
            Sequential([Linear(8, 4), ReLU()]),
        ]
    )
    branch2 = Sequential([Linear(12, 4), ReLU()])
    concat = Concat([branch1, branch2])
    model = Sequential([concat, Linear(8, 2)])
    net = Network("nested_concat_net", model, 12, 2)

    yaml_path = tmp_path / "nested_concat_net.yaml"
    net.save_architecture(yaml_path)

    restored = Loader().load(yaml_path)

    assert restored.name == net.name
    assert restored.input_dim == 12
    assert restored.output_dim == 2
    assert restored.number() == net.number()


def test_network_name_does_not_duplicate_suffix_on_roundtrip(tmp_path):
    model = Sequential([Linear(100, 50), ReLU(), Linear(50, 10)])
    net = Network("classifier", model, 100, 10)

    initial_name = net.name
    assert "_" in initial_name

    yaml_path = tmp_path / "classifier.yaml"
    net.save_architecture(yaml_path)

    restored = Loader().load(yaml_path)
    assert restored.name == initial_name

    yaml_path_2 = tmp_path / "classifier_2.yaml"
    restored.save_architecture(yaml_path_2)

    restored_again = Loader().load(yaml_path_2)
    assert restored_again.name == initial_name
