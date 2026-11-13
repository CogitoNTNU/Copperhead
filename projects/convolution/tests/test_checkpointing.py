import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from convolution.main import (
    DQNAgent,
    main,
    make_checkpoint_callback,
    make_snake_env,
    parse_args,
    restore_checkpoint,
    save_checkpoint,
)
from coppercore.model_driven_nn.network import Network
from coppercore.optimizer import ADAM
from coppercore.training.dqn_harness import train


@pytest.fixture
def dummy_agent():
    env = make_snake_env()
    agent = DQNAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        obs_shape=env.obs_shape,
        seed=42,
        device="cpu",
        dtype=np.float32,
        batch_size=4,
        name="test-agent",
    )
    return agent


@pytest.fixture
def mock_wandb_run(tmp_path):
    run = MagicMock()
    run.id = "test-run-123"
    run.dir = str(tmp_path / "wandb_run")
    os.makedirs(run.dir, exist_ok=True)
    run.summary = {}
    return run


def test_callback_receives_correct_arguments(tmp_path, dummy_agent):
    mock_run = MagicMock()
    mock_run.id = "args-run"
    mock_run.dir = str(tmp_path / "wandb_dir")
    os.makedirs(mock_run.dir, exist_ok=True)
    mock_run.summary = {}

    received_calls = []

    def callback(**kwargs):
        received_calls.append(kwargs)

    with patch("coppercore.training.dqn_harness.wandb.init", return_value=mock_run):
        train(
            dummy_agent,
            make_env=make_snake_env,
            n_steps=2,
            checkpoint_callback=callback,
            eval_every=100,
        )

    # Step 1, Step 2, and Final step 2
    assert len(received_calls) == 3
    assert received_calls[0] == {
        "step": 1,
        "agent": dummy_agent,
        "run": mock_run,
        "is_final": False,
    }
    assert received_calls[1] == {
        "step": 2,
        "agent": dummy_agent,
        "run": mock_run,
        "is_final": False,
    }
    assert received_calls[2] == {
        "step": 2,
        "agent": dummy_agent,
        "run": mock_run,
        "is_final": True,
    }


def test_checkpoint_contains_network_and_optimizer(tmp_path, dummy_agent):
    step_dir = tmp_path / "step_100000"
    save_checkpoint(step_dir, dummy_agent)

    # Check network bundle
    network_dir = step_dir / "network"
    assert network_dir.is_dir()
    assert (network_dir / "manifest.json").exists()
    assert (network_dir / "model.yaml").exists()
    assert (network_dir / "weights.npz").exists()

    # Check optimizer state
    optimizer_file = step_dir / "optimizer.npz"
    assert optimizer_file.exists()

    # Verify reloading
    loaded_net = Network.load_model(network_dir)
    assert loaded_net is not None
    optimizer = ADAM(learning_rate=5e-4)
    optimizer.parameters = loaded_net.params()
    optimizer.load_state(optimizer_file, loaded_net)


def test_local_save_and_wandb_upload_intervals(tmp_path, dummy_agent, mock_wandb_run):
    checkpoints_dir = tmp_path / "checkpoints"
    cb = make_checkpoint_callback(
        checkpoint_interval=25_000,
        wandb_checkpoint_interval=100_000,
        checkpoints_dir=checkpoints_dir,
    )

    # Step 10,000: neither
    cb(step=10_000, agent=dummy_agent, run=mock_wandb_run, is_final=False)
    run_dir = checkpoints_dir / mock_wandb_run.id
    assert not (run_dir / "step_10000").exists()
    assert mock_wandb_run.log_artifact.call_count == 0

    # Step 25,000: local save only
    cb(step=25_000, agent=dummy_agent, run=mock_wandb_run, is_final=False)
    assert (run_dir / "step_25000").exists()
    assert mock_wandb_run.log_artifact.call_count == 0

    # Step 50,000: local save only
    cb(step=50_000, agent=dummy_agent, run=mock_wandb_run, is_final=False)
    assert (run_dir / "step_50000").exists()
    assert mock_wandb_run.log_artifact.call_count == 0

    # Step 75,000: local save, step_25000 pruned
    cb(step=75_000, agent=dummy_agent, run=mock_wandb_run, is_final=False)
    assert (run_dir / "step_75000").exists()
    assert (run_dir / "step_50000").exists()
    assert not (run_dir / "step_25000").exists()
    assert mock_wandb_run.log_artifact.call_count == 0

    # Step 100,000: local save + wandb upload, step_50000 pruned
    cb(step=100_000, agent=dummy_agent, run=mock_wandb_run, is_final=False)
    assert (run_dir / "step_100000").exists()
    assert (run_dir / "step_75000").exists()
    assert not (run_dir / "step_50000").exists()
    assert mock_wandb_run.log_artifact.call_count == 1

    artifact = mock_wandb_run.log_artifact.call_args[0][0]
    assert artifact.name == "snake-dqn-checkpoint"
    assert artifact.type == "model"
    assert artifact.metadata["step"] == 100_000


def test_previous_runs_remain_untouched(tmp_path, dummy_agent, mock_wandb_run):
    checkpoints_dir = tmp_path / "checkpoints"
    prev_run_dir = checkpoints_dir / "previous_run_abc"
    prev_step_dir = prev_run_dir / "step_25000"
    prev_step_dir.mkdir(parents=True)
    (prev_step_dir / "dummy.txt").write_text("keep me")

    cb = make_checkpoint_callback(
        checkpoint_interval=25_000,
        wandb_checkpoint_interval=100_000,
        checkpoints_dir=checkpoints_dir,
    )

    for step in [25_000, 50_000, 75_000, 100_000]:
        cb(step=step, agent=dummy_agent, run=mock_wandb_run, is_final=False)

    # Verify previous run was untouched
    assert prev_step_dir.exists()
    assert (prev_step_dir / "dummy.txt").read_text() == "keep me"


def test_no_duplicate_saves_or_uploads_on_final(tmp_path, dummy_agent, mock_wandb_run):
    checkpoints_dir = tmp_path / "checkpoints"
    cb = make_checkpoint_callback(
        checkpoint_interval=25_000,
        wandb_checkpoint_interval=100_000,
        checkpoints_dir=checkpoints_dir,
    )

    # Step 100,000 during loop
    cb(step=100_000, agent=dummy_agent, run=mock_wandb_run, is_final=False)
    assert mock_wandb_run.log_artifact.call_count == 1

    # Final callback at step 100,000
    cb(step=100_000, agent=dummy_agent, run=mock_wandb_run, is_final=True)
    assert mock_wandb_run.log_artifact.call_count == 1


def test_interrupted_run_saves_and_uploads_final_checkpoint(tmp_path, dummy_agent):
    checkpoints_dir = tmp_path / "checkpoints"
    mock_run = MagicMock()
    mock_run.id = "interrupted-run"
    mock_run.dir = str(tmp_path / "wandb_dir")
    os.makedirs(mock_run.dir, exist_ok=True)
    mock_run.summary = {}

    cb = make_checkpoint_callback(
        checkpoint_interval=25_000,
        wandb_checkpoint_interval=100_000,
        checkpoints_dir=checkpoints_dir,
    )

    call_count = 0

    def mock_observe(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 5:
            raise KeyboardInterrupt()

    dummy_agent.observe = mock_observe

    with patch("coppercore.training.dqn_harness.wandb.init", return_value=mock_run):
        train(
            dummy_agent,
            make_env=make_snake_env,
            n_steps=10,
            checkpoint_callback=cb,
            eval_every=100,
        )

    run_dir = checkpoints_dir / mock_run.id
    # Last completed step before interruption was 4
    assert (run_dir / "step_4").exists()
    assert (run_dir / "step_4" / "network").exists()
    assert (run_dir / "step_4" / "optimizer.npz").exists()
    assert mock_run.log_artifact.call_count == 1
    assert mock_run.finish.call_count == 1


def test_normal_completion_with_train_loop(tmp_path, dummy_agent):
    checkpoints_dir = tmp_path / "checkpoints"
    mock_run = MagicMock()
    mock_run.id = "normal-run"
    mock_run.dir = str(tmp_path / "wandb_dir")
    os.makedirs(mock_run.dir, exist_ok=True)
    mock_run.summary = {}

    cb = make_checkpoint_callback(
        checkpoint_interval=25_000,
        wandb_checkpoint_interval=100_000,
        checkpoints_dir=checkpoints_dir,
    )

    with patch("coppercore.training.dqn_harness.wandb.init", return_value=mock_run):
        train(
            dummy_agent,
            make_env=make_snake_env,
            n_steps=5,
            checkpoint_callback=cb,
            eval_every=100,
        )

    run_dir = checkpoints_dir / mock_run.id
    assert (run_dir / "step_5").exists()
    assert (run_dir / "step_5" / "network").exists()
    assert (run_dir / "step_5" / "optimizer.npz").exists()
    checkpoint_artifacts = [
        call[0][0]
        for call in mock_run.log_artifact.call_args_list
        if getattr(call[0][0], "name", None) == "snake-dqn-checkpoint"
    ]
    assert len(checkpoint_artifacts) == 1
    assert mock_run.finish.call_count == 1


def test_existing_training_works_without_callback(tmp_path, dummy_agent):
    mock_run = MagicMock()
    mock_run.id = "no-cb-run"
    mock_run.dir = str(tmp_path / "wandb_dir")
    os.makedirs(mock_run.dir, exist_ok=True)
    mock_run.summary = {}

    with patch("coppercore.training.dqn_harness.wandb.init", return_value=mock_run):
        train(
            dummy_agent,
            make_env=make_snake_env,
            n_steps=3,
            eval_every=100,
        )

    assert mock_run.finish.call_count == 1


def test_wandb_cleanup_on_checkpoint_failure(tmp_path, dummy_agent):
    mock_run = MagicMock()
    mock_run.id = "fail-run"
    mock_run.dir = str(tmp_path / "wandb_dir")
    os.makedirs(mock_run.dir, exist_ok=True)
    mock_run.summary = {}

    def failing_callback(step, agent, run, is_final=False):
        raise RuntimeError("Checkpoint failure simulated")

    with patch("coppercore.training.dqn_harness.wandb.init", return_value=mock_run):
        with pytest.raises(RuntimeError, match="Checkpoint failure simulated"):
            train(
                dummy_agent,
                make_env=make_snake_env,
                n_steps=3,
                checkpoint_callback=failing_callback,
                eval_every=100,
            )

    # Ensure run.finish() was still called despite error
    assert mock_run.finish.call_count == 1


def test_disabled_checkpoint_intervals(tmp_path, dummy_agent, mock_wandb_run):
    checkpoints_dir = tmp_path / "checkpoints"
    cb = make_checkpoint_callback(
        checkpoint_interval=0,
        wandb_checkpoint_interval=0,
        checkpoints_dir=checkpoints_dir,
    )

    # Periodic step with 0 interval does nothing
    cb(step=25_000, agent=dummy_agent, run=mock_wandb_run, is_final=False)
    run_dir = checkpoints_dir / mock_wandb_run.id
    assert not (run_dir / "step_25000").exists()
    assert mock_wandb_run.log_artifact.call_count == 0

    # Final step still saves and uploads
    cb(step=25_000, agent=dummy_agent, run=mock_wandb_run, is_final=True)
    assert (run_dir / "step_25000").exists()
    assert mock_wandb_run.log_artifact.call_count == 1


def test_sigterm_handling(tmp_path, dummy_agent):
    import signal

    checkpoints_dir = tmp_path / "checkpoints"
    mock_run = MagicMock()
    mock_run.id = "sigterm-run"
    mock_run.dir = str(tmp_path / "wandb_dir")
    os.makedirs(mock_run.dir, exist_ok=True)
    mock_run.summary = {}

    cb = make_checkpoint_callback(
        checkpoint_interval=25_000,
        wandb_checkpoint_interval=100_000,
        checkpoints_dir=checkpoints_dir,
    )

    step_counter = 0

    def mock_observe(*args, **kwargs):
        nonlocal step_counter
        step_counter += 1
        if step_counter == 3:
            # Send SIGTERM to self
            os.kill(os.getpid(), signal.SIGTERM)

    dummy_agent.observe = mock_observe

    with patch("coppercore.training.dqn_harness.wandb.init", return_value=mock_run):
        train(
            dummy_agent,
            make_env=make_snake_env,
            n_steps=10,
            checkpoint_callback=cb,
            eval_every=100,
        )

    run_dir = checkpoints_dir / mock_run.id
    # Step 3 completes and loop stops gracefully on SIGTERM
    assert (run_dir / "step_3").exists()
    assert mock_run.finish.call_count == 1


def test_disabled_wandb_run_saves_local_checkpoints(tmp_path, dummy_agent):
    checkpoints_dir = tmp_path / "checkpoints"
    cb = make_checkpoint_callback(
        checkpoint_interval=25_000,
        wandb_checkpoint_interval=100_000,
        checkpoints_dir=checkpoints_dir,
    )

    mock_run = MagicMock()
    mock_run.id = "disabled-run-1"
    mock_run.disabled = True

    cb(step=25_000, agent=dummy_agent, run=mock_run, is_final=False)
    run_dir = checkpoints_dir / mock_run.id
    assert (run_dir / "step_25000").exists()
    assert mock_run.log_artifact.call_count == 0

    cb(step=100_000, agent=dummy_agent, run=mock_run, is_final=False)
    assert (run_dir / "step_100000").exists()
    assert mock_run.log_artifact.call_count == 0

    # Also test with run.settings.mode = "disabled"
    mock_run2 = MagicMock()
    mock_run2.id = "disabled-run-2"
    mock_run2.disabled = False
    mock_run2.settings.mode = "disabled"

    cb2 = make_checkpoint_callback(
        checkpoint_interval=25_000,
        wandb_checkpoint_interval=100_000,
        checkpoints_dir=checkpoints_dir,
    )
    cb2(step=100_000, agent=dummy_agent, run=mock_run2, is_final=True)
    run_dir2 = checkpoints_dir / mock_run2.id
    assert (run_dir2 / "step_100000").exists()
    assert mock_run2.log_artifact.call_count == 0


def test_checkpoint_cleanup_waits_for_wandb_artifact(
    tmp_path, dummy_agent, mock_wandb_run
):
    checkpoints_dir = tmp_path / "checkpoints"
    cb = make_checkpoint_callback(
        checkpoint_interval=25_000,
        wandb_checkpoint_interval=100_000,
        checkpoints_dir=checkpoints_dir,
    )

    events = []
    mock_artifact = MagicMock()

    def mock_wait():
        events.append("artifact_wait")

    mock_artifact.wait = mock_wait
    mock_wandb_run.log_artifact.return_value = mock_artifact

    # Fill up to 75k
    cb(step=25_000, agent=dummy_agent, run=mock_wandb_run, is_final=False)
    cb(step=50_000, agent=dummy_agent, run=mock_wandb_run, is_final=False)
    cb(step=75_000, agent=dummy_agent, run=mock_wandb_run, is_final=False)

    # At 100k, artifact is logged and wait() is called before retention deletes 50k
    run_dir = checkpoints_dir / mock_wandb_run.id
    assert (run_dir / "step_50000").exists()

    cb(step=100_000, agent=dummy_agent, run=mock_wandb_run, is_final=False)
    assert "artifact_wait" in events
    assert not (run_dir / "step_50000").exists()


def test_parse_args_custom_intervals():
    import sys

    from convolution.main import parse_args

    test_args = [
        "prog",
        "--checkpoint-interval",
        "50000",
        "--wandb-checkpoint-interval",
        "200000",
    ]
    with patch.object(sys, "argv", test_args):
        args = parse_args()
        assert args.checkpoint_interval == 50000
        assert args.wandb_checkpoint_interval == 200000


def test_parse_args_resume_options():
    # Local resume
    args = parse_args(["--resume-local", "checkpoints/run1/step_25000"])
    assert args.resume_local == "checkpoints/run1/step_25000"
    assert args.resume_wandb is None
    assert args.wandb_mode is None

    # W&B resume default mode
    args = parse_args(["--resume-wandb", "entity/proj/art:v1"])
    assert args.resume_wandb == "entity/proj/art:v1"
    assert args.resume_local is None
    assert args.wandb_mode is None

    # W&B resume fork mode
    args = parse_args(["--resume-wandb", "entity/proj/art:v1", "--wandb-mode", "fork"])
    assert args.wandb_mode == "fork"

    # W&B resume continue mode
    args = parse_args(
        ["--resume-wandb", "entity/proj/art:v1", "--wandb-mode", "continue"]
    )
    assert args.wandb_mode == "continue"

    # Mutually exclusive sources
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--resume-local",
                "checkpoints/run1/step_25000",
                "--resume-wandb",
                "entity/proj/art:v1",
            ]
        )

    # wandb-mode without resume-wandb
    with pytest.raises(SystemExit):
        parse_args(["--wandb-mode", "fork"])

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--resume-local",
                "checkpoints/run1/step_25000",
                "--wandb-mode",
                "fork",
            ]
        )


def test_restore_checkpoint_network_and_optimizer(tmp_path, dummy_agent):
    # Train dummy agent with one step of mock gradients to set optimizer moments
    for p in dummy_agent.Q_policy.params():
        p.grad = np.ones_like(p.data) * 0.05
    dummy_agent.optimizer.step()

    step_dir = tmp_path / "step_50000"
    save_checkpoint(step_dir, dummy_agent)

    # Create new agent with modified weights
    env = make_snake_env()
    new_agent = DQNAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        obs_shape=env.obs_shape,
        seed=123,
        device="cpu",
        dtype=np.float32,
    )
    for p in new_agent.Q_policy.params():
        p.data += 5.0
    new_agent.network_update()

    restore_checkpoint(new_agent, step_dir)

    # 1. Verify policy network weights match
    for p_orig, p_restored in zip(
        dummy_agent.Q_policy.params(), new_agent.Q_policy.params()
    ):
        assert np.allclose(p_orig.data, p_restored.data)

    # 2. Verify target network initialized from restored policy network
    for p_target, p_policy in zip(
        new_agent.Q_target.params(), new_agent.Q_policy.params()
    ):
        assert np.allclose(p_target.data, p_policy.data)

    # 3. Verify optimizer references the restored policy network parameters
    assert [id(p) for p in new_agent.optimizer.parameters] == [
        id(p) for p in new_agent.Q_policy.params()
    ]

    # 4. Verify optimizer state matches
    orig_state = dummy_agent.optimizer.state_dict(dummy_agent.Q_policy)
    restored_state = new_agent.optimizer.state_dict(new_agent.Q_policy)
    assert orig_state["time_step"] == restored_state["time_step"]
    assert orig_state["state"].keys() == restored_state["state"].keys()
    for k in orig_state["state"]:
        assert np.allclose(
            orig_state["state"][k]["moment_1"], restored_state["state"][k]["moment_1"]
        )
        assert np.allclose(
            orig_state["state"][k]["moment_2"], restored_state["state"][k]["moment_2"]
        )

    # 5. Verify device and dtype preserved
    assert new_agent.Q_policy.params()[0].data.dtype == np.float32
    assert new_agent.device == "cpu"


def test_resume_local_checkpoint(tmp_path, dummy_agent):
    step_dir = tmp_path / "step_25000"
    save_checkpoint(step_dir, dummy_agent)

    with patch("convolution.main.train") as mock_train:
        main(["--resume-local", str(step_dir), "--n-steps", "30000"])

        assert mock_train.call_count == 1
        _, kwargs = mock_train.call_args
        assert kwargs["start_step"] == 25000
        assert kwargs["n_steps"] == 30000
        assert kwargs["run_id"] is None
        assert kwargs["resume"] is None
        assert kwargs["fork_from"] is None

        # Verify agent steps counter was updated
        agent = mock_train.call_args[0][0]
        assert agent.steps == 25000


def test_resume_wandb_continue_mode(tmp_path, dummy_agent):
    ckpt_dir = tmp_path / "wandb_download"
    save_checkpoint(ckpt_dir, dummy_agent)

    mock_artifact = MagicMock()
    mock_artifact.download.return_value = str(ckpt_dir)
    mock_artifact.metadata = {"step": 40000, "_step": 150}
    mock_artifact.logged_by.return_value = MagicMock(id="original-run-abc")

    mock_api = MagicMock()
    mock_api.artifact.return_value = mock_artifact

    with (
        patch("convolution.main.wandb.Api", return_value=mock_api),
        patch("convolution.main.train") as mock_train,
    ):
        main(
            [
                "--resume-wandb",
                "entity/proj/artifact:v1",
                "--wandb-mode",
                "continue",
                "--n-steps",
                "50000",
            ]
        )

        mock_api.artifact.assert_called_once_with("entity/proj/artifact:v1")
        mock_artifact.download.assert_called_once()

        assert mock_train.call_count == 1
        _, kwargs = mock_train.call_args
        assert kwargs["start_step"] == 40000
        assert kwargs["n_steps"] == 50000
        assert kwargs["run_id"] == "original-run-abc"
        assert kwargs["resume"] == "must"
        assert kwargs["fork_from"] is None
        assert kwargs["use_artifact"] == mock_artifact

        agent = mock_train.call_args[0][0]
        assert agent.steps == 40000


def test_resume_wandb_fork_mode(tmp_path, dummy_agent):
    ckpt_dir = tmp_path / "wandb_download"
    save_checkpoint(ckpt_dir, dummy_agent)

    mock_artifact = MagicMock()
    mock_artifact.download.return_value = str(ckpt_dir)
    mock_artifact.metadata = {"step": 40000, "_step": 150}
    mock_artifact.logged_by.return_value = MagicMock(id="original-run-abc")

    mock_api = MagicMock()
    mock_api.artifact.return_value = mock_artifact

    with (
        patch("convolution.main.wandb.Api", return_value=mock_api),
        patch("convolution.main.train") as mock_train,
    ):
        main(
            [
                "--resume-wandb",
                "entity/proj/artifact:v1",
                "--wandb-mode",
                "fork",
                "--n-steps",
                "50000",
            ]
        )

        assert mock_train.call_count == 1
        _, kwargs = mock_train.call_args
        assert kwargs["start_step"] == 40000
        assert kwargs["n_steps"] == 50000
        assert kwargs["run_id"] is None
        assert kwargs["resume"] is None
        assert kwargs["fork_from"] == "original-run-abc?_step=150"
        assert kwargs["use_artifact"] == mock_artifact

        agent = mock_train.call_args[0][0]
        assert agent.steps == 40000


def test_training_step_progression_on_resume(dummy_agent):
    mock_run = MagicMock()
    mock_run.id = "progress-run"
    mock_run.dir = "/tmp"
    mock_run.summary = {}

    logged_steps = []
    received_callback_steps = []

    def mock_log(data, step=None):
        logged_steps.append(step)

    mock_run.log = mock_log

    def mock_cb(step, agent, run, is_final=False):
        received_callback_steps.append((step, is_final))

    with patch("coppercore.training.dqn_harness.wandb.init", return_value=mock_run):
        train(
            dummy_agent,
            make_env=make_snake_env,
            start_step=100,
            n_steps=103,
            checkpoint_callback=mock_cb,
            eval_every=1000,
            log_every=1,
        )

    # Check that loop only executed steps 101, 102, 103
    callback_steps = [s for s, is_final in received_callback_steps if not is_final]
    assert callback_steps == [101, 102, 103]
    # Final callback should be at step 103
    final_callbacks = [s for s, is_final in received_callback_steps if is_final]
    assert final_callbacks == [103]


def test_restore_checkpoint_does_not_perform_gradient_descent(tmp_path, dummy_agent):
    step_dir = tmp_path / "step_10000"
    save_checkpoint(step_dir, dummy_agent)

    env = make_snake_env()
    new_agent = DQNAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        obs_shape=env.obs_shape,
        seed=999,
        device="cpu",
        dtype=np.float32,
    )

    with patch.object(new_agent.optimizer, "step") as mock_step:
        restore_checkpoint(new_agent, step_dir)
        # Verify optimizer.step() was not called
        assert mock_step.call_count == 0

    # Verify target parameters match restored policy parameters exactly
    for p_target, p_policy in zip(
        new_agent.Q_target.params(), new_agent.Q_policy.params()
    ):
        assert np.allclose(p_target.data, p_policy.data)


def test_resumed_training_with_empty_replay_buffer(tmp_path):
    env = make_snake_env()
    agent = DQNAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        obs_shape=env.obs_shape,
        seed=42,
        device="cpu",
        dtype=np.float32,
        batch_size=4,
        n_train_start=20_000,
        train_freq=4,
    )
    # Simulate restored agent at step 100,000 with empty replay buffer
    agent.steps = 100_000
    assert len(agent.replay_buffer) == 0

    mock_run = MagicMock()
    mock_run.id = "resume-buf-run"
    mock_run.dir = str(tmp_path / "wandb_dir")
    os.makedirs(mock_run.dir, exist_ok=True)
    mock_run.summary = {}

    with patch("coppercore.training.dqn_harness.wandb.init", return_value=mock_run):
        train(
            agent,
            make_env=make_snake_env,
            start_step=100_000,
            n_steps=100_008,
            eval_every=1000,
            log_every=1,
        )

    # Agent should have executed 8 steps
    assert agent.steps == 100_008
    assert len(agent.replay_buffer) == 8
    # Epsilon should follow the 100_008 global step schedule, not reset to step 8
    expected_eps = agent.max_epsilon + min(1.0, 100_008 / agent.eps_decay_steps) * (
        agent.min_epsilon - agent.max_epsilon
    )
    assert np.isclose(agent.epsilon, expected_eps)


def test_fresh_training_warmup_behavior():
    env = make_snake_env()
    agent = DQNAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        obs_shape=env.obs_shape,
        seed=42,
        device="cpu",
        dtype=np.float32,
        batch_size=4,
        n_train_start=20_000,
        train_freq=4,
    )
    obs, _ = env.reset()
    action = 0
    next_obs, reward, terminated, _, _ = env.step(action)

    # In fresh training before reaching n_train_start, observe returns None
    for _ in range(8):
        metrics = agent.observe(obs, action, reward, next_obs, terminated)
        assert metrics is None


def test_wandb_use_artifact_called_in_train():
    env = make_snake_env()
    agent = DQNAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        obs_shape=env.obs_shape,
        seed=42,
        device="cpu",
        dtype=np.float32,
        batch_size=4,
    )

    mock_run = MagicMock()
    mock_run.id = "artifact-test-run"
    mock_run.dir = "/tmp"
    mock_run.summary = {}

    mock_artifact = MagicMock()

    with patch("coppercore.training.dqn_harness.wandb.init", return_value=mock_run):
        train(
            agent,
            make_env=make_snake_env,
            start_step=0,
            n_steps=2,
            use_artifact=mock_artifact,
            eval_every=1000,
        )

    mock_run.use_artifact.assert_called_once_with(mock_artifact)
