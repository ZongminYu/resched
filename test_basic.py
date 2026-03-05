"""Quick test script for ReSched components."""
import torch
from resched.env.instance_generator import InstanceGenerator
from resched.env.fjsp_env import FJSPEnv
from resched.model.network import ReSchedPolicy

def test_all():
    print("=" * 50)
    print("Testing ReSched Components")
    print("=" * 50)

    # Test instance generation
    gen = InstanceGenerator(
        num_jobs=3, num_machines=2, ops_low=2, ops_high=3,
        duration_low=1, duration_high=10
    )
    instance = gen.generate(batch_size=2)
    print("\n[1] Instance generated successfully")
    print(f"    duration_matrix: {instance['duration_matrix'].shape}")
    print(f"    total_ops: {instance['total_ops']}")

    # Test environment
    env = FJSPEnv()
    state = env.reset(instance)
    print("\n[2] Environment reset successfully")
    print(f"    op_available_time: {state['op_available_time'].shape}")
    print(f"    action_mask sum: {state['action_mask'].sum(dim=(1,2))}")
    print(f"    eligible ops: {state['eligible'].sum(dim=1)}")

    # Test policy network (small for testing)
    policy = ReSchedPolicy(
        hidden_dim=32, ffn_dim=64, num_heads=4,
        num_layers=1, mlp_hidden_dim=16, mlp_num_layers=2
    )
    num_params = sum(p.numel() for p in policy.parameters())
    print(f"\n[3] Policy created: {num_params:,} parameters")

    # Test forward pass
    log_probs, mask = policy(state)
    print(f"    log_probs: {log_probs.shape}")

    # Test action selection
    actions, selected_log_probs = policy.select_action(state, greedy=False)
    print(f"    actions: {actions.tolist()}")

    # Test step
    next_state, reward, done = env.step(actions)
    print(f"\n[4] Step completed")
    print(f"    reward: {reward.tolist()}")
    print(f"    done: {done.tolist()}")

    # Full rollout
    state = env.reset(instance)
    steps = 0
    while not env.done.all():
        actions, _ = policy.select_action(env._get_state(), greedy=True)
        _, _, _ = env.step(actions)
        steps += 1
    makespan = env.get_makespan()
    print(f"\n[5] Full rollout: {steps} steps")
    print(f"    makespan: {makespan.tolist()}")

    # Test gradient flow
    state = env.reset(instance)
    total_log_prob = torch.tensor(0.0)
    while not env.done.all():
        actions, log_p = policy.select_action(env._get_state(), greedy=False)
        total_log_prob = total_log_prob + log_p.sum()
        _, _, _ = env.step(actions)
    loss = -total_log_prob
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in policy.parameters() if p.grad is not None)
    print(f"\n[6] Gradient flow OK (grad_norm={grad_norm:.4f})")

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED!")
    print("=" * 50)


if __name__ == "__main__":
    test_all()
