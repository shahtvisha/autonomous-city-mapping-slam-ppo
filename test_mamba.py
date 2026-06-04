"""
test_mamba.py — Quick Test for Mamba Implementation
----------------------------------------------------
Verify that both Mamba architectures work correctly.

Run:
    python test_mamba.py
"""

import torch
import numpy as np
from agent.mamba_memory import MambaSLAMPolicy, MambaMemorySLAMPolicy
from agent.mamba_trainer import MambaPPOTrainer


def test_architectures():
    """Test both Mamba architectures"""
    print("\n" + "="*70)
    print("Testing Mamba-based SLAM Architectures")
    print("="*70 + "\n")
    
    obs_dim = 500  # Typical SLAM observation
    n_actions = 8
    batch_size = 4
    
    # Test 1: Pure Mamba
    print("1. Testing Pure Mamba Policy")
    print("-" * 70)
    try:
        policy = MambaSLAMPolicy(obs_dim, n_actions, d_model=256, n_layers=4)
        n_params = sum(p.numel() for p in policy.parameters())
        print(f"✓ Model created: {n_params:,} parameters")
        
        # Forward pass
        obs = torch.randn(batch_size, obs_dim)
        logits, value = policy(obs, update_buffer=False)
        print(f"✓ Forward pass: logits {logits.shape}, value {value.shape}")
        
        # Sequential processing
        policy.reset_sequence()
        for i in range(10):
            obs = torch.randn(1, obs_dim)
            logits, value = policy(obs, update_buffer=True)
        print(f"✓ Sequential processing: 10 steps completed")
        print(f"✓ Sequence length: {policy.seq_len.item()}")
        
        print("✓ Pure Mamba: ALL TESTS PASSED\n")
    except Exception as e:
        print(f"✗ Pure Mamba FAILED: {e}\n")
        return False
    
    # Test 2: Hybrid Mamba + Memory
    print("2. Testing Hybrid Mamba + Memory Bank Policy")
    print("-" * 70)
    try:
        policy = MambaMemorySLAMPolicy(
            obs_dim, n_actions, d_model=256, n_layers=4, memory_size=1000
        )
        n_params = sum(p.numel() for p in policy.parameters())
        print(f"✓ Model created: {n_params:,} parameters")
        
        # Forward pass
        obs = torch.randn(batch_size, obs_dim)
        outputs = policy(obs, update_buffer=False, update_memory=False)
        print(f"✓ Forward pass: logits {outputs['logits'].shape}, "
              f"value {outputs['value'].shape}")
        print(f"✓ Loop closure prob: {outputs['loop_closure_prob'].shape}")
        
        # Sequential processing with memory updates
        policy.reset_sequence()
        for i in range(20):
            obs = torch.randn(1, obs_dim)
            outputs = policy(obs, update_buffer=True, update_memory=True)
        print(f"✓ Sequential processing: 20 steps completed")
        print(f"✓ Sequence length: {policy.seq_len.item()}")
        
        # Check memory bank
        mem_util = (policy.memory_bank.memory_usage > 0).sum().item()
        print(f"✓ Memory bank: {mem_util} landmarks stored")
        
        # Test loop closure detection
        obs = torch.randn(1, obs_dim)
        candidates = policy.get_loop_closures(obs, threshold=0.85)
        print(f"✓ Loop closure detection: {len(candidates)} candidates found")
        
        print("✓ Hybrid Mamba + Memory: ALL TESTS PASSED\n")
    except Exception as e:
        print(f"✗ Hybrid Mamba + Memory FAILED: {e}\n")
        return False
    
    # Test 3: PPO Trainer
    print("3. Testing PPO Trainer Integration")
    print("-" * 70)
    try:
        # Pure trainer
        trainer_pure = MambaPPOTrainer(
            obs_dim=obs_dim,
            n_actions=n_actions,
            policy_type='pure',
            d_model=128,
            n_layers=2
        )
        print(f"✓ Pure trainer created on {trainer_pure.device}")
        
        # Test act
        obs = np.random.randn(obs_dim)
        action, log_prob, value = trainer_pure.act(obs)
        print(f"✓ Pure act: action={action}, log_prob={log_prob:.3f}, value={value:.3f}")
        
        # Hybrid trainer
        trainer_hybrid = MambaPPOTrainer(
            obs_dim=obs_dim,
            n_actions=n_actions,
            policy_type='hybrid',
            d_model=128,
            n_layers=2,
            memory_size=500
        )
        print(f"✓ Hybrid trainer created on {trainer_hybrid.device}")
        
        # Test act
        action, log_prob, value, loop_closure = trainer_hybrid.act(obs)
        print(f"✓ Hybrid act: action={action}, log_prob={log_prob:.3f}, "
              f"value={value:.3f}, loop={loop_closure:.3f}")
        
        # Test memory stats
        stats = trainer_hybrid.get_memory_stats()
        print(f"✓ Memory stats: utilization={stats['memory_utilization']:.1%}")
        
        print("✓ PPO Trainer: ALL TESTS PASSED\n")
    except Exception as e:
        print(f"✗ PPO Trainer FAILED: {e}\n")
        return False
    
    return True


def test_training_loop():
    """Test a mini training loop"""
    print("4. Testing Mini Training Loop")
    print("-" * 70)
    
    try:
        from envs.city_env import CityExplorerEnv
        
        # Create small environment
        env = CityExplorerEnv(width=20, height=20, max_steps=50, seed=42)
        obs, _ = env.reset()
        
        # Create trainer
        trainer = MambaPPOTrainer(
            obs_dim=env.observation_space.shape[0],
            n_actions=env.action_space.n,
            policy_type='hybrid',
            d_model=128,
            n_layers=2,
            memory_size=100
        )
        
        print(f"✓ Environment created: {env.W}×{env.H}")
        print(f"✓ Trainer created: obs_dim={env.observation_space.shape[0]}")
        
        # Run mini episode
        trainer.reset_episode()
        total_reward = 0
        steps = 0
        
        for _ in range(50):
            action, log_prob, value, loop_closure = trainer.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            trainer.collect(obs, action, log_prob, value, reward, done, loop_closure)
            
            obs = next_obs
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        print(f"✓ Episode completed: {steps} steps, reward={total_reward:.1f}")
        print(f"✓ Coverage: {info['region_coverage']*100:.1f}%")
        print(f"✓ Accuracy: {info['map_accuracy']*100:.1f}%")
        
        # Test training update
        if len(trainer._obs) >= 32:
            losses = trainer.train(obs, done)
            print(f"✓ Training update: policy_loss={losses.get('policy_loss', 0):.4f}")
        
        env.close()
        print("✓ Mini Training Loop: ALL TESTS PASSED\n")
        return True
        
    except Exception as e:
        print(f"✗ Mini Training Loop FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*70)
    print("MAMBA IMPLEMENTATION TEST SUITE")
    print("="*70)
    
    success = True
    
    # Test architectures
    if not test_architectures():
        success = False
    
    # Test training loop
    if not test_training_loop():
        success = False
    
    # Summary
    print("="*70)
    if success:
        print("✓ ALL TESTS PASSED!")
        print("\nYou can now:")
        print("  1. Train Pure Mamba:   python train_mamba.py --policy pure")
        print("  2. Train Hybrid:       python train_mamba.py --policy hybrid")
        print("  3. Compare both:       python train_mamba.py --compare")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nPlease check the error messages above.")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
