"""
Test suite for optimized Mamba implementations
"""

import torch
import numpy as np
from agent.mamba_memory_fast import (
    FastMambaMemorySLAMPolicy, 
    UltraFastMambaSLAMPolicy,
    compare_speeds
)
from agent.mamba_trainer_fast import FastMambaPPOTrainer


def test_fast_hybrid():
    """Test Fast Hybrid policy"""
    print("\n1. Testing Fast Hybrid Policy")
    print("-" * 70)
    
    obs_dim = 390
    n_actions = 8
    
    model = FastMambaMemorySLAMPolicy(
        obs_dim=obs_dim,
        n_actions=n_actions,
        d_model=128,
        n_layers=2,
        memory_size=500
    )
    
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model created: {params:,} parameters")
    
    # Forward pass
    obs = torch.randn(4, obs_dim)
    outputs = model(obs, update_buffer=False, update_memory=False)
    
    assert outputs['logits'].shape == (4, n_actions)
    assert outputs['value'].shape == (4,)
    assert outputs['loop_closure_prob'].shape == (4,)
    print(f"✓ Forward pass: logits {outputs['logits'].shape}, value {outputs['value'].shape}")
    
    # Sequential processing
    model.reset_sequence()
    for i in range(10):
        obs_single = torch.randn(1, obs_dim)
        outputs = model(obs_single, update_buffer=True, update_memory=True)
    
    print(f"✓ Sequential processing: 10 steps completed")
    print(f"✓ Memory bank: {model.memory_bank.cache_valid.sum().item()} landmarks stored")
    print(f"✓ Fast Hybrid: ALL TESTS PASSED\n")


def test_ultra_fast():
    """Test Ultra Fast policy"""
    print("2. Testing Ultra Fast Policy")
    print("-" * 70)
    
    obs_dim = 390
    n_actions = 8
    
    model = UltraFastMambaSLAMPolicy(
        obs_dim=obs_dim,
        n_actions=n_actions,
        d_model=64,
        n_layers=2
    )
    
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model created: {params:,} parameters")
    
    # Forward pass
    obs = torch.randn(4, obs_dim)
    logits, value = model(obs, update_buffer=False)
    
    assert logits.shape == (4, n_actions)
    assert value.shape == (4,)
    print(f"✓ Forward pass: logits {logits.shape}, value {value.shape}")
    
    # Sequential processing
    model.reset_sequence()
    for i in range(10):
        obs_single = torch.randn(1, obs_dim)
        logits, value = model(obs_single, update_buffer=True)
    
    print(f"✓ Sequential processing: 10 steps completed")
    print(f"✓ Ultra Fast: ALL TESTS PASSED\n")


def test_trainers():
    """Test trainers"""
    print("3. Testing Fast Trainers")
    print("-" * 70)
    
    obs_dim = 390
    n_actions = 8
    
    # Fast Hybrid trainer
    trainer1 = FastMambaPPOTrainer(
        obs_dim=obs_dim,
        n_actions=n_actions,
        policy_type='fast_hybrid',
        d_model=128,
        n_layers=2
    )
    
    obs = np.random.randn(obs_dim).astype(np.float32)
    action, log_prob, value, loop_closure = trainer1.act(obs)
    
    print(f"✓ Fast Hybrid trainer: action={action}, value={value:.3f}, loop={loop_closure:.3f}")
    
    # Ultra Fast trainer
    trainer2 = FastMambaPPOTrainer(
        obs_dim=obs_dim,
        n_actions=n_actions,
        policy_type='ultra_fast',
        d_model=64,
        n_layers=2
    )
    
    action, log_prob, value = trainer2.act(obs)
    print(f"✓ Ultra Fast trainer: action={action}, value={value:.3f}")
    print(f"✓ Fast Trainers: ALL TESTS PASSED\n")


def main():
    print("\n" + "="*70)
    print("FAST MAMBA IMPLEMENTATION TEST SUITE")
    print("="*70)
    
    test_fast_hybrid()
    test_ultra_fast()
    test_trainers()
    
    print("="*70)
    print("✓ ALL TESTS PASSED!")
    print("\nYou can now train with:")
    print("  python train_mamba_fast.py --policy fast_hybrid    # 10-30 FPS")
    print("  python train_mamba_fast.py --policy ultra_fast     # 30-100 FPS")
    print("="*70 + "\n")
    
    # Show comparison
    compare_speeds()


if __name__ == '__main__':
    main()
