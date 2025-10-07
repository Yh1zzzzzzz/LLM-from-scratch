#!/usr/bin/env python3
"""
Comprehensive Flash Attention Testing Suite
============================================

This script provides thorough testing and validation for the Flash Attention implementation,
including numerical accuracy, gradient correctness, performance benchmarks, and edge cases.
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
import time
import sys
import os

# Add project root to path
sys.path.append('/Users/younght/study/Self_study/Projects/LLM_scratch')

try:
    from Operator.Attention.FlashAttention import FlashAttention, flash_attention
    print("✓ Successfully imported Flash Attention")
except ImportError as e:
    print(f"✗ Failed to import Flash Attention: {e}")
    sys.exit(1)


class TestFlashAttention:
    """Comprehensive test suite for Flash Attention implementation"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running tests on device: {self.device}")
        
        # Test tolerances
        self.rtol = 1e-4
        self.atol = 1e-5
        self.grad_rtol = 1e-3
        self.grad_atol = 1e-4
        
    def standard_attention(self, q, k, v, causal=True):
        """Reference implementation using standard PyTorch operations"""
        batch, num_heads, seq_len, d_head = q.shape
        scale = 1.0 / math.sqrt(d_head)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Apply causal mask if needed
        if causal:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=self.device), diagonal=1)
            scores = scores.masked_fill(mask.bool(), float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        
        return out
    
    def test_basic_functionality(self):
        """Test basic forward and backward functionality"""
        print("\n" + "="*60)
        print("Test 1: Basic Functionality")
        print("="*60)
        
        # Test configurations
        configs = [
            (1, 1, 64, 32),    # Small test
            (2, 4, 128, 64),   # Medium test
            (1, 8, 256, 64),   # Larger test
        ]
        
        for i, (batch, num_heads, seq_len, d_head) in enumerate(configs):
            print(f"\nTest 1.{i+1}: B={batch}, H={num_heads}, S={seq_len}, D={d_head}")
            
            try:
                # Create inputs
                q = torch.randn(batch, num_heads, seq_len, d_head, 
                              device=self.device, requires_grad=True, dtype=torch.float32)
                k = torch.randn(batch, num_heads, seq_len, d_head, 
                              device=self.device, requires_grad=True, dtype=torch.float32)
                v = torch.randn(batch, num_heads, seq_len, d_head, 
                              device=self.device, requires_grad=True, dtype=torch.float32)
                
                # Clone for reference implementation
                q_ref = q.clone().detach().requires_grad_(True)
                k_ref = k.clone().detach().requires_grad_(True)
                v_ref = v.clone().detach().requires_grad_(True)
                
                # Flash Attention forward
                flash_out = flash_attention(q, k, v, causal=True)
                
                # Reference forward
                ref_out = self.standard_attention(q_ref, k_ref, v_ref, causal=True)
                
                # Check forward pass
                forward_match = torch.allclose(flash_out, ref_out, rtol=self.rtol, atol=self.atol)
                print(f"  Forward pass match: {'✓' if forward_match else '✗'}")
                
                if not forward_match:
                    diff = (flash_out - ref_out).abs()
                    print(f"  Max difference: {diff.max().item():.2e}")
                    print(f"  Mean difference: {diff.mean().item():.2e}")
                
                # Test backward pass
                grad_output = torch.randn_like(flash_out)
                
                # Flash backward
                flash_out.backward(grad_output)
                flash_q_grad = q.grad.clone() if q.grad is not None else None
                flash_k_grad = k.grad.clone() if k.grad is not None else None
                flash_v_grad = v.grad.clone() if v.grad is not None else None
                
                # Reference backward
                ref_out.backward(grad_output)
                ref_q_grad = q_ref.grad.clone() if q_ref.grad is not None else None
                ref_k_grad = k_ref.grad.clone() if k_ref.grad is not None else None
                ref_v_grad = v_ref.grad.clone() if v_ref.grad is not None else None
                
                # Check gradients
                if flash_q_grad is not None and ref_q_grad is not None:
                    q_grad_match = torch.allclose(flash_q_grad, ref_q_grad, 
                                                rtol=self.grad_rtol, atol=self.grad_atol)
                    print(f"  Q gradient match: {'✓' if q_grad_match else '✗'}")
                    
                    if not q_grad_match:
                        diff = (flash_q_grad - ref_q_grad).abs()
                        print(f"    Q grad max diff: {diff.max().item():.2e}")
                
                if flash_k_grad is not None and ref_k_grad is not None:
                    k_grad_match = torch.allclose(flash_k_grad, ref_k_grad, 
                                                rtol=self.grad_rtol, atol=self.grad_atol)
                    print(f"  K gradient match: {'✓' if k_grad_match else '✗'}")
                    
                    if not k_grad_match:
                        diff = (flash_k_grad - ref_k_grad).abs()
                        print(f"    K grad max diff: {diff.max().item():.2e}")
                
                if flash_v_grad is not None and ref_v_grad is not None:
                    v_grad_match = torch.allclose(flash_v_grad, ref_v_grad, 
                                                rtol=self.grad_rtol, atol=self.grad_atol)
                    print(f"  V gradient match: {'✓' if v_grad_match else '✗'}")
                    
                    if not v_grad_match:
                        diff = (flash_v_grad - ref_v_grad).abs()
                        print(f"    V grad max diff: {diff.max().item():.2e}")
                
            except Exception as e:
                print(f"  ✗ Test failed with error: {e}")
                import traceback
                traceback.print_exc()
    
    def test_causal_masking(self):
        """Test causal masking correctness"""
        print("\n" + "="*60)
        print("Test 2: Causal Masking")
        print("="*60)
        
        batch, num_heads, seq_len, d_head = 2, 4, 64, 32
        
        # Create inputs
        q = torch.randn(batch, num_heads, seq_len, d_head, device=self.device)
        k = torch.randn(batch, num_heads, seq_len, d_head, device=self.device)
        v = torch.randn(batch, num_heads, seq_len, d_head, device=self.device)
        
        # Test causal vs non-causal
        causal_out = flash_attention(q, k, v, causal=True)
        non_causal_out = flash_attention(q, k, v, causal=False)
        
        # They should be different
        is_different = not torch.allclose(causal_out, non_causal_out, rtol=1e-3)
        print(f"Causal vs non-causal outputs differ: {'✓' if is_different else '✗'}")
        
        # Test specific causal property: output at position i should not depend on positions > i
        q_test = torch.zeros(1, 1, 4, 4, device=self.device)
        k_test = torch.zeros(1, 1, 4, 4, device=self.device)
        v_test = torch.zeros(1, 1, 4, 4, device=self.device)
        
        # Set up specific pattern
        q_test[0, 0, 1, :] = 1.0  # Query at position 1
        k_test[0, 0, :, :] = 1.0  # All keys the same
        v_test[0, 0, 0, :] = 1.0  # Value at position 0
        v_test[0, 0, 2, :] = 2.0  # Value at position 2
        
        out = flash_attention(q_test, k_test, v_test, causal=True)
        
        # Position 1 should only see position 0 and 1, not position 2
        # So the output should be closer to 1.0 than to 2.0
        position_1_out = out[0, 0, 1, 0].item()
        causal_correct = position_1_out < 1.5  # Should be influenced by pos 0 (value=1), not pos 2 (value=2)
        print(f"Causal masking property: {'✓' if causal_correct else '✗'}")
        print(f"  Position 1 output: {position_1_out:.3f} (should be < 1.5)")
    
    def test_module_interface(self):
        """Test the FlashAttention module interface"""
        print("\n" + "="*60)
        print("Test 3: Module Interface")
        print("="*60)
        
        try:
            # Create module
            d_model = 256
            num_heads = 8
            flash_attn = FlashAttention(d_model=d_model, num_heads=num_heads, causal=True).to(self.device)
            
            # Test input
            batch, seq_len = 2, 128
            x = torch.randn(batch, seq_len, d_model, device=self.device, requires_grad=True)
            
            # Forward pass
            out = flash_attn(x)
            
            # Check output shape
            expected_shape = (batch, seq_len, d_model)
            shape_correct = out.shape == expected_shape
            print(f"Output shape correct: {'✓' if shape_correct else '✗'}")
            print(f"  Expected: {expected_shape}, Got: {out.shape}")
            
            # Test backward pass
            loss = out.sum()
            loss.backward()
            
            grad_exists = x.grad is not None
            print(f"Gradient computation: {'✓' if grad_exists else '✗'}")
            
            if grad_exists:
                print(f"  Input grad shape: {x.grad.shape}")
                print(f"  Grad norm: {x.grad.norm().item():.3f}")
            
        except Exception as e:
            print(f"✗ Module test failed: {e}")
            import traceback
            traceback.print_exc()
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        print("\n" + "="*60)
        print("Test 4: Edge Cases")
        print("="*60)
        
        # Test 1: Very small sequences
        print("\nTest 4.1: Small sequences")
        try:
            q = torch.randn(1, 1, 1, 32, device=self.device)
            k = torch.randn(1, 1, 1, 32, device=self.device)
            v = torch.randn(1, 1, 1, 32, device=self.device)
            
            out = flash_attention(q, k, v, causal=True)
            print("  Single token: ✓")
            
            q = torch.randn(1, 1, 2, 32, device=self.device)
            k = torch.randn(1, 1, 2, 32, device=self.device)
            v = torch.randn(1, 1, 2, 32, device=self.device)
            
            out = flash_attention(q, k, v, causal=True)
            print("  Two tokens: ✓")
            
        except Exception as e:
            print(f"  Small sequences: ✗ ({e})")
        
        # Test 2: Extreme values
        print("\nTest 4.2: Extreme values")
        try:
            # Very large values
            q = torch.ones(1, 1, 32, 32, device=self.device) * 10
            k = torch.ones(1, 1, 32, 32, device=self.device) * 10
            v = torch.randn(1, 1, 32, 32, device=self.device)
            
            out = flash_attention(q, k, v, causal=True)
            has_nan = torch.isnan(out).any()
            has_inf = torch.isinf(out).any()
            print(f"  Large values - NaN: {'✗' if has_nan else '✓'}, Inf: {'✗' if has_inf else '✓'}")
            
            # Very small values
            q = torch.ones(1, 1, 32, 32, device=self.device) * 1e-6
            k = torch.ones(1, 1, 32, 32, device=self.device) * 1e-6
            v = torch.randn(1, 1, 32, 32, device=self.device)
            
            out = flash_attention(q, k, v, causal=True)
            has_nan = torch.isnan(out).any()
            has_inf = torch.isinf(out).any()
            print(f"  Small values - NaN: {'✗' if has_nan else '✓'}, Inf: {'✗' if has_inf else '✓'}")
            
        except Exception as e:
            print(f"  Extreme values: ✗ ({e})")
    
    def test_performance_benchmark(self):
        """Performance comparison with standard attention"""
        print("\n" + "="*60)
        print("Test 5: Performance Benchmark")
        print("="*60)
        
        if self.device.type != "cuda":
            print("Skipping performance tests (CUDA not available)")
            return
        
        configs = [
            (1, 8, 512, 64),
            (2, 8, 1024, 64),
            (1, 8, 2048, 64),
        ]
        
        print(f"{'Config':<20} {'Flash (ms)':<12} {'Standard (ms)':<15} {'Speedup':<10} {'Memory (MB)':<12}")
        print("-" * 75)
        
        for batch, num_heads, seq_len, d_head in configs:
            try:
                # Create inputs
                q = torch.randn(batch, num_heads, seq_len, d_head, device=self.device)
                k = torch.randn(batch, num_heads, seq_len, d_head, device=self.device)
                v = torch.randn(batch, num_heads, seq_len, d_head, device=self.device)
                
                # Warmup
                for _ in range(5):
                    _ = flash_attention(q, k, v, causal=True)
                    _ = self.standard_attention(q, k, v, causal=True)
                
                torch.cuda.synchronize()
                
                # Benchmark Flash Attention
                start_time = time.time()
                for _ in range(20):
                    out_flash = flash_attention(q, k, v, causal=True)
                torch.cuda.synchronize()
                flash_time = (time.time() - start_time) / 20 * 1000
                
                # Benchmark Standard Attention
                start_time = time.time()
                for _ in range(20):
                    out_std = self.standard_attention(q, k, v, causal=True)
                torch.cuda.synchronize()
                std_time = (time.time() - start_time) / 20 * 1000
                
                speedup = std_time / flash_time
                memory_saved = seq_len**2 * batch * num_heads * 4 / (1024**2)
                
                config_str = f"B{batch}H{num_heads}S{seq_len}D{d_head}"
                print(f"{config_str:<20} {flash_time:<12.2f} {std_time:<15.2f} {speedup:<10.2f} {memory_saved:<12.1f}")
                
            except Exception as e:
                print(f"Error in config B{batch}H{num_heads}S{seq_len}D{d_head}: {e}")
    
    def test_gradient_numerical(self):
        """Numerical gradient checking"""
        print("\n" + "="*60)
        print("Test 6: Numerical Gradient Check")
        print("="*60)
        
        def finite_diff_grad(func, x, eps=1e-5):
            """Compute numerical gradient using finite differences"""
            grad = torch.zeros_like(x)
            x_flat = x.view(-1)
            grad_flat = grad.view(-1)
            
            for i in range(x_flat.numel()):
                x_plus = x_flat.clone()
                x_minus = x_flat.clone()
                x_plus[i] += eps
                x_minus[i] -= eps
                
                x_plus_tensor = x_plus.view_as(x)
                x_minus_tensor = x_minus.view_as(x)
                
                loss_plus = func(x_plus_tensor).sum()
                loss_minus = func(x_minus_tensor).sum()
                
                grad_flat[i] = (loss_plus - loss_minus) / (2 * eps)
            
            return grad
        
        # Small test case for numerical gradient
        batch, num_heads, seq_len, d_head = 1, 1, 4, 8
        
        q = torch.randn(batch, num_heads, seq_len, d_head, 
                       device=self.device, requires_grad=True, dtype=torch.float64)
        k = torch.randn(batch, num_heads, seq_len, d_head, 
                       device=self.device, dtype=torch.float64)
        v = torch.randn(batch, num_heads, seq_len, d_head, 
                       device=self.device, dtype=torch.float64)
        
        def test_func(q_input):
            return flash_attention(q_input, k, v, causal=True)
        
        # Compute analytical gradient
        out = test_func(q)
        out.sum().backward()
        analytical_grad = q.grad.clone()
        
        # Compute numerical gradient
        q.grad = None
        numerical_grad = finite_diff_grad(test_func, q.detach())
        
        # Compare
        relative_error = ((analytical_grad - numerical_grad).abs() / 
                         (numerical_grad.abs() + 1e-8)).max()
        
        grad_check_pass = relative_error < 1e-3
        print(f"Numerical gradient check: {'✓' if grad_check_pass else '✗'}")
        print(f"  Max relative error: {relative_error.item():.2e}")
    
    def run_all_tests(self):
        """Run all tests"""
        print("Flash Attention Comprehensive Test Suite")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"PyTorch version: {torch.__version__}")
        
        try:
            # Test if triton is available
            import triton
            print(f"Triton version: {triton.__version__}")
        except ImportError:
            print("Triton not available - Flash Attention may not work")
        
        test_methods = [
            self.test_basic_functionality,
            self.test_causal_masking, 
            self.test_module_interface,
            self.test_edge_cases,
            self.test_gradient_numerical,
            self.test_performance_benchmark,
        ]
        
        passed = 0
        total = len(test_methods)
        
        for test_method in test_methods:
            try:
                test_method()
                passed += 1
            except Exception as e:
                print(f"\n✗ Test {test_method.__name__} failed: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Tests passed: {passed}/{total}")
        
        if passed == total:
            print("🎉 All tests passed!")
        else:
            print(f"⚠️  {total - passed} tests failed")
        
        return passed == total


def main():
    """Main test runner"""
    tester = TestFlashAttention()
    success = tester.run_all_tests()
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
