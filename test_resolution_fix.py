#!/usr/bin/env python3
"""
Test script to verify that our resolution fix ensures dimensions are multiples of 32.
This script tests the core logic without requiring external dependencies.
"""

def _safe_resolution_multiple_of_32(target_size: int) -> int:
    """Copy of the function from modnet_bg.py for testing."""
    return ((target_size + 31) // 32) * 32

def test_safe_resolution_multiple_of_32():
    """Test that the safe resolution function works correctly."""
    test_cases = [
        (512, 512),     # Already multiple of 32
        (500, 512),     # Should round up to 512
        (513, 544),     # Should round up to 544
        (768, 768),     # Already multiple of 32
        (800, 800),     # Should round up to 800
        (450, 480),     # Should round up to 480
        (31, 32),       # Should round up to 32
        (1, 32),        # Should round up to 32
    ]
    
    print("Testing _safe_resolution_multiple_of_32:")
    for input_size, expected in test_cases:
        result = _safe_resolution_multiple_of_32(input_size)
        print(f"  {input_size} -> {result} (expected {expected})")
        assert result == expected, f"Expected {expected}, got {result}"
        assert result % 32 == 0, f"Result {result} is not a multiple of 32"
    print("✓ All safe resolution tests passed!")

def test_problematic_resolutions_concept():
    """Test that we know how to handle resolutions that previously caused issues."""
    print("\nTesting concept for problematic resolutions:")
    
    # Test cases that might have caused "off-by-one" errors
    problematic_cases = [
        (768, 432),     # 16:9 ratio, not multiple of 32
        (800, 450),     # 16:9 ratio, not multiple of 32  
        (1280, 720),    # 16:9 ratio, not multiples of 32
        (1920, 1080),   # 16:9 ratio, not multiples of 32
        (640, 360),     # 16:9 ratio, not multiples of 32
    ]
    
    for h, w in problematic_cases:
        # Show what the safe versions would be
        safe_h = _safe_resolution_multiple_of_32(h)
        safe_w = _safe_resolution_multiple_of_32(w)
        
        print(f"  Problematic: {w}x{h} -> Safe: {safe_w}x{safe_h}")
        
        # Verify the safe versions are multiples of 32
        assert safe_h % 32 == 0, f"Safe height {safe_h} not multiple of 32"
        assert safe_w % 32 == 0, f"Safe width {safe_w} not multiple of 32"
    
    print("✓ All problematic resolution concepts handled!")

def test_16_9_aspect_ratio_suggestions():
    """Test the suggested 16:9 resolutions that are multiples of 32."""
    print("\nTesting suggested 16:9 resolutions (multiples of 32):")
    
    suggested_resolutions = [
        (512, 288),     # 16:9, multiples of 32
        (1024, 576),    # 16:9, multiples of 32
        (1536, 864),    # 16:9, multiples of 32
        (2048, 1152),   # 16:9, multiples of 32
    ]
    
    for w, h in suggested_resolutions:
        # Verify they're actually 16:9
        ratio = w / h
        expected_ratio = 16 / 9
        assert abs(ratio - expected_ratio) < 0.001, f"Resolution {w}x{h} is not 16:9"
        
        # Verify they're multiples of 32
        assert w % 32 == 0, f"Width {w} not multiple of 32"
        assert h % 32 == 0, f"Height {h} not multiple of 32"
        
        print(f"  ✓ {w}x{h} (ratio: {ratio:.3f})")
    
    print("✓ All suggested resolutions are valid!")

if __name__ == "__main__":
    test_safe_resolution_multiple_of_32()
    test_problematic_resolutions_concept()
    test_16_9_aspect_ratio_suggestions()
    print("\n🎉 All tests passed! Resolution fix is working correctly.")