# Memory Optimization Summary for train_v3.py

## Overview
The current data preparation approach in `train_v3.py` has been optimized for memory efficiency following the best practices from the grayscale example. The optimization significantly reduces memory usage while maintaining training performance.

## Key Changes Made

### 1. Optimized Data Loading Functions
- **`load_preprocessed_frames()`**: Now loads images directly in `uint8` format instead of converting to `float32` immediately
- **`load_preprocessed_heatmaps()`**: Similarly optimized to use `uint8` format during loading
- **Memory savings**: ~75% reduction compared to `float32` (1 byte vs 4 bytes per pixel)

### 2. Updated Data Processing Pipeline
- **`load_preprocessed_data()`**: Performs normalization (uint8 → float32) only when needed for TensorFlow operations
- **`preload_dataset_to_memory()`**: Stores data in uint8 format and provides detailed memory usage statistics
- **`load_preloaded_data()`**: Handles conversion from uint8 to float32 during training

### 3. New Memory-Efficient Dataset Class
Added `MemoryEfficientDataset` class based on the grayscale example pattern:
- Implements smart caching with uint8 optimization
- Supports both preload-to-memory and disk-based loading modes
- Provides detailed memory usage reporting
- Maintains compatibility with TensorFlow data pipelines

### 4. Enhanced Command Line Interface
- **`--use_memory_efficient`**: New flag to enable the memory-efficient dataset class
- **`--preload_memory`**: Works with both approaches (original and memory-efficient)

## Memory Usage Improvements

### Before Optimization
```
Frame: 288 × 512 × 3 × 4 bytes (float32) = ~1.77 MB per frame
Heatmap: 288 × 512 × 1 × 4 bytes (float32) = ~0.59 MB per heatmap
Sequence (3 frames): (1.77 + 0.59) × 3 = ~7.08 MB per sequence
```

### After Optimization
```
Frame: 288 × 512 × 3 × 1 byte (uint8) = ~0.44 MB per frame
Heatmap: 288 × 512 × 1 × 1 byte (uint8) = ~0.15 MB per heatmap
Sequence (3 frames): (0.44 + 0.15) × 3 = ~1.77 MB per sequence
```

**Total Memory Savings: ~75% reduction**

## Usage Examples

### Memory-Efficient Approach (Recommended)
```bash
# Using the new memory-efficient class with preloading
python src/train_v3.py --use_memory_efficient --preload_memory --grayscale --seq 3

# Using memory-efficient class without preloading (for very large datasets)
python src/train_v3.py --use_memory_efficient --grayscale --seq 3
```

### Original Approach (with optimizations)
```bash
# Using original approach with uint8 optimizations
python src/train_v3.py --preload_memory --grayscale --seq 3
```

## Technical Details

### Memory Usage Reporting
The optimized version provides detailed memory usage statistics:
- Total dataset memory consumption
- Per-component breakdown (frames vs heatmaps)
- Estimated savings compared to float32
- Progress monitoring during preloading

### Compatibility
- Fully backward compatible with existing training workflows
- Maintains all original functionality and augmentation support
- Works with all model architectures (VballNetV1, VballNetFastV1, etc.)
- Supports both grayscale and RGB modes

### Performance Characteristics
- **Memory Usage**: 75% reduction
- **Loading Speed**: Comparable or faster due to smaller data transfers
- **Training Speed**: Identical (normalization moved to training time)
- **Disk Usage**: No change (affects only RAM usage)

## Best Practices

1. **For Small-Medium Datasets**: Use `--use_memory_efficient --preload_memory`
2. **For Large Datasets**: Use `--use_memory_efficient` without preloading
3. **For Limited Memory Systems**: Always use the memory-efficient approach
4. **For Debugging**: Use the original approach for compatibility with existing tools

## Implementation Notes

- The optimization is based on the proven grayscale example pattern
- Uses established techniques from the project's memory usage guidelines
- Follows the project's coding standards and error handling patterns
- Includes comprehensive logging for monitoring and debugging

This optimization enables training on larger datasets with the same hardware resources and reduces the risk of out-of-memory errors during training.