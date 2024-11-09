import triton
import triton.language as tl
import torch

@triton.jit
def kernel_sgemm(
        a_ptr,                          # pointer to input vector A
        b_ptr,                          # pointer to input matrix b 
        out_ptr,                        # pointer to ouptut vector 
        num_elems: tl.constexpr,        # number of elements in the vectors we are adding
        block_size: tl.constexpr):      # number of elements 
    
    # triton kernels are executed across a grid of blocks, and each block corresponds
    # to an instance of the kernel. we can use the program id (i.e., kerenel instance id)
    # to identify the location of this 'program' in the grid and determine what
    # elements around it to process based on the block size.
    pid = tl.program_id(axis=0)         # axis 0 is row
    block_start = pid * block_size      # 0*2=0, 1*2=2, 2*2=4, 3*2=6, etc. 

    # list of offsets for each thread
    offsets = block_start + tl.arange(0, block_size)

    # mask to avoid out of bounds memory access
    mask = offsets < num_elems

    # load elements from vector 'a' into array of values into SRAM
    a_vals = tl.load(a_ptr + offsets, mask=mask)

    # load elements from vector 'b' into array of values into SRAM
    b_vals = tl.load(b_ptr + offsets, mask=mask)

    # do the addition
    res = a_vals + b_vals

    # store results in location of 'out_ptr' (in this case, GPU HBM)
    tl.store(out_ptr + offsets, res, mask=mask)

def vector_addition(a: torch.tensor, b: torch.tensor) -> torch.tensor:
    assert a.is_cuda and b.is_cuda
    assert a.numel() == b.numel()

    # output buffer location will be on same device as 'a'
    output_buffer = torch.empty_like(a)
    block_size = 128
    num_elems = a.numel()

    # grid size should use ceiling division to ensure we have at minimum
    # the amount of blocks required to store all elements, even if one is
    # not full. we will use a mask in the kernel to avoid accessing the
    # 'empty' elements in the block.
    grid_size = ceil_div(num_elems, block_size)
    grid = (grid_size,)
    k2 = kernel_vector_addition[grid](a, b, output_buffer, num_elems, block_size=block_size)
    return output_buffer

def ceil_div(x: int, y: int) -> int:
    return (x+y-1)//y


def verify_numeric_accuracy():
    torch.manual_seed(42)
    vec_size = 8192
    a = torch.rand(vec_size, device='cuda')
    b = torch.rand(vec_size, device='cuda')
    torch_res = a + b
    triton_res = vector_addition(a, b)
    correct = torch.allclose(torch_res, triton_res)
    print(f"Correct: {correct}")

# source: https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(12, 28, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='vector-add-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(size, provider):
    x = torch.rand(size, device='cuda', dtype=torch.float32)
    y = torch.rand(size, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__ == '__main__':
    benchmark()