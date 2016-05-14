#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
__kernel void mandelbrot(
    // inputs
    __global double2 *data,
    // output
    __global ulong *output,
    // max iterations
    ulong const maxiter,
    // minimum divergence to stop iteration
    double const horizon)
{
    // get the index for our data
    int gid = get_global_id(0);
    double real = data[gid].x;
    double imag = data[gid].y;
    // initialize output
    output[gid] = 0;
    for(ulong curiter = 0; curiter < maxiter; curiter++) {
        double real2 = real*real, imag2 = imag*imag;
        // calculations quickly diverging -- we have a result
        if (real2 + imag2 > horizon){
             output[gid] = curiter;
             return;
        }
        // calculate values for next iteration
        imag = 2* real*imag + data[gid].y;
        real = real2 - imag2 + data[gid].x;
    }
}
