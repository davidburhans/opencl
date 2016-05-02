import numpy as np
import pyopencl as cl
from matplotlib import pyplot as plt
from matplotlib import colors


class Mandelbrot:
    xmin = -2.0
    xmax = 0.5
    ymin = -1.25
    ymax = 1.25
    max_x_diff = xmax - xmin
    max_y_diff = ymax - ymin
    width = 600
    height = 600
    dpi = 72

    @property
    def zoom_percent(self):
        return self.max_x_diff / (self.xmax - self.xmin)

    @property
    def maxiter(self):
        return min(np.sqrt(self.zoom_percent * 2) * 80, 4000)

    def __init__(self, interactive=True, **viewport_kwargs):
        self.ctx = cl.create_some_context(interactive=interactive)
        self._init_viewport(**viewport_kwargs)

    def _init_viewport(self, **kwargs):
        if 'xcenter' not in kwargs:
            self.xmin = kwargs.get('xmin', self.xmin)
            self.xmax = kwargs.get('xmax', self.xmax)
            self.ymin = kwargs.get('ymin', self.ymin)
            self.ymax = kwargs.get('ymax', self.ymax)
        else:
            xcenter = kwargs.get('xcenter')
            ycenter = kwargs.get('ycenter')
            zoom_factor = kwargs.get('zoom_factor', 1)
            self.xmin = xcenter - (self.max_x_diff / zoom_factor) / 2
            self.xmax = xcenter + (self.max_x_diff / zoom_factor) / 2
            self.ymin = ycenter - (self.max_y_diff / zoom_factor) / 2
            self.ymax = ycenter + (self.max_y_diff / zoom_factor) / 2
        self.width = kwargs.get('width', self.width)
        self.height = kwargs.get('height', self.height)
        self.dpi = kwargs.get('dpi', self.dpi)

    def _gpu(self, data):
        queue = cl.CommandQueue(self.ctx)
        output = np.empty(data.shape, dtype=np.uint16)
        prg = cl.Program(self.ctx, """
        #pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
        __kernel void mandelbrot(
            // inputs
            __global float2 *data,
            // output
            __global ushort *output,
            // max iterations
            ushort const maxiter,
            // minimum divergence to stop iteration
            float const horizon)
        {
            // get the index for our data
            int gid = get_global_id(0);
            float real = data[gid].x;
            float imag = data[gid].y;
            // initialize output
            output[gid] = 0;
            for(int curiter = 0; curiter < maxiter; curiter++) {
                float real2 = real*real, imag2 = imag*imag;
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
        """).build()

        mf = cl.mem_flags
        q_opencl = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
        output_opencl = cl.Buffer(self.ctx, mf.WRITE_ONLY, output.nbytes)

        prg.mandelbrot(queue, output.shape, None, q_opencl,
                       output_opencl, np.uint16(self.maxiter), np.float32(2 ** 32))
        cl.enqueue_copy(queue, output, output_opencl).wait()
        return output

    def calculate(self, **viewport_kwargs):
        self._init_viewport(**viewport_kwargs)

        # space representing x coords
        r1 = np.linspace(self.xmin, self.xmax, self.width, dtype=np.float32)
        # space representing y coords
        r2 = np.linspace(self.ymin, self.ymax, self.height, dtype=np.float32)
        # convert y coords to complex equations
        equations = r1 + r2[:,None]*1j
        # convert to contiguous array of equations
        eq_array = np.ravel(equations)
        # evaluate the result of the equations
        n3 = self._gpu(eq_array)
        # convert back to `width` columns in `height` rows
        n3 = n3.reshape((self.height, self.width))
        return (r1,r2,n3)

    def build_plot(self, gamma=0.3, cmap='jet', **viewport_kwargs):
        x, y, z = self.calculate(**viewport_kwargs)

        # convert pixels to inches for matplotlib
        img_width = self.width / self.dpi
        img_height = self.height / self.dpi

        fig = plt.figure(figsize=(img_width, img_height), dpi=self.dpi)
        ax = plt.axes()
        fig.add_axes(ax)

        norm = colors.PowerNorm(gamma)
        plot = ax.imshow(z, cmap=cmap, norm=norm, origin='lower')
        return fig

    def render(self, **kwargs):
        plt.show(self.build_plot(**kwargs))

    def zoom(self, x=-0.75, y=0, factor=2):
        assert factor > 0
        x_range = self.xmax - self.xmin
        y_range = self.ymax - self.ymin
        x_offset = x_range / factor / 2
        y_offset = y_range / factor / 2
        self.xmin = x - x_offset
        self.xmax = x + x_offset
        self.ymin = y - y_offset
        self.ymax = y + y_offset

    def to_png(self, **kwargs):
        from io import BytesIO
        buf = BytesIO()
        fig = self.build_plot(**kwargs)
        a = fig.gca()
        a.set_frame_on(False)
        a.set_xticks([])
        a.set_yticks([])
        fig.savefig(buf, bbox_inches='tight', pad_inches=0, mode='png')
        plt.close('all')
        buf.seek(0)
        return buf

if __name__ == '__main__':
    m = Mandelbrot(interactive=False)
    png = m.to_png()
    with open('test.png', 'wb') as f:
        f.write(png.getbuffer())
