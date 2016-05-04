import numpy as np
import pyopencl as cl
from matplotlib import pyplot as plt
from matplotlib import colors


class Mandelbrot:
    xmin_bound = -2.0
    xmax_bound = 0.5
    ymin_bound = -1.25
    ymax_bound = 1.25
    max_x_diff = xmax_bound - xmin_bound
    max_y_diff = ymax_bound - ymin_bound
    xmin = xmin_bound
    xmax = xmax_bound
    ymin = ymin_bound
    ymax = ymax_bound
    width = 600
    height = 600
    dpi = 72
    cmap = 'gnuplot2'
    gpu_interactive = True

    @property
    def ctx(self):
        if not getattr(self, '_ctx', None):
            self._ctx = cl.create_some_context(interactive=self.gpu_interactive)
        return self._ctx

    @property
    def zoom_factor(self):
        return self.max_x_diff / (self.xmax - self.xmin)

    @property
    def maxiter(self):
        zoom = self.zoom_factor
        if zoom < 2**8:
            return 8000
        elif zoom < 2**16:
            return 16000
        elif zoom < 2**32:
            return 32000
        elif zoom < 2**48:
            return 64000
        return 128000

    @property
    def viewport(self):
        return dict(xmin=self.xmin,
                    xmax=self.xmax,
                    ymin=self.ymin,
                    ymax=self.ymax,
                    xcenter=(self.xmin + (self.xmax - self.xmin) / 2),
                    ycenter=(self.ymin + (self.ymax - self.ymin) / 2),
                    zoom_factor=self.zoom_factor,
                    width=self.width,
                    height=self.height,
                    dpi=self.dpi,
                    cmap=self.cmap,)

    def __init__(self, interactive=True, **viewport_kwargs):
        self.gpu_interactive = interactive
        self.init_viewport(**viewport_kwargs)

    def init_viewport(self, **kwargs):
        if 'xcenter' not in kwargs:
            self.xmin = kwargs.get('xmin', self.xmin)
            self.xmax = kwargs.get('xmax', self.xmax)
            self.ymin = kwargs.get('ymin', self.ymin)
            self.ymax = kwargs.get('ymax', self.ymax)
            center_offset = self.xmin + (self.xmax - self.xmin) / 2
        else:
            xcenter = kwargs.get('xcenter')
            ycenter = kwargs.get('ycenter', 0.0)
            zoom_factor = kwargs.get('zoom_factor', 1.0)
            # max size / zoom multiplier = zoomed size
            center_offset = (self.max_x_diff / zoom_factor) / 2
            self.xmin = xcenter - center_offset
            self.xmax = xcenter + center_offset
            self.ymin = ycenter - center_offset
            self.ymax = ycenter + center_offset

        if self.xmin <= self.xmin_bound:
            self.xmin = self.xmin_bound
            self.xmax = self.xmin + 2 * center_offset
        if self.xmax >= self.xmax_bound:
            self.xmax = self.xmax_bound
            self.xmin = self.xmax - 2 * center_offset
        if self.ymin <= self.ymin_bound:
            self.ymin = self.ymin_bound
            self.ymax = self.ymin + 2 * center_offset
        if self.ymax >= self.ymax_bound:
            self.ymax = self.ymax_bound
            self.ymin = self.ymax - 2 * center_offset

        self.width = kwargs.get('width', self.width)
        self.height = kwargs.get('height', self.height)
        self.dpi = kwargs.get('dpi', self.dpi)
        return self.viewport

    def _gpu(self, data):
        queue = cl.CommandQueue(self.ctx)
        output = np.empty(data.shape, dtype=np.uint64)
        prg = cl.Program(self.ctx, """
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
        """).build()

        mf = cl.mem_flags
        q_opencl = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
        output_opencl = cl.Buffer(self.ctx, mf.WRITE_ONLY, output.nbytes)

        prg.mandelbrot(queue, output.shape, None, q_opencl,
                       output_opencl, np.uint64(self.maxiter), np.double(2 ** 32))
        cl.enqueue_copy(queue, output, output_opencl).wait()
        return output

    def calculate(self, **viewport_kwargs):
        self.init_viewport(**viewport_kwargs)

        # space representing x coords
        r1 = np.linspace(self.xmin, self.xmax, self.width, dtype=np.float64)
        # space representing y coords
        r2 = np.linspace(self.ymin, self.ymax, self.height, dtype=np.float64)
        # convert y coords to complex equations
        equations = r1 + r2[:,None]*1j
        # convert to contiguous array of equations
        eq_array = np.ravel(equations)
        # evaluate the result of the equations
        n3 = self._gpu(eq_array)
        # convert back to `width` columns in `height` rows
        n3 = n3.reshape((self.height, self.width))
        return (r1,r2,n3)

    def build_plot(self, gamma=0.25, cmap=None, **viewport_kwargs):
        if not cmap:
            cmap = self.cmap
        else:
            self.cmap = cmap
        x, y, z = self.calculate(**viewport_kwargs)

        # convert pixels to inches for matplotlib
        img_width = self.width / self.dpi
        img_height = self.height / self.dpi

        fig = plt.figure(figsize=(img_width, img_height), dpi=self.dpi)
        ax = plt.axes()
        fig.add_axes(ax)

        norm = colors.PowerNorm(gamma)
        plot = ax.imshow(z,
            norm=norm,
            cmap=cmap,
            origin='lower')
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
