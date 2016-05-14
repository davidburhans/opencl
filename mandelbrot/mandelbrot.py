from collections import namedtuple
from contextlib import contextmanager
from time import time
import numpy as np
import pyopencl as cl
import gevent
from matplotlib import pyplot as plt
from matplotlib import colors

from resources import mandelbrot_c

_devices = None
_device_iter = None
_ctx = dict()
_ctx_env = dict()
mock_thread = namedtuple('mock_thread', 'value')


def round_robin_context(names=None):
    global _devices, _device_iter, _ctx
    if _devices is None:
        _devices = []
        for plt in cl.get_platforms():
            _devices.extend(plt.get_devices())
        _device_iter = iter(_devices)
    try:
        next_device = next(_device_iter)
    except StopIteration:
        _device_iter = iter(_devices)
        next_device = next(_device_iter)
    if names is not None:
        # if isinstance(names, basestring):
        #     names = tuple(names)
        through = False
        while True:
            if next_device.name == names:
                break
            else:
                try:
                    next_device = next(_device_iter)
                except StopIteration:
                    if through:
                        break;
                    through = True
                    _device_iter = iter(_devices)

    if not _ctx.get(next_device):
        print('Creating context for', next_device.platform.name, next_device.name)
        _ctx[next_device] = cl.Context(devices=[next_device])
    return _ctx[next_device]


@contextmanager
def env_for_context(context, data):
    global _ctx_env
    mf = cl.mem_flags
    if not _ctx_env.get(context):
        queue = cl.CommandQueue(context)
        prg = cl.Program(context, mandelbrot_c).build()
        output = np.empty(data.shape, dtype=np.uint64)
        q_opencl = None
        output_opencl = cl.Buffer(context, mf.WRITE_ONLY, output.nbytes)
        _ctx_env[context] = (queue, prg, output, q_opencl, output_opencl)
    start = time()
    (queue, prg, output, q_opencl, output_opencl) = _ctx_env[context]
    # reinitialize output array if shape changes
    # if output.shape != data.shape:
    output = np.empty(data.shape, dtype=np.uint64)
    # update opencl buffer with new input
    q_opencl = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
    _ctx_env[context] = (queue, prg, output, q_opencl, output_opencl)
    try:
        yield _ctx_env[context]
    finally:
        end = time()
        device = context.devices[0]
        # print(end - start, 'seconds for', device.platform.name, device.name)


class FractalQuality:
    DIRTY = 0.75
    DRAFT = 1.5
    LOW = 2.0
    MED = 3.0
    HIGH = 4.0
    VHIGH = 8.0
    PRESENTATION = 16.0
    SILLY = 128.0


class Mandelbrot:
    quality = FractalQuality.MED
    xmin_bound = -2.0
    xmax_bound = 0.5
    ymin_bound = -1.25
    ymax_bound = 1.25
    max_zoom = 2**36
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
    _context_objs = dict()

    def get_next_context(self):
        return round_robin_context(names='Hawaii')

    @property
    def zoom_factor(self):
        return self.max_x_diff / (self.xmax - self.xmin)

    @property
    def maxiter(self):
        zoom = self.zoom_factor
        base = self.width * self.quality
        if zoom < 2:
            return base * 2
        for z in range(1, 48):
            if zoom < 2**z:
                maxiter = base*(2**(z/4))
                return maxiter
        return base*(2**12)

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
            kwargs['zoom_factor'] = max(kwargs.get('zoom_factor', 1.0), 1.0)
            kwargs['zoom_factor'] = min(kwargs.get('zoom_factor', 1.0), self.max_zoom)
            xcenter = kwargs.get('xcenter', -0.75)
            ycenter = kwargs.get('ycenter', 0.0)
            zoom_factor = kwargs.get('zoom_factor', 1.0)
            # max size / zoom multiplier = zoomed size
            center_offset = (self.max_x_diff / zoom_factor) / 2
            self.xmin = xcenter - center_offset
            self.xmax = xcenter + center_offset
            self.ymin = ycenter - center_offset
            self.ymax = ycenter + center_offset

        self.cmap = kwargs.get('cmap', self.cmap)
        self.width = kwargs.get('width', self.width)
        self.height = kwargs.get('height', self.height)
        self.dpi = kwargs.get('dpi', self.dpi)
        return self.viewport

    def _gpu(self, data):
        context = self.get_next_context()
        with env_for_context(context, data) as (queue, prg, output, q_opencl, output_opencl):
            prg.mandelbrot(queue, output.shape, None, q_opencl,
                           output_opencl, np.uint64(self.maxiter), np.double(2 ** 32))
            cl.enqueue_copy(queue, output, output_opencl).wait()
        return output

    def calculate(self, **viewport_kwargs):
        self.init_viewport(**viewport_kwargs)
        # space representing x coords
        r1 = np.linspace(self.xmin, self.xmax, self.width, dtype=np.double)
        # space representing y coords
        r2 = np.linspace(self.ymin, self.ymax, self.height, dtype=np.double)
        # convert y coords to complex equations
        equations = r1 + r2[:,None]*1j
        # convert to contiguous array of equations
        eq_array = np.ravel(equations)
        # evaluate the result of the equations
        n3 = self._gpu(eq_array)
        # convert back to `width` columns in `height` rows
        n3 = n3.reshape((self.height, self.width))
        return (r1,r2,n3)

    def build_plot(self, gamma=0.25, **viewport_kwargs):
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
            cmap=self.cmap,
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

    def to_png(self, mode=None, **kwargs):
        if not mode:
            mode = '.png'
        from io import BytesIO
        buf = BytesIO()
        fig = self.build_plot(**kwargs)
        a = fig.gca()
        a.set_frame_on(False)
        a.set_xticks([])
        a.set_yticks([])
        fig.savefig(buf, bbox_inches='tight', pad_inches=0, mode=mode)
        plt.close('all')
        buf.seek(0)
        return buf

    def gen_exponential_space(self, start, limit, count, base=10):
        rng = abs(limit - start)  # distance between limits
        result = [start]
        result.extend([start +  # initial offset
            (rng / (base**iter_count))
            for iter_count in range(count - 1, 1, -1)])
        result.append(limit)
        return np.array(result, dtype=np.float64)

    def to_animation(self, start_kwargs, end_kwargs, steps=10, delay=0.1, log_base=1.1):
        start_time = time()
        from gevent.threading import Lock
        from gifmaker import makedelta
        from io import BytesIO
        from PIL import Image
        kwargs = start_kwargs.copy()
        z_start = start_kwargs['zoom_factor']
        z_end = end_kwargs['zoom_factor']
        min_exp = 0
        max_exp = 100
        while log_base**(log_base*max_exp) > min(z_end, self.max_zoom):
            max_exp -= 1
        while log_base**(log_base*min_exp) < z_start:
            min_exp += 1
        min_exp -= 1  # we overshot
        zooms = np.logspace(log_base*min_exp, log_base*max_exp, steps, base=log_base, dtype=np.float64)

        # xcenters = np.linspace(start_kwargs['xcenter'], end_kwargs['xcenter'], steps, dtype=np.double)
        # ycenters = np.linspace(start_kwargs['ycenter'], end_kwargs['ycenter'], steps, dtype=np.double)
        x_min = min(start_kwargs['xcenter'], end_kwargs['xcenter'])
        x_max = max(start_kwargs['xcenter'], end_kwargs['xcenter'])
        if start_kwargs['xcenter'] > end_kwargs['xcenter']:
            x_dir = -1
        else:
            x_dir = 1
        xcenters = self.gen_exponential_space(x_min, x_max, steps, log_base)[::x_dir]

        y_min = min(start_kwargs['ycenter'], end_kwargs['ycenter'])
        y_max = max(start_kwargs['ycenter'], end_kwargs['ycenter'])
        if start_kwargs['ycenter'] > end_kwargs['ycenter']:
            y_dir = -1
        else:
            y_dir = 1
        ycenters = self.gen_exponential_space(y_min, y_max, steps, log_base)[::y_dir]

        # print('X', x_min, x_max, x_dir)
        # print(xcenters[0], xcenters[-1], '\n')
        # print('Y', y_min, y_max, y_dir)
        # print(ycenters[0], ycenters[-1], '\n')
        # print('Z', zooms[0], zooms[-1])
        img_cnt = 0
        threads = []
        breakout_lock = Lock()
        def loop_breakout(count, x, y, z, **kw):
            kw['xcenter'] = x
            kw['ycenter'] = y
            kw['zoom_factor'] = z # start['zoom_factor']*(z**log_base)
            # print(x, y, z)
            with breakout_lock:
                raw_png = self.to_png(mode='.gif', **kw)
            img = Image.open(raw_png).convert(mode='L')
            # img.save('ani.%03d.gif' % (count,))
            # raw_png.seek(0)
            return count, img
        for img_cnt, (x, y, z) in enumerate(zip(xcenters, ycenters, zooms)):
            thread = mock_thread(loop_breakout(img_cnt, x, y, z, **kwargs.copy()))
            # thread = gevent.spawn(loop_breakout, img_cnt, x, y, z, **kwargs.copy())
            threads.append(thread)
        if not isinstance(threads[0], mock_thread):
            gevent.joinall(threads)
        buf = BytesIO()
        gifs = tuple(gif for _, gif in sorted(t.value for t in threads))
        try:
            makedelta(buf, gifs)
        except:
            import traceback
            traceback.print_exc()
        buf.seek(0)
        print('took', time() - start_time, 'to animate')
        return buf


if __name__ == '__main__':
    m = Mandelbrot(interactive=False, cmap='Greys',)  # width=720, height=720)
    png = m.to_png()
    with open('test.png', 'wb') as f:
        f.write(png.getbuffer())
    start = dict(xcenter=-0.75, ycenter=0, zoom_factor=1.0)
    start['xcenter'] = -1.4011551890938043
    start['ycenter'] = 0
    end = start.copy()
    end['xcenter'] = -1.4011551890938043
    end['ycenter'] = 0
    # m.max_zoom = 2**60
    end['zoom_factor'] = m.max_zoom**(1/4)
    m.quality = FractalQuality.DIRTY
    with open('test_animate.gif', 'wb') as f:
        f.write(m.to_animation(start, end, steps=40, log_base=1.2).getbuffer())
