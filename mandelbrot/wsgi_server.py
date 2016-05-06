from functools import lru_cache
from threading import Lock
from wsgiref.simple_server import make_server

from mandelbrot import Mandelbrot
from resources import html

import re

lock = Lock()
EXT = '.png'


# parses size_[cmap]_[zoom]x.png
def parse_filename(filename, ext=EXT):
    assert filename.lower().endswith(ext)
    filename = filename[:-len(ext)]
    fileparts = filename.split('-')
    results = dict(zoom_factor=1, cmap='jet')
    try:
        size = int(fileparts[0])
        results['width'] = size
        results['height'] = size
        try:
            results['zoom_factor'] = parse_zoom(fileparts[1])
        except ValueError:
            results['cmap'] = fileparts[1]
        try:
            results['zoom_factor'] = parse_zoom(fileparts[2])
        except ValueError:
            results['cmap'] = fileparts[2]
    except IndexError:
        pass
    # minimum zoom factor
    results['zoom_factor'] = max(0.1, results['zoom_factor'])
    return results


def parse_zoom(zoom):
    if not zoom.lower().endswith('x'):
        raise ValueError()
    return float(zoom[:-1])


# parses /xcenter/ycenter/size_[cmap]_[zoom]x.png
def parse_path(url_path, ext=EXT):
    parts = url_path.split('/')[1:]
    if len(parts) == 3:
        xcenter = float(parts[0])
        ycenter = float(parts[1])
    else:
        xcenter = -0.75
        ycenter = 0
    try:
        assert xcenter > Mandelbrot.xmin_bound
        assert xcenter < Mandelbrot.xmax_bound
        assert ycenter > Mandelbrot.ymin_bound
        assert ycenter < Mandelbrot.ymax_bound
    except AssertionError:
        xcenter = -0.75
        ycenter = 0
    try:
        filename = parts.pop()
        kwargs = parse_filename(filename, ext)
    except (IndexError, AssertionError):
        kwargs = dict(cmap='jet', width=600, height=600, zoom_factor=1.0)
    kwargs['xcenter'] = xcenter
    kwargs['ycenter'] = ycenter
    return kwargs


def get_mandelbrot(environ, start_response):
    url_path = environ['PATH_INFO'].lower()
    status = '200 OK' # HTTP Status
    if url_path.endswith(EXT):
        # maintain case for parsing color maps
        viewport_kwargs = parse_path(environ['PATH_INFO'])
        headers = [('Content-type', 'image/png')] # HTTP Headers
        img = get_mandelbrot_details(**viewport_kwargs)
        start_response(status, headers)
        return [img.getvalue()]
    elif len(url_path) <= 1 or url_path.endswith('.html'):
        # maintain case for parsing color maps
        viewport_kwargs = parse_path(environ['PATH_INFO'], '.html')
        headers = [('Content-type', 'text/html; charset=UTF-8')] # HTTP Headers
        start_response(status, headers)
        m = Mandelbrot(interactive=False, **viewport_kwargs)
        result = html.format(**m.viewport)
        return [result.encode('utf-8')]
    else:
        status = '404 Not Found' # HTTP Status
        headers = [('Content-type', 'text/plain; charset=UTF-8')] # HTTP Headers
        start_response(status, headers)
        return [''.encode('utf-8')]


@lru_cache()
def get_mandelbrot_details(**viewport_kwargs):
    # show the post with the given id, the id is an integer
    global lock
    with lock:
        m = Mandelbrot(interactive=False)
        img = m.to_png(**viewport_kwargs)
        return img

if __name__ == "__main__":
    httpd = make_server('', 5000, get_mandelbrot)
    print("Serving HTTP on port 5000...")

    # Respond to requests until process is killed
    httpd.serve_forever()
