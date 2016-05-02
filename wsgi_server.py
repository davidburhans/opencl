from functools import lru_cache
from threading import Lock
from wsgiref.simple_server import make_server

from mandelbrot import Mandelbrot

import re

lock = Lock()
EXT = '.png'

html = '''
<!DOCTYPE html>
<html>
<head>
<script>
    document.addEventListener("DOMContentLoaded", ready, false)
    var size = {size};
    var zoom = {zoom_factor};
    var cmap = "{cmap}";
    var xcenter = {xcenter};
    var ycenter = {ycenter};
    var original_diff = 2.5;

    function image_to_fractal(x, y) {{
        console.log('prev', xcenter, ycenter)
        var zoomed_diff = original_diff / zoom / 2;
        console.log('input', x, y, zoomed_diff);

        xmin = xcenter - zoomed_diff
        ymin = ycenter - zoomed_diff

        //x_offset = (x - (size / 2));
        //y_offset = (y - (size / 2));
        x_factor = x / (size / 2);
        y_factor = y / (size / 2);
        //console.log('offsets', x_offset, y_offset)
        console.log('factors', x_factor, y_factor)
        newx_center = xmin + (x_factor * zoomed_diff)
        newy_center = -(ymin + (y_factor * zoomed_diff))
        console.log('output', newx_center, newy_center);
        return {{x: newx_center, y: newy_center}}
    }}

    function onclick(event) {{
        center = image_to_fractal(event.offsetX, event.offsetY)
        zoom *= 1.25
        window.location = "/" + center.x + "/" + center.y + "/{size}_{cmap}_" + zoom + "x.html";
    }}

    function onmousewheel(event) {{
        center = image_to_fractal(event.offsetX, event.offsetY)
        if (event.wheelDelta < 0) {{
            zoom = zoom / 2;
        }} else if (event.wheelDelta > 0) {{
            zoom = zoom * 2;
        }}
        window.location = "/" + center.x + "/" + center.y + "/{size}_{cmap}_" + zoom + "x.html";
    }}

    function ready() {{
        fractal = document.getElementById('fractal')
        fractal.onmousewheel = onmousewheel
        fractal.onclick = onclick
        console.log('wired mouse handlers')
    }}


</script>
</head>
<body>
<img src="/{xcenter}/{ycenter}/{size}_{cmap}_{zoom_factor}x.png" id="fractal" width="{size}px" height="{size}px">
</body>
</html>
'''

# parses size_[cmap]_[zoom]x.png
def parse_filename(filename, ext=EXT):
    assert filename.endswith(ext)
    filename = filename[:-len(ext)]
    fileparts = filename.split('_')
    results = dict(zoom_factor=1, cmap='jet')
    try:
        results['size'] = int(fileparts[0])
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
    return results


def parse_zoom(zoom):
    if not zoom.endswith('x'):
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
        filename = parts.pop()
        kwargs = parse_filename(filename, ext)
    except (IndexError, AssertionError):
        kwargs = dict(cmap='jet', size=600, zoom_factor=1.0)
    kwargs['xcenter'] = xcenter
    kwargs['ycenter'] = ycenter
    return kwargs


def get_mandelbrot(environ, start_response):
    url_path = environ['PATH_INFO'].lower()
    status = '200 OK' # HTTP Status
    if url_path.endswith(EXT):
        viewport_kwargs = parse_path(url_path)
        headers = [('Content-type', 'image/png')] # HTTP Headers
        img = get_mandelbrot_details(**viewport_kwargs)
        start_response(status, headers)
        return [img.getvalue()]
    elif len(url_path) <= 1 or url_path.endswith('.html'):
        viewport_kwargs = parse_path(url_path, '.html')
        headers = [('Content-type', 'text/html; charset=UTF-8')] # HTTP Headers
        start_response(status, headers)
        result = html.format(**viewport_kwargs)
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
