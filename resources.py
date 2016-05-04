def make_color_map_link(color_map):
    return '<a href="/{xcenter}/{ycenter}/{width}-' + color_map + '-{zoom_factor}x.html">' + color_map + '</a>'

color_maps = ['Accent','Accent_r','Blues','Blues_r','BrBG','BrBG_r','BuGn','BuGn_r','BuPu','BuPu_r','CMRmap','CMRmap_r','Dark2','Dark2_r','GnBu','GnBu_r','Greens','Greens_r','Greys','Greys_r','OrRd','OrRd_r','Oranges','Oranges_r','PRGn','PRGn_r','Paired','Paired_r','Pastel1','Pastel1_r','Pastel2','Pastel2_r','PiYG','PiYG_r','PuBu','PuBuGn','PuBuGn_r','PuBu_r','PuOr','PuOr_r','PuRd','PuRd_r','Purples','Purples_r','RdBu','RdBu_r','RdGy','RdGy_r','RdPu','RdPu_r','RdYlBu','RdYlBu_r','RdYlGn','RdYlGn_r','Reds','Reds_r','Set1','Set1_r','Set2','Set2_r','Set3','Set3_r','Spectral','Spectral_r','Wistia','Wistia_r','YlGn','YlGnBu','YlGnBu_r','YlGn_r','YlOrBr','YlOrBr_r','YlOrRd','YlOrRd_r','afmhot','afmhot_r','autumn','autumn_r','binary','binary_r','bone','bone_r','brg','brg_r','bwr','bwr_r','cool','cool_r','coolwarm','coolwarm_r','copper','copper_r','cubehelix','cubehelix_r','flag','flag_r','gist_earth','gist_earth_r','gist_gray','gist_gray_r','gist_heat','gist_heat_r','gist_ncar','gist_ncar_r','gist_rainbow','gist_rainbow_r','gist_stern','gist_stern_r','gist_yarg','gist_yarg_r','gnuplot','gnuplot2','gnuplot2_r','gnuplot_r','gray','gray_r','hot','hot_r','hsv','hsv_r','inferno','inferno_r','jet','jet_r','magma','magma_r','nipy_spectral','nipy_spectral_r','ocean','ocean_r','pink','pink_r','plasma','plasma_r','prism','prism_r','rainbow','rainbow_r','seismic','seismic_r','spectral','spectral_r','spring','spring_r','summer','summer_r','terrain','terrain_r','viridis','viridis_r','winter','winter_r',]

color_html = '''
<h2>Color Maps</h2>
<ul>
'''

for color_map in color_maps:
    color_html += '<li>' + make_color_map_link(color_map) + '</li>\n'

color_html += '</ul>'

html = '''
<!DOCTYPE html>
<html>
<head>
<title>The Mandelbrot Set</title>
<script>
    document.addEventListener("DOMContentLoaded", ready, false)
    var size = {width};
    var zoom = {zoom_factor};
    var cmap = "{cmap}";
    var xcenter = {xcenter};
    var ycenter = {ycenter};
    var original_diff = 2.5;

    function image_to_fractal(x, y) {{
        x_offset_factor = x / size;
        y_offset_factor = y / size;

        var zoom_size = original_diff / zoom;
        var zoom_center_offset = zoom_size / 2;

        // start from the left
        zoom_xmin = xcenter - zoom_center_offset;
        // start from the top
        zoom_ymax = ycenter + zoom_center_offset;

        zoom_xcenter = zoom_xmin + zoom_size * x_offset_factor;
        zoom_ycenter = zoom_ymax - zoom_size * y_offset_factor;

        return {{x: zoom_xcenter, y: zoom_ycenter}}
    }}

    function onclick(event) {{
        center = image_to_fractal(event.offsetX, event.offsetY)
        zoom *= 1.5
        window.location = "/" + center.x + "/" + center.y + "/{width}-{cmap}-" + zoom + "x.html";
    }}

    function onmousewheel(event) {{
        center = image_to_fractal(event.offsetX, event.offsetY)
        if (event.wheelDelta < 0) {{
            zoom = zoom / 2;
        }} else if (event.wheelDelta > 0) {{
            zoom = zoom * 2;
        }}
        window.location = "/" + center.x + "/" + center.y + "/{width}-{cmap}-" + zoom + "x.html";
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
<h1><a href="https://en.wikipedia.org/wiki/Mandelbrot_set">The Mandelbrot Set</a></h1>

<figure style="width:{width}px">
    <figcaption>
    Zoom: {zoom_factor}x
    <span style="float:right">{cmap}</span>
    </figcaption>
    <img style="cursor:pointer" src="/{xcenter}/{ycenter}/{width}-{cmap}-{zoom_factor}x.png" id="fractal" width="{width}px" height="{width}px">
    <figcaption>
    X: {xcenter}
    <span style="float:right">Y: {ycenter}</float>
    </figcaption>
</figure>
<h2>Instructions</h2>
<ul>
    <li>Click or mousewheel-up on the image to zoom in on that point.</li>
    <li>Mousewheel-down to zoom out.</li>
    <li>Select a color map from the list below to change colors.</li>
</ul>
''' + color_html + '''
</body>
</html>
'''
