# SD_mosaic
create an image mosaique based via stable diffusion


[Photographic mosaic](https://en.wikipedia.org/wiki/Photographic_mosaic)

## installation

1. make sure you have a computer with a cuda graphics card (NVidia), otherwise you'll need to adapt the code for the line `device="cuda"` to something appropriate.
2. create a virtual environment:
```bash
cd <this directory>
python -m venv .venv
venv\Scripts\activate
python -m pip install -r requirements.txt
```
3. run the file once to download the necessary AI models, go get yourself a coffee in the meantime.
```bash
python sd_image_mosaic.py wio.jpg "constructivist style tree leaves . geometric shapes, bold colors, dynamic composition, propaganda art style"
```

## running directly

```bash
venv\Scripts\activate
python sd_image_mosaic.py <YOUR IMAGE> "<YOUR PROMPT>"
```

## running via interface

```bash
venv\Scripts\activate
python interface.py
```