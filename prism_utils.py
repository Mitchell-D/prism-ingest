""" methods for downloading from the PRISM database """
import io
import requests
import zipfile
import rasterio as rio
import numpy as np

base_url_gridcount = "https://services.nacse.org/prism/data/get/gridCount/"
base_url_data = "https://services.nacse.org/prism/data/get"
prism_tif_format = "prism_{feat}_{region}_25m_{yyyymmdd}.tif"

def mp_get_prism_grid_count(args):
    return args,get_prism_grid_count(**args)

def get_prism_grid_count(yyyymmdd, region="us", feat="ppt"):
    """
    Request the "grid count", or the number of times the current file has
    been reprocessed, in order to track updates.
    """
    url = base_url_gridcount + f"/{region}/{feat}/{yyyymmdd}"
    resp = requests.get(url)
    if resp.status_code == 200:
        r = resp.content.decode("utf-8")
        if r.isnumeric():
            return int(r)
        else:
            return -1
    else:
        raise ValueError(f"Request failed: {url}")

def mp_download_prism_day(args):
    return args,download_prism_day(**args)

def download_prism_day(yyyymmdd:str, region="us", res="4km", feat="ppt",
        get_grid_count=True, file_format=prism_tif_format, geotiff_path=None):
    """
    Download a PRISM file and extract its array and metadata into a dict
    """
    url = base_url_data + f"/{region}/{res}/{feat}/{yyyymmdd}"
    target = file_format.format(feat=feat, region=region, yyyymmdd=yyyymmdd)

    gc = None
    if get_grid_count:
        gc = get_prism_grid_count(yyyymmdd, region, feat)

    resp = requests.get(url)
    if resp.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zr:
            if target in zr.namelist():
                with zr.open(target) as tr:
                    with rio.open(tr) as src:
                        x = src.read(1)
                        if geotiff_path:
                            pout = src.profile
                            with rio.open(geotiff_path, "w", **pout) as dst:
                                dst.write(np.array([x]))
                        return {
                            "data":np.where(x==-9999., np.nan, x),
                            "crs":src.crs.to_string(),
                            "transform":list(src.transform),
                            "bounds":src.bounds._asdict(),
                            "resolution":list(src.res),
                            "width":int(src.width),
                            "height":int(src.height),
                            "grid_count":gc,
                            }
            else:
                raise ValueError(f"Target {target} not in zip: {url}")
    else:
        raise ValueError(f"Request failed: {url}")

def load_local_prism_day(tif_path, grid_count=8):
    """
    Read a local file into a dict containing the array and metadata in the
    same format as download_prism_day
    """
    with rio.open(tif_path.as_posix()) as src:
        x = src.read(1)
        return {
            "data":np.where(x==-9999., np.nan, x),
            "crs":src.crs.to_string(),
            "transform":list(src.transform),
            "bounds":src.bounds._asdict(),
            "resolution":list(src.res),
            "width":int(src.width),
            "height":int(src.height),
            "grid_count":grid_count,
            }

def compare_cog_metadata(meta_a, meta_b, thresh=1e-6):
    for k in ["crs", "width", "height"]:
        if meta_a[k] != meta_b[k]:
            return False,(k, meta_a[k], meta_b[k])
    for k in ["left", "bottom", "right", "top"]:
        db = meta_a["bounds"][k] - meta_b["bounds"][k]
        if abs(db) > thresh:
            return False,("bounds", k, db)
    for i,(afa,afb) in enumerate(zip(
            list(meta_a["transform"]),
            list(meta_b["transform"]),
            )):
        dt = afa-afb
        if abs(dt) > thresh:
            return False,("transform", i, dt)
    for i,(ra,rb) in enumerate(zip(meta_a["resolution"],meta_b["resolution"])):
        dr = ra-rb
        if abs(dr) > thresh:
            return False,("resolution",i,dr)
    return True,None
