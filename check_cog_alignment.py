"""
Preprocessing to convert Alabama LGI data stored in COGs into zarr store
for the web viewer backend.
"""
import numpy as np
import json
import zarr
zarr.config.set({"async.concurrency":128})
import rasterio as rio
import geopandas
import multiprocessing as mp
import math
from pprint import pprint
from rasterio.warp import transform_bounds
from datetime import datetime
from pathlib import Path

def extract_cog_metadata(cog_path:Path):
    """ Extract the domain and crs information for a cog file """
    with rio.open(cog_path) as src:
        return {
            "crs":src.crs,
            "bounds": src.bounds._asdict(),
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
            "resolution": src.res
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

def extract_common_window(cog_paths:list):
    ## get a reference grid geometry from the first file
    with rio.open(cog_paths[0]) as ref:
        ref_crs = ref.crs
        ref_trans = ref.transform
        res_x, res_y = ref.res

    ## compute the intersection of all the grid bounds
    bounds_list = []
    for p in cog_paths:
        with rio.open(p) as src:
            b = transform_bounds(src.crs, ref_crs, *src.bounds)
            bounds_list.append(b)
    xmin = max(b[0] for b in bounds_list)
    ymin = max(b[1] for b in bounds_list)
    xmax = min(b[2] for b in bounds_list)
    ymax = min(b[3] for b in bounds_list)

    if xmin >= xmax or ymin >= ymax:
        raise ValueError("No overlapping region found.")

    ## snap to the reference grid so that float differences or half-pixel
    ## offsets don't affect the uniformity of the window
    origin_x = ref_trans.c
    origin_y = ref_trans.f

    def snap(val, origin, res, fn):
        return origin + fn((val - origin) / res) * res

    xmin_s = snap(xmin, origin_x, res_x, math.ceil)
    xmax_s = snap(xmax, origin_x, res_x, math.floor)
    ymin_s = snap(ymin, origin_y, -res_y, math.ceil)
    ymax_s = snap(ymax, origin_y, -res_y, math.floor)

    ## return bounds
    return xmin_s,ymin_s,xmax_s,ymax_s

    ## return a window based on the bounds
    #return rio.windows.from_bounds(
    #        xmin_s, ymin_s, xmax_s, ymax_s, transform=ref_trans)

def check_cog_alignment(ref_path, other_path, tol=1e-6):
    with rio.open(ref_path) as ref, rio.open(other_path) as src:
        # Check resolution
        if not np.allclose(ref.res, src.res, atol=tol):
            rry,rrx = ref.res
            rsy,rsx = src.res
            return f"res diff:({rsy-rry}, {rsx-rrx}) " + \
                    f"ref:{ref.res} other:{src.res}"

        # Compute offset in pixel units
        dx = (src.transform.c - ref.transform.c) / ref.res[0]
        dy = (src.transform.f - ref.transform.f) / ref.res[1]

        # Check if offsets are integers
        if abs(dx - round(dx)) < tol and abs(dy - round(dy)) < tol:
            return f"aligned (pixel offset: {round(dx)}, {round(dy)})"
        else:
            return f"misaligned (fractional offset: {dx:.3f}, {dy:.3f})"

if __name__=="__main__":
    cog_src_dir = Path("data/cogs/")
    cog_file_template = "LGI_%Y%m%d_SE.tif"
    nworkers = 22
    check_if_missing_dates = False
    check_cog_metadata_consistency = True
    #analysis_range = [datetime(1981,2,1), datetime(1982, 1, 31)]
    analysis_range = None

    ## get a sorted list of files and their times
    cog_times_paths = []
    for p in cog_src_dir.iterdir():
        try:
            t = datetime.strptime(p.name, cog_file_template)
        except ValueError:
            continue
        cog_times_paths.append((t,p))
    cog_dates,cog_paths = zip(*sorted(cog_times_paths, key=lambda tp:tp[0]))

    day_0 = cog_times_paths[0][0]
    day_f = cog_times_paths[-1][0]

    ## check for missing days. Currently not an issue.
    if check_if_missing_dates:
        all_days = [
            day_0 + timedelta(days=i)
            for i in range((day_f-day_0).days+1)
            ]
        missing_days = [d for d in all_days if d not in cog_dates]
        print(f"{missing_days = }")

    ## subset to within the specified date range
    if not analysis_range is None:
        cog_dates,cog_paths = map(list, zip(*filter(
            lambda dp:dp[0]>=analysis_range[0] and dp[0]<=analysis_range[1],
            zip(cog_dates, cog_paths)
            )))

    ## show that some metadata varies between files
    #'''
    cog_meta = extract_cog_metadata(cog_paths[0])
    if check_cog_metadata_consistency:
        all_cog_meta = []
        with mp.Pool(nworkers) as pool:
            for r in pool.imap_unordered(extract_cog_metadata, cog_paths[:-1]):
                all_cog_meta.append(r)
        for p,m in zip(cog_paths[:-1],all_cog_meta):
            valid,reason = compare_cog_metadata(cog_meta, m, thresh=1e-8)
            if valid:
                continue
            print(f"different meta {cog_paths[0].name} vs {p.name}:", reason)
            #print(f"Metadata for {cog_paths[0]}:\n{cog_meta}")
            #print(f"Metadata for {p}:\n{m}")
    #'''

    ## demonstrates that shapes and geographic bounds vary between files
    '''
    prev_meta = extract_cog_metadata(cog_paths[0])
    for p in cog_paths[1:]:
        cur_meta = extract_cog_metadata(p)
        if not compare_cog_metadata(prev_meta, cur_meta):
            print(f"New metadata: {p.name} " + \
                    f"w:{cur_meta['width']}, h:{cur_meta['height']}, " + \
                    f"lat0:{cur_meta['bounds']['bottom']:.3f}, " + \
                    f"latf:{cur_meta['bounds']['top']:.3f}, " + \
                    f"lon0:{cur_meta['bounds']['left']:.3f}, " + \
                    f"lonf:{cur_meta['bounds']['right']:.3f}")
        prev_meta = cur_meta
    '''

    ## demonstrates that transforms vary, so simple slicing method can't be
    ## used to reliably subgrid the data
    '''
    shared_bounds = extract_common_window(cog_paths)
    print(shared_bounds)
    for p in cog_paths:
        with rio.open(p) as src:
            wdw = rio.windows.from_bounds(
                *shared_bounds,
                transform=src.transform,
                )
            wdw = wdw.round_offsets().round_lengths()
            x = src.read(window=wdw)
            print(p.name, x.shape)
            print(src.transform)
            print()
    '''

    ## demonstrate that, fortunately, there is only one CRS
    '''
    crs_map = {}
    for p in cog_paths:
        with rio.open(p) as src:
            if (tmp_crs := src.crs) not in crs_map.keys():
                crs_map[tmp_crs] = [p]
            else:
                crs_map[tmp_crs].append(p)

    pprint(list(crs_map.keys()))
    '''

    ## check whether grid offsets account for the difference in transform
    ## this shows that alignment varies by ~1.15e-5 degrees, which will
    ## accumulate to about a 1km misalignment across the grid. Gotta resample.
    '''
    for p in cog_paths[1:]:
        print(p, check_cog_alignment(cog_paths[0], p))
    '''
