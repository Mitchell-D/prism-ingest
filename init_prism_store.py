"""
Script for constructing a zarr store of temporally-continuous daily PRISM data
based on a combination of geotiff files stored locally and files acquired from
the PRISM database.

The zarr store tracks the processing iteration as an integer 1-8 in accordance
with the PRISM standard:
https://prism.oregonstate.edu/documents/PRISM_downloads_web_service.pdf
"""
import numpy as np
import json
import zarr
zarr.config.set({"async.concurrency":128})
import rasterio as rio
import multiprocessing as mp
import io
import requests
import zipfile
from pprint import pprint
from rasterio.crs import CRS
from rasterio.warp import reproject,Resampling
from rasterio.transform import Affine
from datetime import datetime,date,timedelta
from pathlib import Path

from prism_utils import load_local_prism_day,download_prism_day
from prism_utils import compare_cog_metadata

def get_latlon_grid(target_profile, collapse_to_1d=False):
    """
    Given the reference profile dict,

    :@param target_profile: dict containing a transform, CRS, height, and
        width that the provided COG should be resampled to match.
    :@param collapse_to_1d: If True, 2d lat and lon arrays will be collapsed
        to 1d if they are equirectangular
    """
    transform = Affine(*target_profile["transform"])
    print(transform)
    height = target_profile["height"]
    width = target_profile["width"]

    ## create row/col index grids
    rows, cols = np.meshgrid(
        np.arange(height),
        np.arange(width),
        indexing="ij"
        )

    ## convert pixel centers to geographic coordinates
    xs, ys = rio.transform.xy(transform, rows, cols)

    lon = np.array(xs).reshape((height, width))
    lat = np.array(ys).reshape((height, width))

    if collapse_to_1d:
        assert np.all(np.isclose(lat, lat[:, [0]])), \
                "lat values not consistent on axis 1"
        assert np.all(np.isclose(lon, lon[0, :])), \
                "lon values not consistent on axis 0"
        lat = lat[:,0]
        lon = lon[0,:]

    return lat, lon

def get_prism_day(yyyymmdd:str, tif_path:Path|None,
        region="us", res="4km", feat="ppt", get_grid_count=True,
        fail_on_error=True, **passthrough_args):
    """
    high-level method for acquiring a PRISM file either from the local file
    system or from the remote database. If tif_path is not None, the file
    is assumed to exist, and the data/metadata is extracted locally. Otherwise,
    the file with the provided date, region, resolution, and data feature is
    acqured and extracted into the same format.

    If there's an error with the download, rather than raising it, optionally
    just return None and let the parent process handle gaps.

    :@return: dict containing fields "data":p.ndarray, "crs":str,
        "transform":list, "width":int, "height":int, "grid_count":int, where
        the data array sets values masked with -9999. to nan.
    """
    if tif_path is None:
        try:
            return download_prism_day(
                    yyyymmdd=yyyymmdd,
                    region=region,
                    res=res,
                    feat=feat,
                    get_grid_count=get_grid_count,
                    )
        except ValueError as e:
            if fail_on_error:
                raise e
            print(e)
            return None
    else:
        ## use default assumed grid count of 8
        return load_local_prism_day(tif_path)

def mp_get_prism_day(args):
    return args,get_prism_day(**args)

if __name__=="__main__":
    prism_cog_src_dirs = {
        "ppt":Path("/rgroup/convergence/prismdata/NonBilFiles/ppt"),
        "tmax":Path("/rgroup/convergence/prismdata/NonBilFiles/tmax"),
        "tmin":Path("/rgroup/convergence/prismdata/NonBilFiles/tmin"),
        "soltotal":Path("/rgroup/convergence/prismdata/NonBilFiles/soltotal"),
        "tdmean":Path("/rgroup/convergence/prismdata/NonBilFiles/tdmean"),
        }
    local_cog_file_template = "prism_{feat}_us_25m_{dstr}.tif"
    #out_zarr_path = Path("data/zarr/prism-pipeline.zarr")
    out_zarr_path = Path(
        "/rgroup/watershed/COMMON_DATASETS/prism-pipeline-2.zarr")
    nworkers = 18

    ## bounds on file dates to ingest; None implies no bounds
    #start_day = datetime(2024,5,28)
    #start_day = datetime(1995,1,1)
    start_day = datetime(1982,1,1)
    #end_day = None
    end_day = datetime(2025,9,30)
    #end_day = datetime(1995, 12, 31)

    get_feats = ["ppt", "tmax", "tmin", "soltotal", "tdmean"]

    ## bounds on file dates to acquire from prism_cog_src_dirs. Any file dates
    ## outside this range will be downloaded from the PRISM database. File
    ## dates inside this range are ASSUMED to have processing level 8.
    local_start_day = datetime(1982, 1, 1)
    local_end_day = datetime(2025, 9, 30)

    raster_chunks = (90, 3, 69, 281) ## 21 MB chunks
    raster_shards = (180, 3, 621, 1405) ## 1.9 GB shards
    raster_dtype = np.float32

    ## determines how often
    write_frequency = 90

    download_if_local_missing = True

    """ -----( end standard configuration )----- """

    if out_zarr_path.exists():
        raise ValueError(f"Zarr store exists: {out_zarr_path.as_posix()}")

    ## set boundless defaults for start/end day
    if start_day is None:
        start_day = datetime(1,1,1)
    if end_day is None:
        end_day = datetime.now()
    assert start_day <= end_day, (start_day, end_day)

    ## develop a list of all needed dates and partition it into dates that
    ## will be acquired locally and dates that will be downloaded
    all_dates = [
        (start_day + timedelta(days=i))
        for i in range((end_day-start_day).days + 1)
        ]
    ## populated whether or not a day is local
    local_paths = {fk:[] for fk in get_feats}
    if not local_start_day is None and not local_end_day is None:
        assert local_start_day < local_end_day
        for d in all_dates:
            if local_start_day <= d <= local_end_day:
                for fk in get_feats:
                    p = prism_cog_src_dirs[fk].joinpath(
                            local_cog_file_template.format(
                                dstr=d.strftime("%Y%m%d"),
                                feat=fk,
                                )
                            )
                    if p.exists():
                        local_paths[fk].append(p)
                    else:
                        if download_if_local_missing:
                            local_paths[fk].append(None)
                        else:
                            raise ValueError(
                                f"Missing required local path: {p.as_posix()}")
            else:
                for fk in get_feats:
                    local_paths[fk].append(None)
    else:
        for fk in get_feats:
            local_paths[fk] = [None] * len(all_dates)

    args = [{
        "yyyymmdd":d.strftime("%Y%m%d"), "tif_path":lp,
        "region":"us", "res":"4km", "feat":fk,
        "get_grid_count":True, "fail_on_error":True,
        "array_index":ix, ## pass-through argument
        }
        for fk in get_feats
        for ix,(d,lp) in enumerate(zip(all_dates, local_paths[fk]))
        ]

    zstor = zarr.storage.LocalStore(out_zarr_path)
    zgrp = zarr.open(zstor, path="/", mode="w")

    missing = []
    ref_geom = None
    ref_geom_date = None
    dstrs = [None] * len(all_dates)
    gcints = {fk:[-1] * len(all_dates) for fk in get_feats}
    prev_write_stop = {fk:0 for fk in get_feats}
    buffer = {fk:[] for fk in get_feats}
    ctr = {fk:0 for fk in get_feats}
    with mp.Pool(nworkers) as pool:
        for a,r in pool.imap(mp_get_prism_day,args):
            ix = a["array_index"]
            fk = a["feat"]
            fix = get_feats.index(fk)
            ctr[fk] += 1

            ## only save the date if it hasn't been loaded already
            if dstrs[ix] == None:
                dstrs[ix] = "-".join([
                    a["yyyymmdd"][:4], a["yyyymmdd"][4:6], a["yyyymmdd"][6:8]
                    ])
            if (not r is None) and (ref_geom is None):
                ashape = (len(all_dates),len(get_feats),r["height"],r["width"])
                print(f"creating array with shape {ashape}")
                zgrp.create_array(
                    "prism",
                    shape=ashape,
                    chunks=raster_chunks,
                    shards=raster_shards,
                    dimension_names=[
                        "date", "variable", "latitude", "longitude"
                        ],
                    dtype=raster_dtype,
                    )
                ## create an array for the daily time coordinate data.
                ## stored as a string that parses to np.datetime
                zgrp.create_array(
                    "date",
                    shape=(len(all_dates),),
                    dimension_names=["date"],
                    dtype="M8[D]",
                    )
                zgrp.create_array(
                    "gcount",
                    shape=(len(all_dates), len(get_feats)),
                    dimension_names=["date", "variable"],
                    dtype=np.int8
                    )
                zgrp.create_array(
                    "lat",
                    shape=(r["height"],r["width"]),
                    dimension_names=["latitude", "longitude"],
                    dtype=np.float64,
                    )
                zgrp.create_array(
                    "lon",
                    shape=(r["height"],r["width"]),
                    dimension_names=["latitude", "longitude"],
                    dtype=np.float64,
                    )
                zgrp.create_array(
                    "latitude",
                    shape=(r["height"],),
                    dimension_names=["latitude"],
                    dtype=np.float64,
                    )
                zgrp.create_array(
                    "longitude",
                    shape=(r["width"],),
                    dimension_names=["longitude"],
                    dtype=np.float64,
                    )
                lat1d,lon1d = get_latlon_grid(r, collapse_to_1d=True)
                lat,lon = np.meshgrid(lat1d, lon1d, indexing="ij")
                zgrp["lat"][...] = lat
                zgrp["lon"][...] = lon
                zgrp["latitude"][...] = lat1d
                zgrp["longitude"][...] = lon1d

                ## keep this geometry dict for comparison to future ones.
                ref_geom = r
                ref_geom_date = a["yyyymmdd"]

                ## make sure the grid is equirectangular
                dlat = np.diff(lat[:,0])
                dlon = np.diff(lon[0,:])
                assert np.all(np.isclose(dlat[0], dlat[1:]))
                assert np.all(np.isclose(dlon[0], dlon[1:]))

                ## update attributes to include geometry information
                ref_crs = CRS.from_string(r["crs"])
                zgrp.attrs.update({
                    "grid":{
                        "min_lat":np.amin(lat),
                        "max_lat":np.amax(lat),
                        "min_lon":np.amin(lon),
                        "max_lon":np.amax(lon),
                        "diff_lat":dlat[0],
                        "diff_lon":dlon[0],
                        },
                    "geom":{
                        "crs":ref_crs.to_wkt(),
                        "crs_basic":str(ref_crs),
                        "transform":list(ref_geom["transform"]),
                        "width":ref_geom["width"],
                        "height":ref_geom["height"],
                        "bounds":ref_geom["bounds"],
                        "resolution":ref_geom["resolution"],
                        },
                    "variable":get_feats,
                    })
            if r is None:
                missing.append((fk,ix,a["yyyymmdd"]))
                buffer[fk].append(None)
            else:
                gcints[fk][ix] = r["grid_count"]
                ## make sure all new geometries align with the reference geom
                valid,reason = compare_cog_metadata(r, ref_geom, thresh=1e-7)
                if not valid:
                    print(f"different meta {ref_geom_date} vs {r['yyyymmdd']}",
                        reason)
                buffer[fk].append(r["data"])

            ## periodically write to the zarr store
            if ((ctr[fk]%write_frequency)==0) or (ctr[fk]==len(all_dates)):
                ## replace missing time steps with full nan arrays
                for i in range(len(buffer[fk])):
                    if buffer[fk][i] is None:
                        if ref_geom is None:
                            raise ValueError(
                                "No files successfully loaded before the"
                                "first batch write!!"
                                )
                        else:
                            print(f"Failed to load {fk} {dstrs[ix]}")
                        buffer[fk][i] = np.full(
                            (ref_geom["height"], ref_geom["width"]),
                            np.nan,
                            dtype=raster_dtype,
                            )
                buf_slc = slice(
                    prev_write_stop[fk],
                    prev_write_stop[fk]+len(buffer[fk])
                    )
                print(f"writing {fk} slice ({buf_slc.start}, {buf_slc.stop})")
                ## write the raster data
                zgrp["prism"][buf_slc,fix,:,:] = np.stack(buffer[fk], axis=0)
                ## load the grid count and date arrays
                zgrp["gcount"][buf_slc,fix] = np.asarray(gcints[fk][buf_slc])
                zgrp["date"][buf_slc] = np.array(dstrs[buf_slc], dtype="M8[D]")
                prev_write_stop[fk] = buf_slc.stop
                buffer[fk] = []
            print(f"loaded {a['yyyymmdd']}")
    if len(missing) > 0:
        print("missing:", missing)
    else:
        print(f"no missing timestesp")
