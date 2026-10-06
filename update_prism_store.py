"""

"""
import numpy as np
import json
import zarr
zarr.config.set({"async.concurrency":128})
import multiprocessing as mp
from pprint import pprint
import datetime as dt
from pathlib import Path

from prism_utils import mp_download_prism_day,mp_get_prism_grid_count
from check_cog_alignment import compare_cog_metadata

if __name__=="__main__":
    #out_zarr_path = Path("data/zarr/prism-pipeline.zarr")
    out_zarr_path = Path(
        "/rgroup/watershed/COMMON_DATASETS/prism-pipeline.zarr")
    #nworkers = 18
    nworkers = 12

    ## bounds on file dates to update or ingest
    end_day = dt.datetime.now()
    start_day = end_day - dt.timedelta(days=12*31)
    #start_day = dt.datetime(2025, 6, 1)
    #end_day = dt.datetime(2026, 6, 1)
    max_processing_level = 8

    """ -----( end standard configuration )----- """

    end_day = dt.date(end_day.year, end_day.month, end_day.day)
    start_day = dt.date(start_day.year, start_day.month, start_day.day)

    all_dates = [
        (start_day + dt.timedelta(days=i)).strftime("%Y%m%d")
        for i in range((end_day-start_day).days + 1)
        ]
    if start_day < dt.date.fromisoformat(all_dates[0]):
        raise ValueError("prepending not yet supported")

    zgrp = zarr.open(out_zarr_path, mode="r")
    get_feats = zgrp.attrs["variable"]
    dates = zgrp["date"][...].astype(dt.datetime).tolist()
    dates_str = [d.strftime("%Y%m%d") for d in dates]
    gcounts = zgrp["gcount"][...].tolist()

    try:
        ix0 = dates.index(start_day)
    except:
        ix0 = 0
    try:
        ixf = dates.index(end_day)+1
    except:
        ixf = len(dates)

    ## identify the old grid counts of in-range dates that are already present
    check_dates = {
        d.strftime("%Y%m%d"):gc
        for d,gc in zip(dates[ix0:ixf], gcounts[ix0:ixf])
        #if any(g < max_processing_level for g in gc)
        }

    ## update with dates that aren't currently present, and remove dates
    ## that already exceed the maximum processing level
    missing_dates = []
    for d in all_dates:
        old_gc = check_dates.get(d)
        if old_gc==None:
            check_dates[d] = [-1]*len(get_feats)
            missing_dates.append(d)
        elif all(g >= max_processing_level for g in old_gc):
            del check_dates[d]

    args = [
        {"yyyymmdd":dstr, "region":"us", "feat":fk}
        for dstr in check_dates.keys()
        for fk in get_feats
        ]

    ## get needed files mapped to their now-available grid count
    download_dates = {}
    with mp.Pool(nworkers) as pool:
        for a,r in pool.imap_unordered(mp_get_prism_grid_count, args):
            fix = get_feats.index(a["feat"])
            if check_dates[a["yyyymmdd"]][fix] < r:
                #print(check_dates[a["yyyymmdd"]], r)
                if not a["yyyymmdd"] in download_dates.keys():
                    download_dates[a["yyyymmdd"]] = [-1] * len(get_feats)
                download_dates[a["yyyymmdd"]][fix] = r

    print("downloading dates:")
    pprint(download_dates)

    if len(download_dates.keys()) == 0:
        exit(0)

    ## enumerate days from after final file day to final download day
    dd0 = dates[-1] + dt.timedelta(days=1)
    ddf = dt.date.fromisoformat(list(sorted(download_dates.keys()))[-1])
    new_days = [
        (dd0 + dt.timedelta(days=i)).strftime("%Y%m%d")
        for i in range((ddf-dd0).days + 1)
        ]
    all_dt64s = np.array(
        ["-".join([d[:4], d[4:6], d[6:8]]) for d in dates_str+new_days],
        dtype="M8[D]",
        )

    all_gcounts = []
    download_ixs = {}
    for i,d in enumerate(dates_str):
        if d in download_dates.keys():
            all_gcounts.append(download_dates[d])
            download_ixs[d] = i
        else:
            all_gcounts.append(gcounts[i])
    for i,d in enumerate(new_days, len(dates_str)):
        all_gcounts.append(download_dates[d])
        download_ixs[d] = i

    all_gcounts = np.array(all_gcounts, dtype=np.int8)

    ## resize the zarr array to fit the new dates
    nd_old,nfeat,nlat,nlon = zgrp["prism"].shape
    nd_new = nd_old + len(new_days)
    assert all_dt64s.size == nd_new, (all_dt64s.size, nd_new)
    assert len(all_gcounts) == nd_new, (len(all_gcounts), nd_new)

    ## open the zarr file and update date/gcount
    zgrp = zarr.open(out_zarr_path, mode="a")
    zgrp["gcount"].resize((nd_new, len(get_feats)))
    zgrp["date"].resize(nd_new)
    zgrp["prism"].resize((nd_new, nfeat, nlat, nlon))

    ##
    ref_geom = None
    args = [
        {"yyyymmdd":d, "region":"us", "feat":fk}
        for d in list(download_ixs.keys())
        for fk in get_feats
        ]
    with mp.Pool(nworkers) as pool:
        for a,r in pool.imap_unordered(mp_download_prism_day, args):
            dix = download_ixs[a["yyyymmdd"]]
            fix = get_feats.index(a["feat"])
            if ref_geom is None:
                ref_geom = {**r, "data":None}
                #print(f"ref_geom:", ref_geom)
            else:
                valid,reason = compare_cog_metadata(r, ref_geom, thresh=1e-6)
                if not valid:
                    print(f"different reference geom vs {r['yyyymmdd']}",
                        reason)
            zgrp["prism"][dix,fix] = r["data"]
    zgrp["gcount"][...] = all_gcounts
    zgrp["date"][...] = all_dt64s
    print("finished updating prism data")
