# Europe Forest Sensitivity to Seasonal Precipitation — Data & Downloads

This repository analyzes how European forests respond to **seasonal precipitation** using satellite products and reanalysis data.  
This README lists the **data sources, key specs, and official download links** so you can obtain the inputs used in the analysis.

> **Note:** Some portals require a free account (e.g., NASA Earthdata, Copernicus Climate Data Store).

---

## Data Sources

### 1) MODIS Terra EVI — **MOD13Q1 v6.1**
- **What it is:** 16-day composite **Enhanced Vegetation Index (EVI)** from MODIS Terra.
- **Coverage & cadence:** Global, **250 m**, 16-day; ~2000–present.
- **Use in this project:** Proxy for canopy greenness/structure.
- **Official links:**
  - Product page (LP DAAC): https://lpdaac.usgs.gov/products/mod13q1v061/
  - Direct access (LAADS DAAC): https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/MOD13Q1/

---

### 2) CSIF — **Contiguous Solar-Induced Fluorescence**
- **What it is:** A satellite-derived proxy of **SIF** that tracks photosynthetic activity.
- **Coverage & cadence:** Global, **0.05°**, 4-day; commonly used **2000–2016 (all-sky)**.
- **Use in this project:** Independent proxy for vegetation productivity.
- **Official links:**
  - Dataset (OSF project): https://osf.io/8xqy6/
  - Method/reference (Biogeosciences): https://bg.copernicus.org/articles/15/5779/2018/

---

### 3) ERA5-Land Monthly Means (C3S/ECMWF)
- **What it is:** Near-surface meteorology from the Copernicus Climate Data Store.
- **Variables used:** Total precipitation, 2-m temperature, dew-point temperature (for derived VPD, etc.).
- **Coverage & cadence:** Global land, **0.1°**, **monthly**; **1950–present** (rolling updates).
- **Official link:**
  - Dataset page (CDS): https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means

---

### 4) Copernicus HRL **Forest Type 2018**
- **What it is:** Pan-European **forest mask** with **broadleaved vs. coniferous** classes.
- **Coverage & resolution:** Europe; **10 m** (also available at **100 m** aggregated); **reference year 2018**.
- **Use in this project:** Define forest pixels and forest type classes.
- **Official links:**
  - Product page & downloads:  
    https://land.copernicus.eu/en/products/high-resolution-layer-forests-and-tree-cover/forest-type-2018-raster-10-m-100-m-europe-3-yearly
  - HRL Forests overview:  
    https://land.copernicus.eu/en/products/high-resolution-layer-forests-and-tree-cover

---

### 5) GEBCO Global DEM (GEBCO_2024 Grid)
- **What it is:** Global bathymetry/topography grid.
- **Resolution:** **15 arc-second** global grid.
- **Use in this project:** Topographic context and elevation-based filtering.
- **Official links:**
  - Download portal: https://download.gebco.net/
  - Product info: https://www.gebco.net/data-products-gridded-bathymetry-data/gebco2024-grid

---

## Access Notes

- **NASA LP DAAC / LAADS DAAC**: Requires a free **Earthdata Login** for direct downloads or programmatic access.
- **Copernicus Climate Data Store (CDS)**: Requires a free account and acceptance of the ERA5-Land license before downloads or API access.
- **Copernicus Land Monitoring Service (CLMS)**: Forest Type 2018 can be downloaded after accepting the product license.
- **Large volumes**: Consider using each provider’s APIs/CLI tools for bulk downloads.

---

## Attribution

If you use these datasets, please cite/acknowledge the providers as requested on each product page:

- **MODIS MOD13Q1 v6.1** — NASA LP DAAC / LAADS DAAC.  
- **CSIF** — Zhang et al. (2018), dataset hosted on OSF.  
- **ERA5-Land** — Copernicus Climate Change Service (C3S) / ECMWF.  
- **Copernicus HRL Forest Type 2018** — Copernicus Land Monitoring Service (CLMS) / EEA.  
- **GEBCO 2024 Grid** — IHO/IOC-UNESCO.

---
The CSV data we used could be downloaded from https://drive.google.com/drive/folders/1_4E6yP7Dpw5Cyy6sGuO5aMlKbqrR2tf3?usp=drive_link

*This README intentionally focuses on data provenance and download locations only.*
