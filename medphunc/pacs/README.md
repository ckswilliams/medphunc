# medphunc.pacs

DICOM PACS query/retrieve utilities built on [pynetdicom](https://github.com/pydicom/pynetdicom), with optional [Orthanc](https://www.orthanc-server.com/) REST API support.


## Overview

In order to get the most out of these scripts, you need to run them on a computer which has a direct connection to PACS, and ideally an instance of Orthanc installed. Without this, the scripts will not be useful.

Furthermore, you'll need to configure the Orthanc REST service.


## Contents

| Module | Purpose |
|---|---|
| `pacsify` | Core C-FIND/C-MOVE/C-STORE operations; `AEInfo`, `SearchSet`, `RDSR` classes |
| `sorting` | Filter and rank query result DataFrames |
| `thanks` | Orthanc REST API layer (requires `pyorthanc`) |

---

## Configuration

Create a JSON file with your DICOM network details and point the environment variable `MEDPHUNC-PACSCONFIG` at it. The default location is at /path/to/medphunc/pacs/aeinfo.json

```json
{
  "dicom": {
    "physicspc": { "aet": "MEDPHYS01", "address": "192.168.1.10", "port": 104 },
    "pacs":      { "aet": "PACS",      "address": "192.168.1.20", "port": 104 },
    "default":   { "me": "physicspc", "remote": "pacs" }
  },
  "orthanc": {
    "local": { "url": "http://192.168.1.10:8042", "username": "admin", "password": "secret" },
    "default": "local"
  }
}
```

Set the environment variable before running Python (or in a `.env` file):

```
MEDPHUNC-PACSCONFIG=C:\path\to\aeinfo.json
```



The `default` block maps the `"me"` alias (your local AE) and `"remote"` alias (the PACS to query) to named entries. These aliases are used automatically by most functions.

---

## Quick-start

```python
from medphunc.pacs import pacsify as pi

# Verify the connection
pi.do_ping()

```

---

## AEInfo – managing DICOM nodes

```python
from medphunc.pacs.pacsify import AEInfo

# Load the node named "pacs" from config
remote = AEInfo(name="pacs")

# Load by role alias ("me" = local, "remote" = PACS)
me = AEInfo(default="me")

# Override explicitly
me.set_ae_info(aet="MEDPHYS01", address="192.168.1.10", port=104)

# Interactive CLI selection
remote.interactive_set_ae_from_saved()

print(remote)  # AET / address / port
```

---

## SearchSet – building and running queries

`SearchSet` is the main query class. It inherits from `pydicom.Dataset` and adds `.find()`, `.move()`, and drill-down navigation.

### Study-level find

```python
from medphunc.pacs.pacsify import SearchSet

# Query by accession number at study level
ss = SearchSet(query_level="study", AccessionNumber="ACC123456")
results = ss.find()  # returns pd.DataFrame
print(results[["AccessionNumber", "StudyDate", "StudyDescription"]])
```

### Series-level find from a study

```python
# Build from known identifiers
ss = SearchSet.from_study_uid_or_accession(accession_number="ACC123456")
results = ss.find()
print(results[["SeriesDescription", "Modality", "NumberOfSeriesRelatedInstances"]])
```

### Drilling down

```python
# Find all series, then drill down to instance level for row 0
ss.find()
instance_ss = ss.drill_down(i=0, find=True)
print(instance_ss.result)

# Drill down all series at once and merge into a single DataFrame
ss.drill_all(find=True)
print(ss.drill_merge)
```

### Moving a study or series

```python
# Move the entire study
ss = SearchSet(query_level="study", AccessionNumber="ACC123456")
ss.find()
ss.move()

# Move one instance per series (useful to get image headers quickly)
ss.move_one_instance_all_series()
```

### Date-range search

```python
from medphunc.pacs.pacsify import make_daterange
from datetime import datetime

date_range = make_daterange(datetime(2024, 3, 1), window=7)  # "20240301-20240307"

ss = SearchSet(query_level="study", StudyDate=date_range, Modality="CT")
ss.find()
```

### Fuzzy study lookup

```python
from medphunc.pacs.pacsify import study_from_patient_and_fuzzy_date
from datetime import date

result = study_from_patient_and_fuzzy_date(
    patient_id="PT12345",
    nominal_study_date=date(2024, 6, 15),
    modality="CT",
)
if result:
    print(result["study_instance_uid"], "found at delta", result["delta_found"])
```

---

## RDSR retrieval

`RDSR` is a `SearchSet` subclass that filters for Radiation Dose Structured Reports.

```python
from medphunc.pacs.pacsify import RDSR

# Find and move all RDSRs for a given accession
rdsr = RDSR.from_study_uid_or_accession(accession_number="ACC123456")
rdsr.move_rdsrs()

# Alternatively, iterate and move individually
for item in rdsr.find_rdsrs():
    item.move()
```

---

## Sorting and filtering results

`medphunc.pacs.sorting` provides helpers for narrowing query result DataFrames.

```python
from medphunc.pacs import sorting

# Filter results by date proximity
close = sorting.best_result_match(
    ss.result,
    reference_date="20240615",
    date_window=3,
    study_description_search_terms=["chest", "thorax"],
)

# Progressive keyword filter (skips terms that return zero results)
matches = sorting.search(
    ss.result,
    search_terms=["abdomen", "pelvis"],
    column="SeriesDescription",
    negation=False,
    search_type="str",
    search_strictness="filter",
)

# Find the best axial CT series from a series-level SearchSet
axial_idx = sorting.get_axial_index(ss)

# Get first and last axial slices (returns pydicom Datasets)
first, last = sorting.get_first_last_axial_slices(ss)
```

---

## Storing a DICOM object

```python
import pydicom

ds = pydicom.dcmread("path/to/file.dcm")
pi.do_store(ds)
```

If the SOP class is not yet registered in the association context, `do_store` will automatically add it and retry.

---

## Orthanc REST API (`thanks` module)

The `thanks` module wraps [pyorthanc](https://github.com/gacou54/pyorthanc) and adds DICOM-level helper methods.

Configure using the `orthanc` section of the JSON config (see above). Override the default instance with the `MEDPHUNC-ORTHANCDEFAULT` environment variable.

---



## Environment variables

| Variable | Purpose |
|---|---|
| `MEDPHUNC-PACSCONFIG` | Path to the JSON network config file |
| `MEDPHUNC-ORTHANCDEFAULT` | Named Orthanc entry to use as default |
