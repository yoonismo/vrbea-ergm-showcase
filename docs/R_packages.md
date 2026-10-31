# R packages

The code in this repository calls R (`network`, `sna`, `ergm`) from Python through **rpy2**.
This page explains where those R packages should live and why.

Tested on macOS and Linux. Windows has not been tested; rpy2 itself can be
difficult to install on Windows.

---

## Warning: use a separate R library for this project

R packages that contain compiled code (`.so` shared libraries) are built against
the C library (**glibc**) of the environment that installs them. A package built
against a newer glibc cannot be loaded in an environment with an older one.

If RStudio, or any other R installation running in a **different container or
environment**, shares the same R library directory as this project, packages
installed from there may fail to load here with errors such as:

```
/usr/lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.38' not found
  (required by .../R_container/rlang/libs/rlang.so)
```

```
lazy-load database '.../rlang/R/rlang.rdb' is corrupt
internal error 1 in R_decompress1 with libdeflate
```

**Rule:** install the R packages for this project from the same environment
you run it in (the Jupyter / Python environment), and build them from source
(`type = "source"`). Do not let another R installation write into the same
library directory.

### How this happened (for reference)

On an HPC cluster, RStudio and Jupyter ran in two different Apptainer
containers with different glibc versions (an RStudio image based on a newer
Ubuntu, and a Python image with glibc 2.35). Both containers set `R_LIBS_USER`
to the same folder. Packages installed from RStudio were compiled against the
newer glibc and could not be loaded from Jupyter through rpy2.

The fix was to give each environment its own R library directory.

---

## Where to install the packages

### Laptop

Nothing special is needed if you only use R through this project. If you also
use R/RStudio separately on the same machine, keep this project's packages in
their own folder (see the helper below).

### HPC cluster

Home directories are often small, and scratch space is often purged
automatically (on UW Hyak, `/gscratch/scrubbed/` is cleared after 21 days of
inactivity). Use persistent, non-purged storage for the R library.

If you also use RStudio on the cluster, point it to a **different** folder.
For example, in an RStudio job script:

```bash
export R_LIBS_USER=${RSTUDIO_CWD}/R_studio      # RStudio only
```

and keep the Jupyter / rpy2 library elsewhere.

> Note: if a container's `.def` file sets `R_LIBS_USER` in its `%environment`
> section, that value can override `apptainer exec --env R_LIBS_USER=...`.
> Setting it inside the RStudio session script (as above), or using a variable
> name the container does not set, avoids this.

### Installing (from the Jupyter environment)

```python
import rpy2.robjects as ro
ro.r('''
install.packages(c("network", "sna", "ergm", "tictoc"),
                 lib   = "<your R library path>",
                 repos = "https://cloud.r-project.org",
                 type  = "source",
                 dependencies = c("Depends", "Imports", "LinkingTo"))
''')
```

Use `dependencies = c("Depends", "Imports", "LinkingTo")` rather than
`dependencies = TRUE`. `TRUE` also pulls in suggested packages such as `rgl`,
which need extra system libraries and can take a long time to compile.

Restart the Python kernel after installing, then check:

```python
import rpy2.robjects as ro
print(ro.r('.libPaths()'))
print(ro.r('library(network); library(sna); library(ergm); "OK"'))
```

---

## Optional helper: a project-specific R library

The code below keeps this project's R packages in their own folder, chosen
per operating system, and splits that folder by R version, platform and glibc
version. Packages built in one environment are then never loaded in another,
even if two environments point to the same root folder.

| Environment | Default location |
| --- | --- |
| `ERGM_SHOWCASE_R_LIB` set | `$ERGM_SHOWCASE_R_LIB/<tag>` |
| macOS | `~/Library/Application Support/ergm-showcase/R/<tag>` |
| Linux / HPC | `~/.local/share/ergm-showcase/R/<tag>` |

`<tag>` looks like `R4.4_x86_64-pc-linux-gnu_glibc-2.35`.

On an HPC cluster, set `ERGM_SHOWCASE_R_LIB` to persistent storage before running, for
example in a job script:

```bash
apptainer exec --env ERGM_SHOWCASE_R_LIB=/path/to/persistent/storage/ergm_showcase_R ...
```

```python
import os
import sys
import platform
import rpy2.robjects as ro


def _default_r_lib_root():
    """Per-OS user data directory for this project's R packages."""
    if sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    elif os.name == "nt":
        base = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
    else:
        base = os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))
    return os.path.join(base, "ergm-showcase", "R")


def _r_env_tag():
    """R version + platform + glibc. Packages built in one environment
    are never loaded from another, even if the root folder is shared."""
    r_ver = ro.r('paste(R.version$major, '
                 'strsplit(R.version$minor, "[.]")[[1]][1], sep=".")')[0]
    arch = ro.r('R.version$platform')[0]
    parts = [f"R{r_ver}", arch]
    if sys.platform.startswith("linux"):
        libc, ver = platform.libc_ver()
        if libc:
            parts.append(f"{libc}-{ver}")
    return "_".join(parts)


def use_project_r_library():
    """Put this project's own R library first on .libPaths()."""
    root = os.environ.get("ERGM_SHOWCASE_R_LIB", _default_r_lib_root())
    lib = os.path.join(root, _r_env_tag())
    os.makedirs(lib, exist_ok=True)
    ro.r(f'.libPaths(c("{lib}", .libPaths()))')

    if "SLURM_JOB_ID" in os.environ and "ERGM_SHOWCASE_R_LIB" not in os.environ:
        print("[ergm-showcase] Running under SLURM. Home quota may be small; "
              "set ERGM_SHOWCASE_R_LIB to a persistent (non-purged) storage path.")
    return lib


def ensure_r_packages(pkgs=("network", "sna", "ergm", "tictoc")):
    """Install only the missing packages, from source, into the project library."""
    lib = use_project_r_library()
    missing = [p for p in pkgs
               if not ro.r(f'requireNamespace("{p}", quietly=TRUE)')[0]]
    if missing:
        print(f"[ergm-showcase] Installing R packages into {lib}: {missing}")
        pkg_vec = ", ".join(f'"{p}"' for p in missing)
        ro.r(f'''install.packages(c({pkg_vec}), lib = "{lib}",
                 repos = "https://cloud.r-project.org", type = "source",
                 dependencies = c("Depends", "Imports", "LinkingTo"))''')
    return lib
```

Call `ensure_r_packages()` once before importing the R packages with
`importr(...)`. The first run compiles from source and can take 10 to 30
minutes; later runs skip installed packages.