# relayglm_paper

Code to recreate figures from the paper [Dynamics of temporal integration in the lateral geniculate nucleus](https://www.eneuro.org/content/9/4/ENEURO.0088-22.2022.long). The figures can also be viewed [HERE](https://www.eneuro.org/content/9/4/ENEURO.0088-22.2022/tab-figures-data).

## Usage

* This code requires [Julia](https://julialang.org/), if you do not already have Julia installed you can download it from [https://julialang.org/downloads/](https://julialang.org/downloads/).
    * **NOTE**: this code has only been tested on Julia version 1.6.1, versions between 1.2.x and 1.6.1 *should* all work, they are, however, untested.
* Download or clone code into a suitable location (we will refer to the path to the directory holding the code as `<code_dir>`)
* Launch `julia`:

```julia
code_dir = "..." #location of the cloned repo
cd(code_dir)
```

### Install dependencies

Run the following (you only need to do this ***ONCE***)

```julia
# install dependencies (you only need to do this once)
include("./install_dependencies.jl")

# use of local Pkg environment / project is recommended, pass false to install
# deps to the home project
install_dependencies(true)
```

### Setup the local Pkg environment
Run the following code ***each time*** you launch a new Julia session (and want to recreate the figures):

```Julia
code_dir = "..." #location of the cloned repo
cd(code_dir)

# make sure Julia can find all the figure and utility modules (needed once per Julia session)
include("./setpath.jl")
```
### Note on run times
Estimated / tested run times are given within a comment above the call to the `collate_data()` function for each figure. Note that run times reflect estimates based on a single machine (Dell Precision T3610 w/ Intel Xenon E5-1620, 4 physical / 8 logical cores). Julia is given access to 8 threads (i.e. julia is launched as `julia-1.6.1 -t 8`) as is BLAS (i.e. `LinearAlgebra.BLAS.get_num_threads()` -> 8). Different configurations will likely result in different run times.

### Note on Figure5, Figure6_7 and FigureS2
The code referenced here does **not** actually run the lengthy cross-validation procedure that was used to produced the results depicted in Figures 5, 6 & 7, and S2 (aka 7-2). Instead, the results of the cross-validation are simply loaded from the `preprocessed_data` directory. To actually run the cross-validation (not recommended due to **very** long run time, on the order of several days) one would use the `HyperParamsCV` module (in the `utils` directory) and the `collate_data()` function therein. Once you have set up the local environment (see above) simply running

 ```julia
 using HyperParamsCV

 # probably don't actually run this...
 d = HyperParamsCV.collate_data()
 ```

 will run the cross-validation. Again, this takes days (even a week depending on the system) so I wouldn't recommended running it unless you know what you are doing.

The `HyperParamsCV.collate_data()` function takes two optional arguments, a vector of pair IDS and a directory in which to save log files. Omitting the first argument (the vector of ids) will result in all pairs being processed.

### Recreate the figures:

```Julia
# ============================================================================ #
# recreate figure 1
using Figure1

# use Pair 214 as our example as the paper does
# NOTE: tested run time is ~2 seconds
d1 = Figure1.collate_data(214)

# make the figure
Figure1.make_figure(d1, 214)

# ============================================================================ #
# recreate figure 2
using Figure2

# NOTE: tested run time is ~50 minutes
d2 = Figure2.collate_data()

# use Pair 208 as the example (same as paper does)
Figure2.make_figure(d2, 208)

# ============================================================================ #
# recreate figure 3
using Figure3

# NOTE: tested run time is ~7.5 minutes
d3 = Figure3.collate_data()

# use Pair 208 as the example (same as figure 2 and as paper does)
Figure3.make_figure(d3, 208)

# ============================================================================ #
# recreate figure 4
using Figure4

# NOTE: tested run time is ~14.5 minutes
d4 = Figure4.collate_data()

Figure4.make_figure(d4)
# ============================================================================ #
# recreate figure 5
using Figure5, JLD

# NOTE: tested run time is ~13 seconds
d5, _ = Figure5.collate_data()

Figure5.make_figure(d5)
# ============================================================================ #
# recreate figures 6 & 7
using Figure6_7

# NOTE: tested run time is ~4 seconds
d67 = Figure6_7.collate_data()

# NOTE: this will make figures 6 & 7 and an analogous figure that does not
# appear in the paper for the awake data set
Figure6_7.make_figure(d67)

# ============================================================================ #
# recreate figure 8
using Figure8

# we can specify the time window durations to use for RGC spike partitioning
# here we use 100 ms (or 0.1 sec) which is what Figure 8 in the paper uses
# NOTE: tested run time is ~6 minutes
d8 = Figure8.collate_data(twin=0.1)

Figure8.make_figure(d8)

# ============================================================================ #
# recreate figure 9
using Figure9

# NOTE: the data for Figures 6 and 7 are the same, so if you just ran the
# analysis for figure 6 you can just use the following:
d9 = d8

# # otherwise...
# NOTE: tested run time is ~6 minutes
# d9 = Figure9.collate_data()

Figure9.make_figure(d8)

# NOTE: you may need to reset the y-axis limits to see the blue and gold triangles

# ============================================================================ #
# recreate figure S1 aka Figure 7-1
using FigureS1

# NOTE: the data for Figures 6\7 and S1 are the same, so if you just ran the
# analysis for figures 6\7 you can just use the following:
ds1 = d67

# otherwise... (run time is ~1 second)
# ds1 = FigureS1.collate_data()

FigureS1.make_figure(ds1)

# ============================================================================ #
# recreate figure S2 aka Figure 7-2
using FigureS2

# NOTE: tested run time is ~12 seconds
ds2 = FigureS2.collate_data()

FigureS2.make_figure(ds2)

# ============================================================================ #
# recreate figure S3 aka Figure 8-1
using FigureS3

# NOTE: in the paper GLM models are simulated for 50 iterations, but that takes
# quite some time, here we'll just use 5 to speed things up
# NOTE: tested run time is ~24 minutes
ds3 = FigureS3.collate_data(niter=5)

FigureS3.make_figure(ds3)

# ============================================================================ #
# recreate figure S4 aka Figure 8-2
using FigureS4

# NOTE: tested run time is ~14 minutes
ds4 = FigureS4.collate_data()

FigureS4.make_figure(ds4)

```

## Dependencies

The non-registered dependencies on which this code relies are automatically installed by the script `install_dependencies.jl` (see instructions above). Their code can be found at:

* [RelayGLM.jl](https://github.com/scottiealexander/RelayGLM.jl.git)
* [PairsDB.jl](https://github.com/scottiealexander/PairsDB.jl.git)
* [WeyandDB.jl](https://github.com/scottiealexander/WeyandDB.jl.git)

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
